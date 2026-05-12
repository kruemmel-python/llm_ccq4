from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import math
import os
import random
import sys
import time
from pathlib import Path

try:
    from .forward import GemmaForwardRuntime
except ModuleNotFoundError as exc:
    if exc.name != "numpy":
        raise
    from .forward_pure import GemmaForwardRuntime

from .matvec import GpuCcq4Session
from .tokenizer import GemmaBpeTokenizer


def flush_native_stdio() -> None:
    for dll_name in (None, "msvcrt", "ucrtbase"):
        try:
            crt = ctypes.CDLL(dll_name) if dll_name else ctypes.CDLL(None)
            crt.fflush.argtypes = [ctypes.c_void_p]
            crt.fflush.restype = ctypes.c_int
            crt.fflush(None)
        except Exception:
            pass


class DriverOutputSilencer:
    def __init__(self, enabled: bool, log_path: str | Path):
        self.enabled = enabled
        self.log_path = Path(log_path)
        self._stack = None
        self._log = None
        self._saved_stdout_fd = None
        self._saved_stderr_fd = None

    def __enter__(self):
        if not self.enabled:
            return self
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout.flush()
        sys.stderr.flush()
        flush_native_stdio()
        self._log = self.log_path.open("a", encoding="utf-8", buffering=1)
        self._saved_stdout_fd = os.dup(1)
        self._saved_stderr_fd = os.dup(2)
        os.dup2(self._log.fileno(), 1)
        os.dup2(self._log.fileno(), 2)
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(contextlib.redirect_stdout(self._log))
        self._stack.enter_context(contextlib.redirect_stderr(self._log))
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return False
        sys.stdout.flush()
        sys.stderr.flush()
        flush_native_stdio()
        if self._stack is not None:
            self._stack.close()
        if self._saved_stdout_fd is not None:
            os.dup2(self._saved_stdout_fd, 1)
            os.close(self._saved_stdout_fd)
        if self._saved_stderr_fd is not None:
            os.dup2(self._saved_stderr_fd, 2)
            os.close(self._saved_stderr_fd)
        if self._log is not None:
            self._log.close()
        return False


def parse_prompt_ids(prompt: str, tokenizer: GemmaBpeTokenizer, add_bos: bool) -> list[int]:
    stripped = prompt.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            values = json.loads(stripped)
            if isinstance(values, list) and all(isinstance(v, int) for v in values):
                return [int(v) for v in values]
        except json.JSONDecodeError:
            pass
    return tokenizer.encode(prompt, add_bos=add_bos)


def compose_prompt(system_prompt: str | None, prompt: str) -> str:
    if not system_prompt:
        return prompt
    return (
        "<system>\n"
        f"{system_prompt.strip()}\n"
        "</system>\n"
        "<user>\n"
        f"{prompt.strip()}\n"
        "</user>\n"
        "<assistant>\n"
    )


def vector_norm(values) -> float:
    return sum(float(v) * float(v) for v in values) ** 0.5


def emotion_sampling_profile(mode: str) -> dict[str, float]:
    profiles = {
        "precise": {"temperature": 0.65, "top_p": 0.75, "repetition_penalty": 1.08},
        "calm": {"temperature": 0.85, "top_p": 0.90, "repetition_penalty": 1.05},
        "creative": {"temperature": 1.15, "top_p": 0.96, "repetition_penalty": 1.02},
        "dream": {"temperature": 1.35, "top_p": 0.985, "repetition_penalty": 1.0},
    }
    if mode not in profiles:
        raise ValueError(f"unknown emotion mode {mode!r}; valid: {', '.join(sorted(profiles))}")
    return profiles[mode]


def banned_generation_token_ids(tokenizer: GemmaBpeTokenizer) -> set[int]:
    banned: set[int] = set()
    for token_id, token in tokenizer.id_to_token.items():
        is_angle_special = token.startswith("<") and token.endswith(">") and not token.startswith("<0x")
        is_bracket_special = token.startswith("[") and token.endswith("]")
        if (is_angle_special or is_bracket_special) and int(token_id) != tokenizer.eos_id:
            banned.add(int(token_id))
    return banned


def blocked_ngram_token_ids(token_history: list[int], ngram_size: int) -> set[int]:
    if ngram_size <= 1 or len(token_history) < ngram_size - 1:
        return set()
    prefix = tuple(token_history[-(ngram_size - 1):])
    blocked: set[int] = set()
    for index in range(0, len(token_history) - ngram_size + 1):
        ngram = tuple(token_history[index:index + ngram_size])
        if ngram[:-1] == prefix:
            blocked.add(int(ngram[-1]))
    return blocked


def recent_token_ids(token_history: list[int], window: int) -> set[int]:
    if window <= 0:
        return set()
    return {int(token_id) for token_id in token_history[-window:]}


def apply_repetition_penalty(logits: list[dict], token_history: list[int], penalty: float) -> list[dict]:
    if penalty <= 1.0 or not token_history:
        return [dict(item) for item in logits]
    seen = set(token_history)
    adjusted = []
    for item in logits:
        token_id = int(item["token_id"])
        logit = float(item["logit"])
        if token_id in seen:
            logit = logit / penalty if logit > 0.0 else logit * penalty
        adjusted.append({"token_id": token_id, "logit": logit})
    return sorted(adjusted, key=lambda x: float(x["logit"]), reverse=True)


def sample_next_token(
    logits: list[dict],
    token_history: list[int],
    rng: random.Random,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    banned_token_ids: set[int] | None = None,
    no_repeat_ngram_size: int = 0,
    no_repeat_window: int = 0,
) -> dict:
    if not logits:
        raise ValueError("cannot sample from empty logits")
    adjusted = apply_repetition_penalty(logits, token_history, repetition_penalty)
    blocked_ids = set(banned_token_ids or set())
    blocked_ids.update(blocked_ngram_token_ids(token_history, no_repeat_ngram_size))
    blocked_ids.update(recent_token_ids(token_history, no_repeat_window))
    if blocked_ids:
        filtered = [item for item in adjusted if int(item["token_id"]) not in blocked_ids]
        if filtered:
            adjusted = filtered
    adjusted = sorted(adjusted, key=lambda x: float(x["logit"]), reverse=True)
    if temperature <= 0.0:
        selected = adjusted[0]
        return {**selected, "probability": 1.0, "sampling": "greedy"}

    max_logit = max(float(item["logit"]) for item in adjusted)
    probs = []
    denom = 0.0
    for item in adjusted:
        p = math.exp((float(item["logit"]) - max_logit) / temperature)
        probs.append(p)
        denom += p
    ranked = []
    cumulative = 0.0
    for item, p in zip(adjusted, probs):
        prob = p / denom if denom > 0.0 else 0.0
        cumulative += prob
        ranked.append({**item, "probability": prob})
        if cumulative >= top_p:
            break
    total = sum(float(item["probability"]) for item in ranked) or 1.0
    threshold = rng.random()
    running = 0.0
    for item in ranked:
        running += float(item["probability"]) / total
        if threshold <= running:
            return {**item, "sampling": "sample"}
    return {**ranked[-1], "sampling": "sample"}


def generate(
    runtime: GemmaForwardRuntime,
    tokenizer: GemmaBpeTokenizer,
    prompt_ids: list[int],
    max_new_tokens: int,
    max_layers: int,
    run_mlp: bool,
    top_k: int,
    vocab_limit: int | None,
    resident: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int,
    stop_eos: bool,
    no_repeat_ngram_size: int,
    no_repeat_window: int,
    preload_resident_layers: bool,
) -> dict:
    start = time.time()
    all_ids = list(prompt_ids)
    steps = []
    layer_count = min(max_layers, runtime.config.num_layers)
    prefill_steps = []
    last_logits = []
    rng = random.Random(seed)
    banned_ids = banned_generation_token_ids(tokenizer)

    def run_token(session: GpuCcq4Session, token_id: int, position: int) -> dict:
        x = runtime.embedding(token_id)
        layer_summaries = []
        for layer_index in range(layer_count):
            result = runtime.layer(session, x, layer_index, position, run_mlp, resident=resident)
            x = result["hidden"]
            layer_summaries.append({
                "layer": layer_index,
                "hidden_norm": vector_norm(x),
                "cache_len": result["attention"]["cache_len"],
                "mlp": result["mlp"],
            })
        return {
            "token_id": token_id,
            "position": position,
            "hidden_norm": vector_norm(x),
            "top_logits": runtime.logits_topk(x, top_k=top_k, vocab_limit=vocab_limit),
            "layers": layer_summaries,
        }

    with GpuCcq4Session(runtime.dll, runtime.gpu) as session:
        preload_summary = preload_layer_resident_weights(runtime, session, layer_count, run_mlp) if preload_resident_layers and resident else None
        position = 0
        for token_id in prompt_ids:
            step = run_token(session, token_id, position)
            prefill_steps.append(step)
            last_logits = step["top_logits"]
            position += 1

        for step_index in range(max_new_tokens):
            if not last_logits:
                break
            selected = sample_next_token(
                last_logits,
                all_ids,
                rng,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                banned_token_ids=banned_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                no_repeat_window=no_repeat_window,
            )
            next_id = int(selected["token_id"])
            all_ids.append(next_id)
            steps.append({
                "step": step_index,
                "token_id": next_id,
                "piece": tokenizer.decode([next_id], skip_special=False),
                "logit": float(selected["logit"]),
                "probability": float(selected.get("probability", 0.0)),
                "sampling": selected.get("sampling", "unknown"),
                "top_logits": last_logits,
            })
            if stop_eos and next_id == tokenizer.eos_id:
                break
            step = run_token(session, next_id, position)
            position += 1
            last_logits = step["top_logits"]
        resident_matrix_count = session.resident_count

    prefill = {
        "tokens": prompt_ids,
        "layers_executed": layer_count,
        "steps": prefill_steps,
    }

    return {
        "prompt_ids": prompt_ids,
        "generated_ids": all_ids[len(prompt_ids):],
        "all_ids": all_ids,
        "decoded": tokenizer.decode(all_ids),
        "generated_text": tokenizer.decode(all_ids[len(prompt_ids):]),
        "max_layers": max_layers,
        "run_mlp": run_mlp,
        "resident_weights": resident,
        "resident_matrix_count": resident_matrix_count,
        "preload": preload_summary,
        "vocab_limit": vocab_limit,
        "sampling": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "no_repeat_window": no_repeat_window,
            "seed": seed,
            "stop_eos": stop_eos,
        },
        "elapsed_sec": round(time.time() - start, 3),
        "prefill": prefill,
        "steps": steps,
}


def layer_weight_names(layer_index: int, include_mlp: bool = True) -> list[str]:
    prefix = f"model.language_model.layers.{layer_index}"
    names = [
        f"{prefix}.self_attn.q_proj.weight",
        f"{prefix}.self_attn.k_proj.weight",
        f"{prefix}.self_attn.v_proj.weight",
        f"{prefix}.self_attn.o_proj.weight",
    ]
    if include_mlp:
        names.extend([
            f"{prefix}.mlp.gate_proj.weight",
            f"{prefix}.mlp.up_proj.weight",
            f"{prefix}.mlp.down_proj.weight",
        ])
    return names


def preload_layer_resident_weights(
    runtime: GemmaForwardRuntime,
    session: GpuCcq4Session,
    layer_count: int,
    include_mlp: bool,
) -> dict:
    start = time.time()
    loaded = []
    for layer_index in range(layer_count):
        for name in layer_weight_names(layer_index, include_mlp=include_mlp):
            path = runtime.tensor_path(name)
            matrix = session.resident_matrix(path)
            loaded.append({
                "layer": layer_index,
                "name": name,
                "rows": matrix.rows,
                "cols": matrix.cols,
            })
    return {
        "enabled": True,
        "matrix_count": len(loaded),
        "resident_matrix_count": session.resident_count,
        "elapsed_sec": round(time.time() - start, 3),
        "layers": layer_count,
        "include_mlp": include_mlp,
    }


class EnterpriseChatSession:
    """Autoregressive chat loop with one runtime, one GPU session, and persistent KV cache."""

    def __init__(
        self,
        runtime: GemmaForwardRuntime,
        tokenizer: GemmaBpeTokenizer,
        max_layers: int,
        run_mlp: bool,
        top_k: int,
        vocab_limit: int | None,
        resident: bool,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        seed: int,
        stop_eos: bool,
        no_repeat_ngram_size: int,
        no_repeat_window: int,
        preload_resident_layers: bool,
        system_prompt: str | None = None,
    ):
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.layer_count = min(max_layers, runtime.config.num_layers)
        self.run_mlp = run_mlp
        self.top_k = top_k
        self.vocab_limit = vocab_limit
        self.resident = resident
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.stop_eos = stop_eos
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.no_repeat_window = no_repeat_window
        self.preload_resident_layers = preload_resident_layers
        self.preload_summary = None
        self.system_prompt = system_prompt
        self.rng = random.Random(seed)
        self.banned_ids = banned_generation_token_ids(tokenizer)
        self.session: GpuCcq4Session | None = None
        self.position = 0
        self.all_ids: list[int] = []
        self.last_logits: list[dict] = []
        self.turns: list[dict] = []

    def __enter__(self) -> "EnterpriseChatSession":
        self.session = GpuCcq4Session(self.runtime.dll, self.runtime.gpu).__enter__()
        if self.preload_resident_layers and self.resident:
            self.preload_summary = preload_layer_resident_weights(
                self.runtime,
                self.session,
                self.layer_count,
                self.run_mlp,
            )
        if self.system_prompt:
            self.prefill_text(
                "<system>\n"
                f"{self.system_prompt.strip()}\n"
                "</system>\n"
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.session is not None:
            self.session.__exit__(exc_type, exc, tb)
        self.session = None

    def _run_token(self, token_id: int) -> dict:
        if self.session is None:
            raise RuntimeError("EnterpriseChatSession is not initialized")
        x = self.runtime.embedding(token_id)
        layer_summaries = []
        for layer_index in range(self.layer_count):
            result = self.runtime.layer(
                self.session,
                x,
                layer_index,
                self.position,
                self.run_mlp,
                resident=self.resident,
            )
            x = result["hidden"]
            layer_summaries.append({
                "layer": layer_index,
                "hidden_norm": vector_norm(x),
                "cache_len": result["attention"]["cache_len"],
                "mlp": result["mlp"],
            })
        self.position += 1
        self.all_ids.append(token_id)
        self.last_logits = self.runtime.logits_topk(x, top_k=self.top_k, vocab_limit=self.vocab_limit)
        return {
            "token_id": token_id,
            "position": self.position - 1,
            "hidden_norm": vector_norm(x),
            "top_logits": self.last_logits,
            "layers": layer_summaries,
        }

    def prefill_ids(self, token_ids: list[int]) -> list[dict]:
        return [self._run_token(int(token_id)) for token_id in token_ids]

    def prefill_text(self, text: str, add_bos: bool = False) -> list[dict]:
        return self.prefill_ids(self.tokenizer.encode(text, add_bos=add_bos))

    def reply(self, user_text: str, max_new_tokens: int) -> dict:
        turn_start = time.time()
        prompt_text = (
            "<user>\n"
            f"{user_text.strip()}\n"
            "</user>\n"
            "<assistant>\n"
        )
        prefill_steps = self.prefill_text(prompt_text, add_bos=(self.position == 0))
        generated_ids: list[int] = []
        generated_steps = []
        for step_index in range(max_new_tokens):
            if not self.last_logits:
                break
            selected = sample_next_token(
                self.last_logits,
                self.all_ids,
                self.rng,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                banned_token_ids=self.banned_ids,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                no_repeat_window=self.no_repeat_window,
            )
            next_id = int(selected["token_id"])
            generated_ids.append(next_id)
            generated_steps.append({
                "step": step_index,
                "token_id": next_id,
                "piece": self.tokenizer.decode([next_id], skip_special=False),
                "logit": float(selected["logit"]),
                "probability": float(selected.get("probability", 0.0)),
                "sampling": selected.get("sampling", "unknown"),
                "top_logits": self.last_logits,
            })
            self._run_token(next_id)
            if self.stop_eos and next_id == self.tokenizer.eos_id:
                break
        assistant_text = self.tokenizer.decode(generated_ids)
        close_text = "\n</assistant>\n"
        close_steps = self.prefill_text(close_text, add_bos=False)
        turn = {
            "user": user_text,
            "generated_ids": generated_ids,
            "assistant_text": assistant_text,
            "prefill_steps": prefill_steps,
            "generated_steps": generated_steps,
            "close_steps": close_steps,
            "position": self.position,
            "resident_matrix_count": self.session.resident_count if self.session is not None else 0,
            "preload": self.preload_summary,
            "elapsed_sec": round(time.time() - turn_start, 3),
        }
        self.turns.append(turn)
        return turn


def run_interactive_chat(
    runtime: GemmaForwardRuntime,
    tokenizer: GemmaBpeTokenizer,
    system_prompt: str | None,
    max_new_tokens: int,
    max_layers: int,
    run_mlp: bool,
    top_k: int,
    vocab_limit: int | None,
    resident: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    seed: int,
    stop_eos: bool,
    no_repeat_ngram_size: int,
    no_repeat_window: int,
    preload_resident_layers: bool,
    quiet_driver: bool,
    driver_log: str,
) -> int:
    print("CC Enterprise interactive session. Empty input or /exit stops.", flush=True)
    if quiet_driver:
        print(f"Driver output redirected to {driver_log}", flush=True)
    with DriverOutputSilencer(quiet_driver, driver_log):
        chat_context = EnterpriseChatSession(
            runtime=runtime,
            tokenizer=tokenizer,
            max_layers=max_layers,
            run_mlp=run_mlp,
            top_k=top_k,
            vocab_limit=vocab_limit,
            resident=resident,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            stop_eos=stop_eos,
            no_repeat_ngram_size=no_repeat_ngram_size,
            no_repeat_window=no_repeat_window,
            preload_resident_layers=preload_resident_layers,
            system_prompt=system_prompt,
        )
        chat = chat_context.__enter__()
    try:
        print("CC Enterprise ready.", flush=True)
        while True:
            try:
                user_text = input("user> ").strip()
            except EOFError:
                break
            if not user_text or user_text.lower() in {"/exit", "/quit"}:
                break
            with DriverOutputSilencer(quiet_driver, driver_log):
                turn = chat.reply(user_text, max_new_tokens=max_new_tokens)
            print(f"assistant> {turn['assistant_text']}")
            print(
                json.dumps({
                    "generated_ids": turn["generated_ids"],
                    "position": turn["position"],
                    "resident_matrix_count": turn["resident_matrix_count"],
                    "preload": turn["preload"],
                    "elapsed_sec": turn["elapsed_sec"],
                }, ensure_ascii=False)
            )
    finally:
        with DriverOutputSilencer(quiet_driver, driver_log):
            chat_context.__exit__(None, None, None)
    return 0


def run_co_gpu_residency_probe(ccq4_dir: str, dll: str, gpu: int, resident: bool) -> dict:
    probe_runtime = GemmaForwardRuntime(ccq4_dir, dll, gpu)
    probe_name = "model.language_model.layers.0.self_attn.q_proj.weight"
    x = [0.0] * probe_runtime.config.hidden_size
    if x:
        x[0] = 1.0
        x[17 % len(x)] = -0.5
        x[113 % len(x)] = 0.25
    start = time.time()
    with GpuCcq4Session(probe_runtime.dll, probe_runtime.gpu) as session:
        path = probe_runtime.tensor_path(probe_name)
        y = session.matvec_resident(path, x) if resident else session.matvec(path, x)
        resident_count = session.resident_count
    return {
        "gpu": gpu,
        "probe": probe_name,
        "output_count": len(y),
        "output_norm": vector_norm(y),
        "resident_matrix_count": resident_count,
        "elapsed_sec": round(time.time() - start, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prompt-testable CCQ4 Gemma enterprise runtime.")
    parser.add_argument("--prompt", default=None, help="Text prompt, or a JSON list of token ids such as [2,186].")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--tokenizer-json", default="model/tokenizer/tokenizer.json")
    parser.add_argument("--ccq4-dir", default="model/ccq4")
    parser.add_argument("--dll", default="driver/build/CC_OpenCl.dll")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument(
        "--also-gpu",
        type=int,
        action="append",
        default=[],
        help="Initialize and probe an additional GPU before the main inference session. Can be repeated.",
    )
    parser.add_argument("--max-layers", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=2)
    parser.add_argument("--no-repeat-window", type=int, default=8)
    parser.add_argument("--emotion-mode", choices=["precise", "calm", "creative", "dream"], default="calm")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--no-stop-eos", action="store_true")
    parser.add_argument("--vocab-limit", type=int, default=2048)
    parser.add_argument("--skip-mlp", action="store_true")
    parser.add_argument("--no-bos", action="store_true")
    parser.add_argument("--no-resident-weights", action="store_true")
    parser.add_argument(
        "--preload-resident-layers",
        action="store_true",
        help="Register all active layer CCQ4 matrices in the driver before the first token.",
    )
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--quiet-driver", action="store_true", help="Redirect verbose DLL/kernel output to --driver-log.")
    parser.add_argument("--driver-log", default="logs/enterprise_driver.log")
    args = parser.parse_args()

    try:
        tokenizer = GemmaBpeTokenizer(args.tokenizer_json)
        runtime = GemmaForwardRuntime(args.ccq4_dir, args.dll, args.gpu)
        profile = emotion_sampling_profile(args.emotion_mode)
        temperature = profile["temperature"] if args.temperature is None else args.temperature
        top_p = profile["top_p"] if args.top_p is None else args.top_p
        repetition_penalty = profile["repetition_penalty"] if args.repetition_penalty is None else args.repetition_penalty
        co_gpu_reports = []
        for co_gpu in args.also_gpu:
            if co_gpu == args.gpu:
                co_gpu_reports.append({"gpu": co_gpu, "skipped": "same as primary --gpu"})
                continue
            with DriverOutputSilencer(args.quiet_driver, args.driver_log):
                report = run_co_gpu_residency_probe(
                    ccq4_dir=args.ccq4_dir,
                    dll=args.dll,
                    gpu=co_gpu,
                    resident=not args.no_resident_weights,
                )
            co_gpu_reports.append(report)
            print(json.dumps({"co_gpu_probe": report}, indent=2, ensure_ascii=False), flush=True)
        if args.interactive:
            return run_interactive_chat(
                runtime=runtime,
                tokenizer=tokenizer,
                system_prompt=args.system_prompt,
                max_new_tokens=args.max_new_tokens,
                max_layers=args.max_layers,
                run_mlp=not args.skip_mlp,
                top_k=args.top_k,
                vocab_limit=args.vocab_limit,
                resident=not args.no_resident_weights,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=args.seed,
                stop_eos=not args.no_stop_eos,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                no_repeat_window=args.no_repeat_window,
                preload_resident_layers=args.preload_resident_layers,
                quiet_driver=args.quiet_driver,
                driver_log=args.driver_log,
            )
        if args.prompt is None:
            raise ValueError("--prompt is required unless --interactive is set")
        prompt_text = compose_prompt(args.system_prompt, args.prompt)
        prompt_ids = parse_prompt_ids(prompt_text, tokenizer, add_bos=not args.no_bos)
        with DriverOutputSilencer(args.quiet_driver, args.driver_log):
            result = generate(
                runtime=runtime,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                max_new_tokens=args.max_new_tokens,
                max_layers=args.max_layers,
                run_mlp=not args.skip_mlp,
                top_k=args.top_k,
                vocab_limit=args.vocab_limit,
                resident=not args.no_resident_weights,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=args.seed,
                stop_eos=not args.no_stop_eos,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                no_repeat_window=args.no_repeat_window,
                preload_resident_layers=args.preload_resident_layers,
            )
        result["system_prompt"] = args.system_prompt
        result["emotion_mode"] = args.emotion_mode
        result["composed_prompt"] = prompt_text
        result["co_gpu_reports"] = co_gpu_reports
        if args.json_out:
            out = Path(args.json_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "prompt_ids": result["prompt_ids"],
            "generated_ids": result["generated_ids"],
            "decoded": result["decoded"],
            "generated_text": result["generated_text"],
            "emotion_mode": result["emotion_mode"],
            "co_gpu_reports": result["co_gpu_reports"],
            "sampling": result["sampling"],
            "preload": result["preload"],
            "elapsed_sec": result["elapsed_sec"],
            "json_out": args.json_out,
        }, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"gemma_runtime.enterprise_model: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
