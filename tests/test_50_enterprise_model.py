from __future__ import annotations

from pathlib import Path

from gemma_runtime.enterprise_model import (
    banned_generation_token_ids,
    blocked_ngram_token_ids,
    compose_prompt,
    layer_weight_names,
    parse_prompt_ids,
    recent_token_ids,
    sample_next_token,
)
from gemma_runtime.tokenizer import GemmaBpeTokenizer
import random


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_JSON = PACKAGE_ROOT / "model" / "tokenizer" / "tokenizer.json"


def test_enterprise_prompt_parser_accepts_json_token_ids() -> None:
    tokenizer = GemmaBpeTokenizer(TOKENIZER_JSON)
    assert parse_prompt_ids("[2, 9259, 18315]", tokenizer, add_bos=True) == [2, 9259, 18315]


def test_enterprise_tokenizer_roundtrip_basic_prompt() -> None:
    tokenizer_path = TOKENIZER_JSON
    if not tokenizer_path.exists():
        return
    tokenizer = GemmaBpeTokenizer(tokenizer_path)
    ids = parse_prompt_ids("Hello enterprise", tokenizer, add_bos=True)
    assert ids[:1] == [2]
    assert tokenizer.decode(ids) == "Hello enterprise"


def test_compose_prompt_adds_system_block() -> None:
    prompt = compose_prompt("Du bist CC.", "Hallo")
    assert "<system>" in prompt
    assert "Du bist CC." in prompt
    assert "<assistant>" in prompt


def test_sample_next_token_respects_zero_temperature_greedy() -> None:
    logits = [{"token_id": 10, "logit": 1.0}, {"token_id": 11, "logit": 2.0}]
    selected = sample_next_token(logits, [], random.Random(1), temperature=0.0, top_p=1.0, repetition_penalty=1.0)
    assert selected["token_id"] == 11
    assert selected["sampling"] == "greedy"


def test_banned_generation_tokens_exclude_bos_not_eos() -> None:
    tokenizer_path = TOKENIZER_JSON
    if not tokenizer_path.exists():
        return
    tokenizer = GemmaBpeTokenizer(tokenizer_path)
    banned = banned_generation_token_ids(tokenizer)
    assert tokenizer.bos_id in banned
    assert tokenizer.unk_id in banned
    assert tokenizer.eos_id not in banned


def test_blocked_ngram_token_ids_prevents_repeating_seen_bigram() -> None:
    assert blocked_ngram_token_ids([10, 11, 10], 2) == {11}
    assert blocked_ngram_token_ids([10, 11, 12], 3) == set()


def test_recent_token_ids_blocks_last_window() -> None:
    assert recent_token_ids([10, 11, 12, 13], 2) == {12, 13}
    assert recent_token_ids([10, 11], 0) == set()


def test_sample_next_token_respects_recent_window_block() -> None:
    logits = [{"token_id": 10, "logit": 9.0}, {"token_id": 11, "logit": 8.0}]
    selected = sample_next_token(
        logits,
        [10],
        random.Random(1),
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        no_repeat_window=4,
    )
    assert selected["token_id"] == 11


def test_layer_weight_names_lists_attention_and_mlp_matrices() -> None:
    names = layer_weight_names(3, include_mlp=True)
    assert len(names) == 7
    assert names[0] == "model.language_model.layers.3.self_attn.q_proj.weight"
    assert names[-1] == "model.language_model.layers.3.mlp.down_proj.weight"
    assert len(layer_weight_names(3, include_mlp=False)) == 4


if __name__ == "__main__":
    test_enterprise_prompt_parser_accepts_json_token_ids()
    test_enterprise_tokenizer_roundtrip_basic_prompt()
    test_compose_prompt_adds_system_block()
    test_sample_next_token_respects_zero_temperature_greedy()
    test_banned_generation_tokens_exclude_bos_not_eos()
    test_blocked_ngram_token_ids_prevents_repeating_seen_bigram()
    test_recent_token_ids_blocks_last_window()
    test_sample_next_token_respects_recent_window_block()
    test_layer_weight_names_lists_attention_and_mlp_matrices()
