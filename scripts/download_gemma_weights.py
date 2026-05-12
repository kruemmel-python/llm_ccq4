from __future__ import annotations

import argparse
import os
import pathlib
import ssl
import sys
import time
import urllib.error
import urllib.request


DEFAULT_MODEL = "google/gemma-3n-E4B"
DEFAULT_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
]


def resolve_url(model: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/{model}/resolve/{revision}/{filename}"


def build_ssl_context(insecure: bool) -> ssl.SSLContext:
    if insecure:
        return ssl._create_unverified_context()
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def remote_size(model: str, revision: str, filename: str, token: str | None, ssl_context: ssl.SSLContext) -> int | None:
    request = urllib.request.Request(resolve_url(model, revision, filename), method="HEAD")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=60, context=ssl_context) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length and length.isdigit() else None
    except Exception:
        return None


def download_file(
    model: str,
    revision: str,
    filename: str,
    output_dir: pathlib.Path,
    token: str | None,
    retries: int,
    ssl_context: ssl.SSLContext,
) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    tmp_path = output_dir / f"{filename}.part"
    expected_size = remote_size(model, revision, filename, token, ssl_context)

    if out_path.exists() and expected_size is not None and out_path.stat().st_size == expected_size:
        print(f"OK already complete: {out_path} ({expected_size} bytes)")
        return out_path
    if out_path.exists() and expected_size is None:
        print(f"OK exists, remote size unknown: {out_path} ({out_path.stat().st_size} bytes)")
        return out_path

    for attempt in range(1, retries + 1):
        existing = tmp_path.stat().st_size if tmp_path.exists() else 0
        request = urllib.request.Request(resolve_url(model, revision, filename))
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        if existing > 0:
            request.add_header("Range", f"bytes={existing}-")

        mode = "ab" if existing > 0 else "wb"
        try:
            with urllib.request.urlopen(request, timeout=120, context=ssl_context) as response:
                status = getattr(response, "status", 200)
                if existing > 0 and status == 200:
                    # Server ignored Range; restart cleanly.
                    existing = 0
                    mode = "wb"
                with tmp_path.open(mode) as out:
                    downloaded = existing
                    last_report = time.monotonic()
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        downloaded += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 2.0:
                            if expected_size:
                                pct = downloaded * 100.0 / expected_size
                                print(f"{filename}: {downloaded}/{expected_size} bytes ({pct:.1f}%)")
                            else:
                                print(f"{filename}: {downloaded} bytes")
                            last_report = now

            final_size = tmp_path.stat().st_size
            if expected_size is not None and final_size != expected_size:
                raise IOError(f"incomplete download {final_size} != {expected_size}")
            tmp_path.replace(out_path)
            print(f"Downloaded: {out_path} ({final_size} bytes)")
            return out_path
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise RuntimeError(
                    f"HTTP {exc.code} for {filename}. Check HF_TOKEN and accept the model license for {model}."
                ) from exc
            if exc.code == 404:
                raise RuntimeError(f"HTTP 404 for {filename}; file is not present in {model}@{revision}") from exc
            print(f"{filename}: HTTP {exc.code} attempt {attempt}/{retries}")
        except Exception as exc:
            print(f"{filename}: {exc} attempt {attempt}/{retries}")

        time.sleep(min(10, attempt * 2))

    raise RuntimeError(f"failed to download {filename} after {retries} attempts")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Gemma metadata and safetensors shards without huggingface_hub.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", default="D:/Models/gemma-3n-E4B")
    parser.add_argument("--token", default=None, help="Prefer HF_TOKEN env var instead.")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification. Use only if local CA store is broken.")
    parser.add_argument("--file", action="append", default=None, help="Download only this file; can be repeated.")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("HF token missing. Set $env:HF_TOKEN to a read token with Gemma access.", file=sys.stderr)
        return 1

    files = args.file if args.file else DEFAULT_FILES
    if args.metadata_only:
        files = [name for name in files if not name.endswith(".safetensors")]

    output_dir = pathlib.Path(args.output_dir)
    ssl_context = build_ssl_context(args.insecure)
    try:
        for filename in files:
            download_file(args.model, args.revision, filename, output_dir, token, args.retries, ssl_context)
        print(f"Done: {output_dir}")
        return 0
    except Exception as exc:
        print(f"download_gemma_weights: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
