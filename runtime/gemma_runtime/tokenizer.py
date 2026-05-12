from __future__ import annotations

import argparse
import json
from pathlib import Path


class GemmaBpeTokenizer:
    def __init__(self, tokenizer_json: str | Path):
        path = Path(tokenizer_json)
        data = json.loads(path.read_text(encoding="utf-8"))
        model = data.get("model", {})
        if model.get("type") != "BPE":
            raise ValueError(f"unsupported tokenizer model type: {model.get('type')}")
        self.vocab: dict[str, int] = {str(k): int(v) for k, v in model.get("vocab", {}).items()}
        self.id_to_token: dict[int, str] = {idx: token for token, idx in self.vocab.items()}
        self.merge_ranks: dict[tuple[str, str], int] = {
            (str(pair[0]), str(pair[1])): rank
            for rank, pair in enumerate(model.get("merges", []))
            if isinstance(pair, list) and len(pair) == 2
        }
        self.bos_id = self.vocab.get("<bos>", 2)
        self.eos_id = self.vocab.get("<eos>", 1)
        self.unk_id = self.vocab.get("<unk>", 3)

    def _piece_for_char(self, ch: str) -> list[str]:
        if ch in self.vocab:
            return [ch]
        out = []
        for b in ch.encode("utf-8"):
            token = f"<0x{b:02X}>"
            out.append(token if token in self.vocab else "<unk>")
        return out

    def _bpe(self, pieces: list[str]) -> list[str]:
        if len(pieces) < 2:
            return pieces
        while True:
            best_rank = None
            best_index = -1
            best_token = None
            for index in range(len(pieces) - 1):
                pair = (pieces[index], pieces[index + 1])
                rank = self.merge_ranks.get(pair)
                if rank is None:
                    continue
                merged = pieces[index] + pieces[index + 1]
                if merged not in self.vocab:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_index = index
                    best_token = merged
            if best_rank is None or best_token is None:
                return pieces
            pieces = pieces[:best_index] + [best_token] + pieces[best_index + 2:]

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        normalized = text.replace(" ", "▁")
        pieces: list[str] = []
        for ch in normalized:
            pieces.extend(self._piece_for_char(ch))
        token_pieces = self._bpe(pieces)
        ids = [self.vocab.get(piece, self.unk_id) for piece in token_pieces]
        return ([self.bos_id] if add_bos else []) + ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        byte_buffer = bytearray()
        out: list[str] = []

        def flush_bytes() -> None:
            if byte_buffer:
                out.append(byte_buffer.decode("utf-8", errors="replace"))
                byte_buffer.clear()

        for token_id in ids:
            token = self.id_to_token.get(int(token_id), "<unk>")
            if skip_special and token.startswith("<") and token.endswith(">") and not token.startswith("<0x"):
                continue
            if token.startswith("<0x") and token.endswith(">") and len(token) == 6:
                try:
                    byte_buffer.append(int(token[3:5], 16))
                    continue
                except ValueError:
                    pass
            flush_bytes()
            out.append(token)
        flush_bytes()
        return "".join(out).replace("▁", " ")


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Gemma tokenizer smoke tool.")
    parser.add_argument("--tokenizer-json", default="D:/Models/gemma-3n-E4B/tokenizer.json")
    parser.add_argument("--text", required=True)
    parser.add_argument("--no-bos", action="store_true")
    args = parser.parse_args()
    tok = GemmaBpeTokenizer(args.tokenizer_json)
    ids = tok.encode(args.text, add_bos=not args.no_bos)
    print(json.dumps({"ids": ids, "decoded": tok.decode(ids)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
