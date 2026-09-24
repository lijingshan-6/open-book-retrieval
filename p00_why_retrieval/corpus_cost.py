"""Part 00: how big is SciFact, and what does one query cost with and without retrieval?

Usage:
    python corpus_cost.py
    python corpus_cost.py --price 2 --top-k 10 --window 1000000

Notes:
- Token counts are estimated with the rough rule from Anthropic's docs: 1 token ~= 0.75 English words.
- If tiktoken is installed, the corpus is also tokenized with cl100k_base as a reference point.
  Different models use different tokenizers, so treat that number as indicative only.
"""

import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_corpus():
    corpus_path = DATA_DIR / "scifact" / "corpus.jsonl"
    zip_path = DATA_DIR / "scifact.zip"
    if not corpus_path.exists():
        if zip_path.exists():
            zipfile.ZipFile(zip_path).extractall(DATA_DIR)
        else:
            print(f"Downloading {BEIR_URL} ...")
            try:
                with urllib.request.urlopen(BEIR_URL) as resp:
                    zipfile.ZipFile(io.BytesIO(resp.read())).extractall(DATA_DIR)
            except urllib.error.URLError as e:
                raise SystemExit(
                    f"Download failed ({e.reason}). Download the file manually, e.g.\n"
                    f"  curl -L -o {zip_path} {BEIR_URL}\n"
                    f"and run this script again."
                )
    with open(corpus_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--price", type=float, default=2.0, help="USD per million input tokens")
    parser.add_argument("--cache-multiplier", type=float, default=0.1, help="cache-read price as a fraction of base input price")
    parser.add_argument("--top-k", type=int, default=10, help="documents placed in context by the retrieval approach")
    parser.add_argument("--window", type=int, default=1_000_000, help="context window size in tokens")
    parser.add_argument("--words-per-token", type=float, default=0.75)
    args = parser.parse_args()

    docs = load_corpus()
    texts = [f"{d.get('title', '')} {d['text']}" for d in docs]
    words = [len(t.split()) for t in texts]
    total_words = sum(words)
    avg_words = total_words / len(docs)
    total_tokens = total_words / args.words_per_token

    print(f"Documents:                    {len(docs):,}")
    print(f"Avg words (title + abstract): {avg_words:.1f}")
    print(f"Total words:                  {total_words:,}")
    print(f"Estimated tokens (/{args.words_per_token}):     {total_tokens:,.0f} "
          f"({total_tokens / args.window:.2f}x a {args.window:,}-token window)")

    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        measured = sum(len(enc.encode(t)) for t in texts)
        print(f"cl100k_base tokens (reference): {measured:,}")
    except ImportError:
        print("tiktoken not installed, skipping measured token count.")

    topk_tokens = args.top_k * avg_words / args.words_per_token
    full = total_tokens * args.price / 1e6
    cached = full * args.cache_multiplier
    topk = topk_tokens * args.price / 1e6

    print()
    print(f"Input cost per query at ${args.price}/MTok:")
    print(f"  Whole corpus in context         {total_tokens:>12,.0f} tokens   ${full:.4f}")
    print(f"  Whole corpus, prompt-cache hit  {total_tokens:>12,.0f} tokens   ${cached:.4f}")
    print(f"  Retrieve top-{args.top_k:<2}, then read      {topk_tokens:>12,.0f} tokens   ${topk:.4f}")
    print(f"  Retrieval is {full / topk:.0f}x cheaper (no cache), {cached / topk:.0f}x cheaper (cache hit)")


if __name__ == "__main__":
    main()
