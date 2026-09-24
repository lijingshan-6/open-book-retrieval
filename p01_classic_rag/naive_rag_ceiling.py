"""Part 01: the retrieval ceiling of a naive RAG pipeline.

A naive RAG pipeline retrieves once, keeps a fixed top-k, and hands those passages to the
generator. If the evidence is not in the top-k, the generator never sees it. This script
measures how often that happens on SciFact with a plain BM25 retriever, and then runs a
hand-built two-hop example where a single retrieval step cannot find the second fact.

Usage:
    python naive_rag_ceiling.py

Needs ../data/scifact (see ../p00_why_retrieval/README.md for the download). No third-party
packages are required.
"""

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "scifact"

STOPWORDS = set("""
a an and are as at be by for from has have in is it its of on or that the this to was were
which with what who whom where when why how in into than then there these those
""".split())


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, docs, k1=0.9, b=0.4):
        self.k1, self.b = k1, b
        self.ids = list(docs)
        self.tfs = [Counter(tokenize(docs[i])) for i in self.ids]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = sum(self.lens) / len(self.lens)
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(self.ids)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query, k):
        scores = []
        for doc_id, tf, dl in zip(self.ids, self.tfs, self.lens):
            s = 0.0
            for t in tokenize(query):
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores.append((s, doc_id))
        scores.sort(reverse=True)
        return [(doc_id, s) for s, doc_id in scores[:k]]


def scifact_ceiling():
    with open(DATA_DIR / "corpus.jsonl", encoding="utf-8") as f:
        docs = {}
        for line in f:
            d = json.loads(line)
            docs[d["_id"]] = f"{d.get('title', '')} {d['text']}"
    with open(DATA_DIR / "queries.jsonl", encoding="utf-8") as f:
        queries = {q["_id"]: q["text"] for q in map(json.loads, f)}
    qrels = defaultdict(set)
    with open(DATA_DIR / "qrels" / "test.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if int(row["score"]) > 0:
                qrels[row["query-id"]].add(row["corpus-id"])

    bm25 = BM25(docs)
    ks = [1, 3, 5, 10, 20, 100]
    hits = Counter()
    for qid, rel in qrels.items():
        ranked = [d for d, _ in bm25.search(queries[qid], max(ks))]
        for k in ks:
            if rel & set(ranked[:k]):
                hits[k] += 1

    n = len(qrels)
    print(f"SciFact test queries: {n}")
    print("k     Hit@k   queries whose evidence never reaches the generator")
    for k in ks:
        print(f"{k:<5} {hits[k] / n:6.1%}   {n - hits[k]:>3} ({(n - hits[k]) / n:.1%})")


TOY_DOCS = {
    "D1": "Penicillin was discovered in 1928 by Alexander Fleming at St Mary's Hospital in London.",
    "D2": "Alexander Fleming was born in 1881 on a farm near Darvel in Scotland.",
    "D3": "Marie Curie was born in Warsaw, in what was then the Kingdom of Poland.",
    "D4": "Louis Pasteur was born in Dole, France, and later developed the first rabies vaccine.",
    "D5": "The scientist Robert Koch was born in Clausthal, Germany; he identified the bacterium that causes tuberculosis.",
    "D6": "Streptomycin, discovered in 1943 by Albert Schatz, was the first antibiotic effective against tuberculosis.",
}


def toy_multi_hop():
    bm25 = BM25(TOY_DOCS)
    question = "In which country was the scientist who discovered penicillin born?"
    print()
    print(f"Hop 1 query: {question}")
    for doc_id, s in bm25.search(question, 3):
        print(f"  {doc_id}  {s:.3f}  {TOY_DOCS[doc_id]}")
    follow_up = "Alexander Fleming born"
    print(f"Hop 2 query (needs the name found in hop 1): {follow_up}")
    for doc_id, s in bm25.search(follow_up, 3):
        print(f"  {doc_id}  {s:.3f}  {TOY_DOCS[doc_id]}")


if __name__ == "__main__":
    scifact_ceiling()
    toy_multi_hop()
