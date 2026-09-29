"""Minimal BM25, same as in p01_classic_rag, kept local so each part runs on its own."""

import math
import re
from collections import Counter

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
