"""Part 03: how chunk size and chunking strategy change what the generator gets to read.

Data: the QASPER test set (Dasigi et al., 2021; CC BY 4.0). 416 NLP papers with full text, and
questions written by people who had only read the title and abstract. Each answer comes with
highlighted evidence sentences in the full text. We keep answerable questions whose evidence is
plain text found verbatim in the paper (963 questions), and retrieve within the paper the question
is about.

A question counts as answered by the retrieved context if at least 80% of the characters of one
of its evidence spans are covered by the retrieved text ("any span"). The stricter "all spans"
requires every evidence span to be covered.

Two ways to compare chunk sizes:
  * fixed k: always return the top 5 chunks, whatever their size (bigger chunks hand over more text)
  * fixed budget: fill the context with ranked chunks up to B words, the way a real prompt is filled

Usage:
    python chunk_eval.py

Needs ../data/qasper/qasper-test-v0.3.json:
    curl -L -o ../data/qasper/qasper-test.tgz https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz
    tar -xzf ../data/qasper/qasper-test.tgz -C ../data/qasper
"""

import json
import re
from pathlib import Path

from bm25 import BM25

DATA = Path(__file__).resolve().parent.parent / "data" / "qasper" / "qasper-test-v0.3.json"
COVER = 0.8


# ---------------------------------------------------------------- documents and questions

class Doc:
    """Full text as one string, with paragraph spans and the section each paragraph belongs to."""

    def __init__(self, paper):
        self.title = paper["title"]
        parts, self.paras, pos = [], [], 0
        for sec in paper["full_text"]:
            for p in sec["paragraphs"]:
                p = p.strip()
                if not p:
                    continue
                self.paras.append((pos, pos + len(p), sec["section_name"] or ""))
                parts.append(p)
                pos += len(p) + 2
        self.text = "\n\n".join(parts)
        self.words = [(m.start(), m.end()) for m in re.finditer(r"\S+", self.text)]

    def section_at(self, char):
        for s, e, name in self.paras:
            if s <= char < e + 2:
                return name
        return ""


def load():
    papers = json.load(open(DATA, encoding="utf-8"))
    docs, questions = {}, []
    for pid, paper in papers.items():
        doc = Doc(paper)
        docs[pid] = doc
        for qa in paper["qas"]:
            ans = qa["answers"][0]["answer"]
            spans = [h.strip() for h in ans["highlighted_evidence"] if h.strip()]
            if ans["unanswerable"] or not spans or any("FLOAT SELECTED" in h for h in spans):
                continue
            locs = [(doc.text.find(h), doc.text.find(h) + len(h)) for h in spans]
            if any(s < 0 for s, _ in locs):
                continue
            questions.append({"pid": pid, "q": qa["question"], "spans": locs})
    return docs, questions


# ---------------------------------------------------------------- chunkers: lists of (start, end) char spans

def fixed(doc, size, stride=None):
    stride = stride or size
    w = doc.words
    return [(w[i][0], w[min(i + size, len(w)) - 1][1]) for i in range(0, len(w), stride)]


def sentences(doc):
    """Sentence spans inside paragraphs (a simple splitter is enough for this purpose)."""
    out = []
    for s, e, _ in doc.paras:
        para = doc.text[s:e]
        start = 0
        for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z(\[])", para):
            out.append((s + start, s + m.start()))
            start = m.end()
        out.append((s + start, e))
    return out


def sentence_pack(doc, size):
    """Pack whole sentences up to `size` words; never cross a paragraph boundary."""
    chunks, cur, cur_words, cur_para = [], None, 0, None
    for s, e in sentences(doc):
        n = len(doc.text[s:e].split())
        para = next(i for i, (ps, pe, _) in enumerate(doc.paras) if ps <= s < pe + 2)
        if cur and (cur_words + n > size or para != cur_para):
            chunks.append(cur)
            cur, cur_words = None, 0
        cur = (cur[0], e) if cur else (s, e)
        cur_words += n
        cur_para = para
    if cur:
        chunks.append(cur)
    return chunks


def paragraphs(doc):
    return [(s, e) for s, e, _ in doc.paras]


def sections(doc):
    out = []
    for s, e, name in doc.paras:
        if out and out[-1][2] == name:
            out[-1] = (out[-1][0], e, name)
        else:
            out.append((s, e, name))
    return [(s, e) for s, e, _ in out]


# ---------------------------------------------------------------- retrieval and scoring

def run(docs, questions, chunker, budget=None, k=None, prefix=False, parent=None, cover=COVER):
    """Return (any-span hit rate, all-span hit rate, average words handed over)."""
    cache = {}
    any_hit = all_hit = words_used = 0
    for q in questions:
        doc = docs[q["pid"]]
        if q["pid"] not in cache:
            spans = chunker(doc)
            texts = {i: (f"{doc.title}. {doc.section_at(s)}. " if prefix else "") + doc.text[s:e]
                     for i, (s, e) in enumerate(spans)}
            cache[q["pid"]] = (spans, BM25(texts))
        spans, index = cache[q["pid"]]
        ranked = [spans[i] for i, _ in index.search(q["q"], len(spans))]
        if parent:
            parents = parent(doc)
            ranked = list(dict.fromkeys(next(p for p in parents if p[0] <= s < p[1] + 2) for s, _ in ranked))
        if k:
            ranked = ranked[:k]
        covered, used = [], 0
        for s, e in ranked:
            n = len(doc.text[s:e].split())
            if budget and used + n > budget:
                cut = doc.words[[i for i, (ws, _) in enumerate(doc.words) if ws >= s][0] + budget - used - 1][1] \
                    if budget > used else s
                if cut > s:
                    covered.append((s, cut))
                    used = budget
                break
            covered.append((s, e))
            used += n
        cov = [coverage(span, covered) >= cover for span in q["spans"]]
        any_hit += any(cov)
        all_hit += all(cov)
        words_used += used
    n = len(questions)
    return any_hit / n, all_hit / n, words_used / n


def coverage(span, covered):
    s, e = span
    hit = [False] * (e - s)
    for cs, ce in covered:
        for i in range(max(s, cs), min(e, ce)):
            hit[i - s] = True
    return sum(hit) / max(1, len(hit))


def run_pooled(docs, questions, chunker, budget, prefix=False):
    """All papers' chunks in one index; a hit must come from the right paper."""
    from bm25 import InvertedBM25
    keys, texts = [], []
    for pid, doc in docs.items():
        for s, e in chunker(doc):
            keys.append((pid, s, e))
            texts.append((f"{doc.title}. {doc.section_at(s)}. " if prefix else "") + doc.text[s:e])
    index = InvertedBM25(texts)
    hit = 0
    for q in questions:
        covered, used = [], 0
        for i, _ in index.search(q["q"], 200):
            pid, s, e = keys[i]
            n = len(docs[pid].text[s:e].split())
            if used + n > budget:
                break
            used += n
            if pid == q["pid"]:
                covered.append((s, e))
        hit += any(coverage(span, covered) >= COVER for span in q["spans"])
    return hit / len(questions)


# ---------------------------------------------------------------- experiments

def main():
    docs, questions = load()
    print(f"papers: {len(docs)}, questions with text evidence: {len(questions)}")
    sizes = [32, 64, 128, 256, 512, 1024]

    print("\n1) Fixed k = 5 chunks (bigger chunks simply hand over more text)")
    print("size   any-span  all-span  words handed over")
    for size in sizes:
        a, b, w = run(docs, questions, lambda d, s=size: fixed(d, s), k=5)
        print(f"{size:<6} {a:7.1%}  {b:7.1%}  {w:8.0f}")

    for budget in (256, 512, 1024):
        print(f"\n2) Fixed budget = {budget} words of context")
        print("chunking                                  any-span  all-span")
        rows = [(f"fixed {s} words", lambda d, s=s: fixed(d, s), {}) for s in sizes]
        rows += [
            ("fixed 128, 25% overlap", lambda d: fixed(d, 128, 96), {}),
            ("sentences packed to 128, within paragraph", lambda d: sentence_pack(d, 128), {}),
            ("one paragraph per chunk", paragraphs, {}),
            ("one section per chunk", sections, {}),
            ("fixed 64 + title/section prefix", lambda d: fixed(d, 64), {"prefix": True}),
            ("search 64-word sentence packs, return paragraph", lambda d: sentence_pack(d, 64),
             {"parent": paragraphs}),
        ]
        for name, chunker, kw in rows:
            a, b, _ = run(docs, questions, chunker, budget=budget, **kw)
            print(f"{name:<44} {a:7.1%}  {b:7.1%}")

    print("\n3) Sensitivity: same budget (512 words), looser coverage threshold (50%)")
    for name, chunker in [("fixed 128 words", lambda d: fixed(d, 128)),
                          ("sentences packed to 128, within paragraph", lambda d: sentence_pack(d, 128)),
                          ("one paragraph per chunk", paragraphs)]:
        a, b, _ = run(docs, questions, chunker, budget=512, cover=0.5)
        print(f"  {name:<44} any-span {a:6.1%}")

    print("\n4) All 416 papers in one index (budget 512 words; the hit must come from the right paper)")
    for name, chunker, prefix in [("fixed 64", lambda d: fixed(d, 64), False),
                                  ("fixed 64 + title/section prefix", lambda d: fixed(d, 64), True),
                                  ("one paragraph per chunk", paragraphs, False),
                                  ("one paragraph per chunk + title/section prefix", paragraphs, True)]:
        print(f"  {name:<48} any-span {run_pooled(docs, questions, chunker, 512, prefix):6.1%}")

    this_effect_demo()


def this_effect_demo():
    """'This effect persisted for 8 weeks': a chunk that carries the answer but not its subject.

    Uses the Part 02 paper (Yoshimoto et al. 2023, CC BY 4.0), one sentence per chunk.
    """
    from lxml import etree
    path = Path(__file__).resolve().parent.parent / "data" / "p02" / "paper.xml"
    if not path.exists():
        print("\n5) skipped: run ../p02_document_parsing/parse_compare.py first to download the paper")
        return
    xml = etree.parse(str(path))
    body = " ".join(" ".join(p.itertext()) for s in xml.find(".//body").findall("./sec")
                    if s.findtext("title") in ("Background", "Methods", "Results", "Discussion", "Conclusions")
                    for p in s.iter("p"))
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", re.sub(r"\s+", " ", body))
    query = "How long did the reduction in alcohol consumption last after the free non-alcoholic beverages stopped?"
    targets = [i for i, s in enumerate(sents) if "this effect persisted" in s.lower()]
    context = "Context: effect of providing free non-alcoholic beverages for 12 weeks on alcohol consumption. "
    print(f"\n5) One sentence per chunk ({len(sents)} chunks). Query: {query}")
    for i in targets:
        print(f"   answer sentence: {sents[i]}")
        print(f"   sentence before: {sents[i - 1]}")
    for label, docs in [("as is", dict(enumerate(sents))),
                        ("with a one-line context prefix", {i: (context + s if i in targets else s)
                                                            for i, s in enumerate(sents)})]:
        ranked = [i for i, _ in BM25(docs).search(query, len(docs))]
        print(f"   {label:<32} answer sentences rank {[ranked.index(t) + 1 for t in targets]}; "
              f"top result: {sents[ranked[0]][:80]!r}")


if __name__ == "__main__":
    main()
