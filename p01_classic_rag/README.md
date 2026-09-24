# Part 01 · Classic RAG as a reference point

The article walks through the original RAG paper (Lewis et al., 2020) and the naive RAG pipeline that engineering practice later kept under the same name. Two of its claims are checked here:

1. **Retrieval recall is the ceiling of a retrieve-once, fixed-top-k pipeline.** If the evidence is not in the top-k, the generator never sees it.
2. **A single retrieval step cannot answer a multi-hop question** when the bridge entity only appears in the first hop's results.

`naive_rag_ceiling.py` uses a small, dependency-free BM25 (k1 = 0.9, b = 0.4) so the numbers do not depend on any library version.

```bash
python naive_rag_ceiling.py
```

It reads SciFact from `../data/scifact`. See [`../p00_why_retrieval/README.md`](../p00_why_retrieval/README.md) for the download.

## Measured output

SciFact test set, 300 queries. Hit@k is the share of queries with at least one relevant abstract in the top k.

```
k     Hit@k   queries whose evidence never reaches the generator
1      54.3%   137 (45.7%)
3      71.0%    87 (29.0%)
5      74.0%    78 (26.0%)
10     80.3%    59 (19.7%)
20     84.0%    48 (16.0%)
100    89.3%    32 (10.7%)
```

With a fixed top-5, a quarter of the queries never get any relevant abstract in front of the generator.

The two-hop example uses a hand-built six-sentence corpus, because SciFact has no multi-hop questions:

```
Hop 1 query: In which country was the scientist who discovered penicillin born?
  D1  2.527  Penicillin was discovered in 1928 by Alexander Fleming at St Mary's Hospital in London.
  D5  1.910  The scientist Robert Koch was born in Clausthal, Germany; he identified the bacterium that causes tuberculosis.
  D6  1.012  Streptomycin, discovered in 1943 by Albert Schatz, was the first antibiotic effective against tuberculosis.
Hop 2 query (needs the name found in hop 1): Alexander Fleming born
  D2  2.563  Alexander Fleming was born in 1881 on a farm near Darvel in Scotland.
  D1  2.024  Penicillin was discovered in 1928 by Alexander Fleming at St Mary's Hospital in London.
  D3  0.473  Marie Curie was born in Warsaw, in what was then the Kingdom of Poland.
```

The document with the answer (D2) is missing from the first retrieval, and the context that does come back points toward England or Germany.

## Hand-worked example in the article

The article also computes RAG-Sequence and RAG-Token by hand for a two-document, two-token answer, and shows that the retriever's gradient is posterior minus prior:

| | Prior p(z\|x) | P(Fleming \| z) | P(1928 \| z, Fleming) | Posterior p(z\|x,y) | Gradient |
| --- | --- | --- | --- | --- | --- |
| z1 | 0.6 | 0.9 | 0.2 | 0.5 | -0.1 |
| z2 | 0.4 | 0.3 | 0.9 | 0.5 | +0.1 |

RAG-Sequence gives 0.216 and RAG-Token gives 0.66 × 0.48 ≈ 0.317.
