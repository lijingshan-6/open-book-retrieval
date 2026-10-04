# Part 03 · Chunking and granularity

A chunk is both the unit the retriever scores and the unit the generator reads. This part measures how chunk size and chunking strategy change whether the evidence for a question ends up in the context.

## Data

SciFact's corpus is short abstracts with nothing to chunk, so this part uses the **QASPER** test set ([Dasigi et al., NAACL 2021](https://arxiv.org/abs/2105.03011), CC BY 4.0): 416 NLP papers with full text. Questions were written by people who had read only the title and abstract, and other annotators marked the evidence sentences in the full text.

We keep answerable questions whose evidence is plain text found verbatim in the paper: **1,056 questions**. Each question is answered by retrieving within its own paper with BM25 (k1 = 0.9, b = 0.4). A question counts as found if the retrieved text covers at least 80% of one of its evidence spans.

```bash
mkdir -p ../data/qasper
curl -L -o ../data/qasper/qasper-test.tgz https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz
tar -xzf ../data/qasper/qasper-test.tgz -C ../data/qasper
python chunk_eval.py
```

Step 5 also uses the Part 02 paper. Run `../p02_document_parsing/parse_compare.py` once to download it. The full run takes about a minute.

## Results

**Comparing at a fixed k is misleading.** With the top 5 chunks, bigger chunks win, because they hand over more text. At 1,024 words per chunk that is 3,217 words on average, close to a whole paper (median length about 3,200 words).

| Chunk size (words) | Evidence found, top 5 | Words handed over |
| --- | --- | --- |
| 32 | 16.6% | 159 |
| 128 | 50.9% | 627 |
| 512 | 90.8% | 2,277 |
| 1,024 | 99.3% | 3,217 |

**At a fixed context budget the curve turns over.** Chunks are added in rank order until B words are filled:

| Chunk size (words) | B = 256 | B = 512 | B = 1,024 |
| --- | --- | --- | --- |
| 32 | 23.3% | 37.6% | 55.3% |
| 64 | 28.3% | 42.4% | 62.1% |
| 128 | **28.8%** | **45.0%** | **64.9%** |
| 256 | 23.3% | 40.4% | 64.5% |
| 512 | 17.9% | 36.2% | 60.7% |
| 1,024 | 11.9% | 26.4% | 52.2% |

**Where you cut matters more than how big the chunks are** (B = 512):

| Chunking | Evidence found |
| --- | --- |
| Fixed 128 words | 45.0% |
| Fixed 128, 25% overlap | 46.1% |
| Sentences packed up to 128 words, within a paragraph | 54.1% |
| One paragraph per chunk (mean 71 words) | **54.6%** |
| One section per chunk | 41.4% |
| Search 64-word sentence packs, return the paragraph | 52.6% |

With a looser 50% coverage threshold the gap between fixed 128 (52.3%) and paragraphs (56.1%) shrinks to about 4 points. Roughly half of the advantage comes from evidence sentences being cut in two.

**Title and section prefixes barely help BM25.** Within one paper every chunk gets the same title. With all 416 papers in one index, the prefix moves fixed-64 chunks from 11.4% to 12.2% and paragraph chunks from 14.7% to 15.1%. A generic prefix does not say what "this effect" refers to.

**A chunk can carry the answer without its subject.** The Part 02 paper is split into one sentence per chunk (180 chunks). The query asks how long the drop in alcohol consumption lasted after the free beverages stopped. The two sentences with the answer ("this effect persisted ... 8 weeks") rank **#125 and #65**, and the sentence before one of them, which names the intervention, ranks #3. With a one-line, chunk-specific context prefix, written by hand in the spirit of [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval), they rank **#9 and #2**.

All numbers come from one dataset and a lexical retriever. Dense retrieval adds its own constraint: many embedding models truncate input at 512 tokens. That is covered in the parts on embeddings.
