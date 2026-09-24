# Part 00 · Why we still need retrieval in the age of agents

The article argues that the fixed 2023-style RAG pipeline is fading, while retrieval itself is becoming one of the capabilities agents use most. One of its arguments is a back-of-the-envelope calculation: even a small corpus does not fit in a 1M-token context window, and reading the whole corpus on every query costs far more than retrieving a few documents first.

`corpus_cost.py` reproduces that calculation on SciFact.

```bash
python corpus_cost.py
```

The first run downloads the BEIR version of SciFact (a few MB) into `../data/`.

## Estimate used in the article

The article uses the published statistics (5,183 abstracts, ~215 words each), the rough rule 1 token ≈ 0.75 English words, and an input price of $2 per million tokens:

| Approach | Input tokens per query | Input cost per query |
| --- | --- | --- |
| Whole corpus in context | ~1.49M | ~$2.97 |
| Whole corpus, prompt-cache hit (0.1x price) | ~1.49M | ~$0.30 |
| Retrieve top 10 abstracts, read only those | ~2,900 | ~$0.006 |

Every flag can be changed, so you can plug in your own window size, price and top-k.
