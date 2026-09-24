# Part 00 · Why we still need retrieval in the age of agents

The article argues that the fixed 2023-style RAG pipeline is fading, while retrieval itself is becoming one of the capabilities agents use most. One of its arguments is a back-of-the-envelope calculation: even a small corpus does not fit in a 1M-token context window, and reading the whole corpus on every query costs far more than retrieving a few documents first.

`corpus_cost.py` reproduces that calculation on SciFact.

```bash
python corpus_cost.py
```

The first run downloads the BEIR version of SciFact (about 2.8 MB) into `../data/`. If Python's download fails with an SSL certificate error, fetch the file with curl and rerun:

```bash
curl -L -o ../data/scifact.zip https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
```

## Measured output

```
Documents:                    5,183
Avg words (title + abstract): 214.6
Total words:                  1,112,417
Estimated tokens (/0.75):     1,483,223 (1.48x a 1,000,000-token window)
cl100k_base tokens (reference): 1,657,885

Input cost per query at $2.0/MTok:
  Whole corpus in context            1,483,223 tokens   $2.9664
  Whole corpus, prompt-cache hit     1,483,223 tokens   $0.2966
  Retrieve top-10, then read             2,862 tokens   $0.0057
  Retrieval is 518x cheaper (no cache), 52x cheaper (cache hit)
```

The measured tokenizer count is about 12% higher than the 0.75-words-per-token estimate, so the estimate is on the low side.

## Estimate used in the article

The article uses the published statistics (5,183 abstracts, ~215 words each), the rough rule 1 token ≈ 0.75 English words, and an input price of $2 per million tokens:

| Approach | Input tokens per query | Input cost per query |
| --- | --- | --- |
| Whole corpus in context | ~1.49M | ~$2.97 |
| Whole corpus, prompt-cache hit (0.1x price) | ~1.49M | ~$0.30 |
| Retrieve top 10 abstracts, read only those | ~2,900 | ~$0.006 |

Every flag can be changed, so you can plug in your own window size, price and top-k.
