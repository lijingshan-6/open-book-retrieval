# Open Book: Retrieval Principles in the Age of Agents

Companion code for *开卷：Agent 时代的检索原理* (Open Book: Retrieval Principles in the Age of Agents), a 29-part series in Chinese by Kuangye (旷野) on how large language models find, select, use and verify external knowledge.

The series covers retrieval as a capability rather than any one pipeline or framework. Each article comes with a hand-worked example or a small experiment you can run yourself, and this repository holds the code for them.

## Dataset

Unless an article says otherwise, experiments use the [BEIR](https://github.com/beir-cellar/beir) version of [SciFact](https://github.com/allenai/scifact):

- 5,183 scientific abstracts (~215 words on average), 300 test queries
- about 1.13 relevant documents per query, binary relevance
- the original release adds sentence-level rationales and SUPPORT / CONTRADICT / NOT ENOUGH INFO labels, used later in the series for citation, abstention and hallucination-detection experiments

Queries in SciFact are scientific claims, not questions. Where SciFact does not fit a topic (long-document chunking, multi-hop retrieval, graded relevance), the article uses a small hand-built example and explains why.

Data is downloaded on first run into `data/`, which is not committed.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+ is recommended.

## Articles

| Part | Topic | Code |
| --- | --- | --- |
| 00 | Why we still need retrieval in the age of agents | [p00_why_retrieval](p00_why_retrieval/) |
| 01 | Classic RAG as a reference point | [p01_classic_rag](p01_classic_rag/) |

More parts are added as they are published.

## License

[MIT](LICENSE)
