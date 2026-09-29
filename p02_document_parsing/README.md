# Part 02 · Document parsing sets the ceiling for retrieval

A PDF stores drawing instructions: which glyphs to paint at which coordinates. Paragraphs, sections, tables and reading order have to be inferred back from those coordinates, and the inference goes wrong in predictable ways. This part parses one open-access paper twice, once from the PDF and once from its JATS XML, and compares the results.

The paper: Yoshimoto et al., *Effect of provision of non-alcoholic beverages on alcohol consumption: a randomized controlled study*, BMC Medicine 21:379 (2023), [doi:10.1186/s12916-023-03085-1](https://doi.org/10.1186/s12916-023-03085-1), CC BY 4.0. It is a two-column, ten-page PDF with three tables.

```bash
python parse_compare.py
```

The script downloads the PDF from the publisher and the XML from Europe PMC into `../data/p02/`. Publishers sometimes answer scripted requests with a browser check. If that happens, the script stops and prints a `curl` command to fetch the file manually.

Requires `pypdf`, `PyMuPDF` and `lxml` (see the top-level `requirements.txt`). `bm25.py` is the same minimal BM25 used in Part 01.

## What it measures, and what it found

| Check | PDF (plain text extraction) | JATS XML |
| --- | --- | --- |
| Words | 7,362 (everything on every page) | 4,927 (IMRaD body paragraphs) |
| Reference list | 952 words mixed into the text (12.9%) | 32 `<ref>` elements, kept apart |
| Running heads and page numbers | 19, on all 10 pages | none |
| Line-end hyphenation | 143 breaks with PyMuPDF (e.g. `coef-ficient`), 14 of them real compounds such as `low-alcohol`; pypdf gives `signifi - cant` | none |
| Tables | flattened, header split over seven lines, footnote markers glued to values (`0.74a`) | one `<td>` per cell, footnote markers as `<sup>` |
| Tables inside sentences | 245 words of tables, captions and header land between "from Week 4 to" and "Week 20." | tables stored separately |
| Metadata | title and first author only | all authors, affiliations, year, DOI, license |
| Scanned copy of page 4 | 0 characters (original page: 5,542) | n/a |

Retrieval over 100-word chunks with BM25, for the query *correlation between non-alcoholic beverage consumption and alcohol consumption in the intervention group*:

- XML: the Results chunk with the answer (ρ = −0.500, n = 54) ranks **#1 of 50**.
- PDF: the chunk with the same sentence ranks **#6 of 74**. A third of it is the tail of Table 3 plus a running head, the sentence before it is cut in half, and `significant` is split in two:

```
IQR) 6.0 (12.0) 6.0 (16.0) 6.0 (11.0) 0.92d Page 7 of 10 Yoshimoto et al. BMC Medicine (2023) 21:379 Week 20. The main outcome, alcohol consumption at Week 12, was not significantly correlated with non- alcoholic beverage consumption in the control group (ρ = − 0.063, n = 67, p = 0.615). In contrast, a signifi - cant negative correlation was noted in the intervention group (ρ = − 0.500, n = 54, p < 0.001, Fig. 3) and
```

This is one paper, one query and the simplest retriever, so treat it as an illustration of the mechanism, not a benchmark.

GROBID, Docling, Nougat and olmOCR are discussed in the article but not run here. They need a Java service, large model downloads or a GPU.
