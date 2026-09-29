"""Part 02: the same paper, parsed from PDF and from JATS XML.

The paper is Yoshimoto et al., "Effect of provision of non-alcoholic beverages on alcohol
consumption: a randomized controlled study", BMC Medicine 21:379 (2023), CC BY 4.0,
doi:10.1186/s12916-023-03085-1, PMCID PMC10544561.

Usage:
    python parse_compare.py

Downloads the PDF (from the publisher) and the JATS XML (from Europe PMC) into ../data/p02/,
then compares what naive PDF text extraction produces with the structured XML:

  1. what a PDF actually stores (a snippet of the page content stream)
  2. how much of the extracted text is not body text
  3. words broken by end-of-line hyphenation
  4. a table, flattened by PDF extraction vs. kept as rows in XML
  5. metadata available from each source
  6. a scanned version of a page: no text layer at all
  7. retrieval over 100-word chunks built from each source

Requires pypdf, PyMuPDF (fitz) and lxml.
"""

import re
import urllib.request
from pathlib import Path

import fitz
import pypdf
from lxml import etree

from bm25 import BM25

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "p02"
PDF_URL = "https://bmcmedicine.biomedcentral.com/counter/pdf/10.1186/s12916-023-03085-1.pdf"
XML_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10544561/fullTextXML"
IMRAD = ("Background", "Methods", "Results", "Discussion", "Conclusions")


def fetch(url, path, magic):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        if not data.startswith(magic):
            raise SystemExit(
                f"{url} did not return the expected file (publishers sometimes answer scripts with a "
                f"browser check). Download it manually and rerun:\n  curl -L -A Mozilla/5.0 -o {path} {url}"
            )
        path.write_bytes(data)
    return path


def words(text):
    return re.findall(r"\S+", text)


def chunks(text, size=100):
    w = words(text)
    return [" ".join(w[i:i + size]) for i in range(0, len(w), size)]


def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main():
    pdf_path = fetch(PDF_URL, DATA_DIR / "paper.pdf", b"%PDF")
    xml_path = fetch(XML_URL, DATA_DIR / "paper.xml", b"<?xml")

    doc = fitz.open(pdf_path)
    pypdf_pages = [p.extract_text() for p in pypdf.PdfReader(pdf_path).pages]
    pypdf_text = "\n".join(pypdf_pages)
    mupdf_text = "\n".join(p.get_text() for p in doc)

    xml = etree.parse(str(xml_path))
    body_secs = [s for s in xml.find(".//body").findall("./sec") if s.findtext("title") in IMRAD]
    xml_body = "\n".join(" ".join(p.itertext()) for s in body_secs for p in s.iter("p"))

    section("1. What a PDF stores: drawing instructions (page 4, first lines of 'Intervention')")
    raw = doc[3].read_contents().decode("latin-1")
    start = raw.find("/T1_2 1 Tf")
    print(raw[start:start + 360])

    section("2. How much of the extracted PDF text is body text")
    ref_start = pypdf_text.rfind("References")
    pdf_words = len(words(pypdf_text))
    ref_words = len(words(pypdf_text[ref_start:]))
    page_numbers = re.findall(r"Page \d+ of \d+", pypdf_text)
    running_heads = re.findall(r"Yoshimoto\s+et\s+al\.\s+BMC Medicine\s+\(2023\) 21:379", pypdf_text)
    print(f"PDF (pypdf), all text:           {pdf_words:>6,} words")
    print(f"XML, IMRaD body paragraphs only: {len(words(xml_body)):>6,} words")
    print(f"  reference list in the PDF text: {ref_words:>5,} words ({ref_words / pdf_words:.1%})")
    print(f"  page numbers ('Page x of 10'):  {len(page_numbers):>5}")
    print(f"  running heads (journal, year):  {len(running_heads):>5}  (one per page, all {doc.page_count} pages)")
    print(f"XML sections: {[s.findtext('title') for s in body_secs]}")
    methods = next(s for s in body_secs if s.findtext("title") == "Methods")
    print(f"  Methods subsections: {[c.findtext('title') for c in methods.findall('./sec')]}")
    print(f"  references: {len(xml.findall('.//ref'))} <ref> elements, kept apart from the body")

    section("3. Words broken by end-of-line hyphenation")
    xml_all = " ".join(xml.getroot().itertext())
    for name, text, pattern in [("pypdf", pypdf_text, r"([A-Za-z]+) -\s*\n\s*([a-z]+)"),
                                ("PyMuPDF", mupdf_text, r"([A-Za-z]+)-\n([a-z]+)")]:
        pairs = re.findall(pattern, text)
        compound = [a + "-" + b for a, b in pairs if (a + "-" + b) in xml_all]
        print(f"{name:<8} {len(pairs):>3} line-end breaks, e.g. {[a + '|' + b for a, b in pairs[:4]]}")
        print(f"         of which {len(compound)} are real hyphenated words that must keep the hyphen, e.g. {compound[:3]}")

    section("4. Table 2: PDF extraction vs. XML rows")
    i = pypdf_text.find("Table 2 Characteristics")
    print("pypdf:")
    print(pypdf_text[i:i + 520])
    print("\nXML:")
    table = xml.find('.//table-wrap[@id="Tab2"]')
    for tr in table.findall(".//tr")[:4]:
        print(["".join(td.itertext()).strip() for td in tr])
    print("\nWhere the table lands in the running text (pypdf):")
    j = pypdf_text.find("at each time point from Week 4 to")
    k = pypdf_text.find("Week 20. The main outcome")
    print(f"  '...from Week 4 to' [{len(words(pypdf_text[j:k])) - 8} words of tables, captions and page header] 'Week 20. The main outcome...'")

    section("5. Metadata")
    print(f"PDF document info: title={doc.metadata.get('title')!r}, author={doc.metadata.get('author')!r}")
    meta = xml.find(".//article-meta")
    authors = [f"{n.findtext('given-names')} {n.findtext('surname')}"
               for n in meta.findall(".//contrib-group/contrib/name")]
    print(f"XML title:   {''.join(meta.find('.//article-title').itertext())}")
    print(f"XML authors: {authors[:3]} ... ({len(authors)} total)")
    print(f"XML year: {meta.findtext('.//pub-date/year')}, DOI: {meta.findtext('.//article-id[@pub-id-type=\"doi\"]')}")

    section("6. A scanned copy of page 4")
    scan = fitz.open()
    pix = doc[3].get_pixmap(dpi=150)
    page = scan.new_page(width=doc[3].rect.width, height=doc[3].rect.height)
    page.insert_image(page.rect, pixmap=pix)
    print(f"characters extracted from the original page 4: {len(doc[3].get_text()):,}")
    print(f"characters extracted from the scanned copy:    {len(page.get_text()):,}")

    section("7. Retrieval over 100-word chunks")
    pdf_chunks, xml_chunks = chunks(pypdf_text), chunks(xml_body)
    ref_chunk = len(words(pypdf_text[:ref_start])) // 100
    print(f"chunks: PDF {len(pdf_chunks)}, XML body {len(xml_chunks)}")
    query = "correlation between non-alcoholic beverage consumption and alcohol consumption in the intervention group"
    print(f"query: {query}")
    for name, cs in [("PDF", pdf_chunks), ("XML", xml_chunks)]:
        ranked = [i for i, _ in BM25(dict(enumerate(cs))).search(query, len(cs))]
        answer = next(i for i, c in enumerate(cs) if re.search(r"0\.500, n\s*=\s*54", c))
        print(f"\n  {name}: the Results chunk with the answer (rho = -0.500, n = 54) ranks "
              f"#{ranked.index(answer) + 1} of {len(cs)}")
        print(f"  {cs[answer][:420]}")


if __name__ == "__main__":
    main()
