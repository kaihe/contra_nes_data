import re
from pathlib import Path


DOC_NAME = re.compile(r"^\d{4}-(?:design|exp)-[a-z0-9]+(?:-[a-z0-9]+)*\.md$")
EXP_HEADINGS = [
    "## 1. Goal",
    "## 2. Setup",
    "## 3. Evaluation metrics",
    "## 4. Conclusion",
]
LEGACY_DESIGN_STRUCTURE = {
    "0001-design-boss-search-curriculum.md",
    "0002-design-spread-only-validation.md",
    "0003-design-incremental-spread-scaling.md",
    "0004-design-tokenized-datahouse.md",
    "0006-design-cloud-trace-worker.md",
}
GENERIC_DESIGN_HEADING = re.compile(
    r"^## (?:\d+\.\s*)?(?:decision|why(?:\s.*)?|evidence|the design|design|"
    r"what was rejected(?:, and why)?|rejected alternatives|risks(?: and gates)?|"
    r"sequencing|rationale|caveats|recommendation|appendix — provenance)$",
    re.IGNORECASE,
)


def test_numbered_doc_names() -> None:
    doc_dir = Path(__file__).resolve().parents[1] / "doc"
    invalid = sorted(
        path.name
        for path in doc_dir.glob("[0-9][0-9][0-9][0-9]-*.md")
        if not DOC_NAME.fullmatch(path.name)
    )
    assert not invalid, f"invalid numbered doc names: {invalid}"


def test_experiment_doc_sections() -> None:
    doc_dir = Path(__file__).resolve().parents[1] / "doc"
    invalid = {}
    for path in sorted(doc_dir.glob("[0-9][0-9][0-9][0-9]-exp-*.md")):
        headings = [line for line in path.read_text().splitlines() if line.startswith("## ")]
        if headings != EXP_HEADINGS:
            invalid[path.name] = headings
    assert not invalid, f"experiment docs must use exactly four sections: {invalid}"


def test_new_design_docs_use_feature_sections() -> None:
    doc_dir = Path(__file__).resolve().parents[1] / "doc"
    invalid = {}
    for path in sorted(doc_dir.glob("[0-9][0-9][0-9][0-9]-design-*.md")):
        if path.name in LEGACY_DESIGN_STRUCTURE:
            continue
        headings = [line for line in path.read_text().splitlines()
                    if GENERIC_DESIGN_HEADING.fullmatch(line)]
        if headings:
            invalid[path.name] = headings
    assert not invalid, f"design docs must use feature-named sections: {invalid}"
