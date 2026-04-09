"""CLI entrypoint for the ICD-10 PDF parser."""

from __future__ import annotations

import argparse
import logging

from mnc.datasets.icd10_pdf import DEFAULT_PDF_URL, parse_icd10_official_pdf


def main() -> None:
    """Run the ICD-10 PDF bronze parser."""
    parser = argparse.ArgumentParser(
        description="Parse official ICD-10 PDF into page-level DocumentRecords.",
    )
    parser.add_argument(
        "pdf_source",
        nargs="?",
        default=DEFAULT_PDF_URL,
        help="URL or local path to the PDF (default: official ICD-10 PDF)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="data/bronze/icd10_official_pdf/primary",
        help="Output directory for artifacts",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parse_icd10_official_pdf(
        pdf_source=args.pdf_source,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
