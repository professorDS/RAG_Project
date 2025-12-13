import fitz  # PyMuPDF
import easyocr
from pathlib import Path
from src.utils.logger import logger

# Output folder
OUTPUT_FOLDER = Path("data/processed_text")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Initialize EasyOCR (English only for performance)
reader = easyocr.Reader(['en'], gpu=False, verbose=False)


def extract_text(pdf_path):
    """
    Hybrid extraction:
    1. Try PyMuPDF for digital text (fast)
    2. Use EasyOCR only when needed (for scanned pages)
    """
    doc = fitz.open(pdf_path)
    all_text = []

    logger.info(f"Processing PDF: {pdf_path.name}")

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        # Step 1: Fast text extraction (digital PDFs)
        text = page.get_text().strip()

        # If text exists and is meaningful, skip OCR → VERY fast
        if len(text) > 20:
            all_text.append(text)
            continue

        # Step 2: Fallback to OCR only when digital text is missing
        logger.info(f"OCR needed on page {page_num+1} of {pdf_path.name}")

        pix = page.get_pixmap(dpi=150)  # Lower DPI = faster
        image = pix.tobytes("png")

        results = reader.readtext(
            image,
            detail=0,          # only return text
            paragraph=True     # combine lines
        )

        ocr_text = "\n".join(results)
        all_text.append(ocr_text)

    doc.close()
    return "\n\n".join(all_text)


def process_pdf(pdf_path):
    pdf_path = Path(pdf_path)
    output_file = OUTPUT_FOLDER / f"{pdf_path.stem}.txt"

    # Skip already processed PDFs (safe restart)
    if output_file.exists():
        logger.info(f"Skipping {pdf_path.name} (already processed)")
        return

    try:
        text = extract_text(pdf_path)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)

        logger.info(f"Saved processed text: {output_file}")

    except Exception as e:
        logger.error(f"Failed to process {pdf_path.name}: {e}")


def process_all_pdfs(pdf_folder="data/pdfs"):
    folder = Path(pdf_folder)
    pdfs = list(folder.glob("*.pdf"))

    logger.info(f"Found {len(pdfs)} PDFs to process.")

    for pdf in pdfs:
        process_pdf(pdf)


if __name__ == "__main__":
    process_all_pdfs()
