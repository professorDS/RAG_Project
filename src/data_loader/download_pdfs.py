import hashlib
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from src.utils.logger import logger

# Paths
CSV_PATH = "D:\\RAG_project\\RAG_project\\data\\announcment_data.csv"
PDF_FOLDER = Path("data/pdfs")
LOG_FILE = "download_log_all.csv"

PDF_FOLDER.mkdir(parents=True, exist_ok=True)

TIMEOUT = 60
CHUNK_SIZE = 8192


def sanitize_filename(name: str):
    invalid = '<>:"/\\|?*'
    for ch in invalid:
        name = name.replace(ch, "_")
    return name[:200]


def filename_from_url(url: str):
    try:
        base = url.split("/")[-1]
        if "." in base and len(base) < 200:
            base = sanitize_filename(base)
        else:
            base = "document"

        url_hash = hashlib.sha1(url.encode()).hexdigest()[:8]

        if base.endswith(".pdf"):
            return f"{base[:-4]}_{url_hash}.pdf"
        return f"{base}_{url_hash}.pdf"
    except:
        return f"document_{hashlib.sha1(url.encode()).hexdigest()[:8]}.pdf"


def create_session():
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    return session


def download_stream(url, dest, session):
    try:
        response = session.get(url, stream=True, timeout=TIMEOUT)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))

        with open(dest, "wb") as f:
            if total > 0:
                with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name, leave=False) as pbar:
                    for chunk in response.iter_content(CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
        return True, "Downloaded"
    except Exception as e:
        return False, str(e)


def detect_pdf_column(df: pd.DataFrame):
    for col in df.columns:
        sample = df[col].astype(str).head(15).tolist()
        for v in sample:
            if "http" in v and "pdf" in v.lower():
                logger.info(f"Detected PDF link column: {col}")
                return col
    raise ValueError("No PDF URL column found automatically.")


def main():
    logger.info("Starting PDF download process...")

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return

    pdf_col = detect_pdf_column(df)
    urls = df[pdf_col].dropna().astype(str).tolist()

    logger.info(f"Total links found: {len(urls)}")

    session = create_session()
    results = []

    for i, url in enumerate(urls):
        url = url.strip()
        filename = filename_from_url(url)
        dest = PDF_FOLDER / filename

        logger.info(f"[{i+1}/{len(urls)}] Downloading {filename}")

        if dest.exists():
            logger.info(f"Already exists: {filename}")
            results.append({"url": url, "file": filename, "status": "Exists"})
            continue

        success, msg = download_stream(url, dest, session)

        if not success and dest.exists():
            dest.unlink()

        results.append({"url": url, "file": filename, "status": "Success" if success else "Failed", "reason": msg})
        logger.info(f"{filename}: {msg}")

    pd.DataFrame(results).to_csv(LOG_FILE, index=False)
    logger.info("Download process complete.")


if __name__ == "__main__":
    main()
