"""
Dataset Acquisition Module for FlakyGuard AI.
Acquires the official dataset from the IEEE TSE 2023 paper:
"Test Flakiness Across Programming Languages" (Barbosa et al., 2023)
Repository: https://github.com/Test-Flaky/TSE22
"""

import os
import sys
import json
import time
import re
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
import pandas as pd

REPO_URL = "https://github.com/Test-Flaky/TSE22.git"
RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
EXCEL_FILENAME = "DataBase_GitHub.xlsx"
EXCEL_PATH = os.path.join(RAW_DIR, EXCEL_FILENAME)
METADATA_CACHE_PATH = os.path.join(RAW_DIR, "github_issues_metadata.json")
RAW_CSV_PATH = os.path.join(RAW_DIR, "raw_flaky_issues.csv")


def ensure_raw_directory(raw_dir: str = RAW_DIR) -> None:
    """Ensure data/raw directory exists."""
    os.makedirs(raw_dir, exist_ok=True)


def download_official_dataset(raw_dir: str = RAW_DIR, repo_url: str = REPO_URL) -> str:
    """
    Downloads or copies the official TSE22 Excel dataset into data/raw/.
    If already present, uses the local copy.
    """
    ensure_raw_directory(raw_dir)
    target_excel = os.path.join(raw_dir, EXCEL_FILENAME)
    if os.path.exists(target_excel) and os.path.getsize(target_excel) > 10000:
        print(f"[Dataset Acquisition] Found official dataset at: {target_excel}")
        return target_excel

    print(f"[Dataset Acquisition] Fetching official repository from {repo_url}...")
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        clone_cmd = ["git", "clone", "--depth", "1", repo_url, tmp_dir]
        subprocess.run(clone_cmd, check=True)
        src_excel = os.path.join(tmp_dir, "data", EXCEL_FILENAME)
        if not os.path.exists(src_excel):
            raise FileNotFoundError(f"Expected {EXCEL_FILENAME} in cloned repo under data/")
        import shutil
        shutil.copy2(src_excel, target_excel)
    print(f"[Dataset Acquisition] Official dataset stored at: {target_excel}")
    return target_excel


def fetch_single_issue_metadata(url: str, headers: dict, timeout: int = 8) -> dict:
    """
    Scrapes the title and initial issue description/comment body from a GitHub issue/PR URL.
    Returns a dict with url, title, body, and status_code.
    """
    result = {
        "url": url,
        "title": "",
        "body": "",
        "status_code": 0,
        "error": None
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result["status_code"] = resp.status
            html = resp.read().decode("utf-8", errors="ignore")

        soup = BeautifulSoup(html, "html.parser")

        # Extract issue/PR title
        title = ""
        title_el = soup.find("bdi", class_="js-issue-title")
        if title_el:
            title = title_el.get_text(strip=True)
        elif soup.title and soup.title.string:
            title = soup.title.string.strip()
            title = re.sub(r" · (Issue|Pull Request) #\d+ · .*$", "", title)
            title = re.sub(r"\s+·\s+GitHub$", "", title)
        result["title"] = title

        # Extract first issue comment body
        body = ""
        body_div = soup.find("div", class_=re.compile(r"comment-body|markdown-body"))
        if body_div:
            body = body_div.get_text(separator=" ", strip=True)
            body = re.sub(r"\s+", " ", body)
        result["body"] = body

    except urllib.error.HTTPError as e:
        result["status_code"] = e.code
        result["error"] = f"HTTP {e.code}"
    except Exception as e:
        result["status_code"] = 500
        result["error"] = str(e)

    return result


def fetch_all_issue_metadata(
    urls: list,
    cache_path: str = METADATA_CACHE_PATH,
    max_workers: int = 15,
    force_refresh: bool = False
) -> dict:
    """
    Fetches GitHub issue titles and descriptions concurrently with local JSON caching.
    """
    cached_metadata = {}
    if os.path.exists(cache_path) and not force_refresh:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_metadata = json.load(f)
            print(f"[Dataset Acquisition] Loaded {len(cached_metadata)} cached issue metadata entries.")
        except Exception as e:
            print(f"[Dataset Acquisition] Cache load error: {e}. Will re-fetch.")

    missing_urls = [u for u in urls if u and isinstance(u, str) and u not in cached_metadata]
    if not missing_urls:
        print("[Dataset Acquisition] All URLs are present in metadata cache.")
        return cached_metadata

    print(f"[Dataset Acquisition] Fetching metadata for {len(missing_urls)} missing URLs using {max_workers} threads...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    t0 = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(fetch_single_issue_metadata, url, headers): url
            for url in missing_urls
        }
        for future in as_completed(future_to_url):
            res = future.result()
            cached_metadata[res["url"]] = res
            completed += 1
            if completed % 100 == 0 or completed == len(missing_urls):
                elapsed = time.time() - t0
                print(f"  -> Fetched {completed}/{len(missing_urls)} ({completed/elapsed:.1f} req/s)")

    # Save updated cache
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cached_metadata, f, indent=2, ensure_ascii=False)
    print(f"[Dataset Acquisition] Saved {len(cached_metadata)} issue metadata items to {cache_path}")
    return cached_metadata


def build_raw_dataset(
    raw_dir: str = RAW_DIR,
    fetch_online: bool = True,
    max_workers: int = 15
) -> pd.DataFrame:
    """
    Loads official Excel spreadsheet, enriches with issue metadata, and produces raw_flaky_issues.csv.
    """
    excel_path = download_official_dataset(raw_dir)
    print(f"[Dataset Acquisition] Reading sheet 'Data' from {excel_path}...")

    # Data is row 2 indexed (skiprows=2 gives proper column names)
    df = pd.read_excel(excel_path, sheet_name="Data", skiprows=2)

    # Clean leading/trailing spaces from column headers
    df.columns = [str(c).strip() for c in df.columns]

    # Standardize column naming
    rename_map = {
        "ID": "issue_id",
        "Repository/Project": "repository",
        "Language": "language",
        "Status": "flakiness_status",
        "Labels": "labels",
        "Status Issue": "issue_status",
        "Year": "year",
        "Category": "root_cause_category",
        "Reviewers": "root_cause_reviewers",
        "Description": "problem_description_excerpt",
        "URL Issue": "url_issue",
        "Category.1": "fix_strategy_category",
        "Reviewers.1": "fix_strategy_reviewers",
        "Description.1": "fix_description_excerpt",
        "M1\nNumber of days until issue is closed": "days_to_close",
        "M2\nNumber of comments until issue is closed": "comments_count",
        "Domain": "domain"
    }
    df = df.rename(columns=rename_map)

    # Collect unique URLs
    urls = [u for u in df["url_issue"].dropna().unique() if isinstance(u, str) and u.startswith("http")]

    if fetch_online:
        metadata = fetch_all_issue_metadata(urls, max_workers=max_workers)
    elif os.path.exists(METADATA_CACHE_PATH):
        with open(METADATA_CACHE_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # Map fetched titles and bodies
    titles = []
    bodies = []
    for url in df["url_issue"]:
        meta = metadata.get(url, {})
        titles.append(meta.get("title", ""))
        bodies.append(meta.get("body", ""))

    df["issue_title"] = titles
    df["issue_body"] = bodies

    # Save to raw CSV
    output_csv = os.path.join(raw_dir, "raw_flaky_issues.csv")
    df.to_csv(output_csv, index=False)
    print(f"[Dataset Acquisition] Raw dataset built successfully with shape {df.shape} -> {output_csv}")
    return df


if __name__ == "__main__":
    build_raw_dataset(fetch_online=True)
