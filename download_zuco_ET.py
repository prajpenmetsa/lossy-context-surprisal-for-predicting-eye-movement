"""
ZuCo-1 Sentence Reading — corrected_ET.mat downloader
Downloads ONLY *_corrected_ET.mat files from all subject folders
under task1-SR/Preprocessed on OSF (project q3zws).

Requirements:  pip install requests tqdm
Usage:         python download_zuco_ET.py
Output:        ./zuco_ET/<subject>/<file>.mat
"""

import os
import time
import requests
from tqdm import tqdm

# ── OSF project config ────────────────────────────────────────────────────────
PROJECT_ID   = "q3zws"
BASE_API     = "https://api.osf.io/v2"
STORAGE_ROOT = f"{BASE_API}/nodes/{PROJECT_ID}/files/osfstorage/"
OUT_DIR      = "zuco_ET"
SESSION      = requests.Session()
SESSION.headers.update({"User-Agent": "zuco-downloader/1.0"})

# ── helpers ───────────────────────────────────────────────────────────────────
def api_get(url):
    """GET with simple retry on rate-limit (429) or transient errors."""
    for attempt in range(5):
        r = SESSION.get(url, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 10))
            print(f"  Rate-limited, waiting {wait}s…")
            time.sleep(wait)
        else:
            print(f"  HTTP {r.status_code} on {url}, retry {attempt+1}/5")
            time.sleep(3)
    raise RuntimeError(f"Failed to fetch {url} after 5 attempts")

def list_folder(files_url):
    """Return all items in an OSF folder, handling pagination."""
    items = []
    url = files_url
    while url:
        data = api_get(url)
        items.extend(data["data"])
        url = data["links"].get("next")   # None when no more pages
    return items

def download_file(download_url, dest_path):
    """Stream-download a file with a progress bar."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    r = SESSION.get(download_url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True,
        desc=os.path.basename(dest_path), leave=False
    ) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

# ── navigation ────────────────────────────────────────────────────────────────
def find_folder(items, name):
    for item in items:
        if item["attributes"]["kind"] == "folder" and \
           item["attributes"]["name"] == name:
            return item["relationships"]["files"]["links"]["related"]["href"]
    raise FileNotFoundError(f"Folder '{name}' not found")

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=== ZuCo-1 corrected_ET.mat downloader ===\n")

    # Navigate: root → task1- SR → Preprocessed
    print("Navigating OSF folder structure…")
    root_items     = list_folder(STORAGE_ROOT)
    task1_url      = find_folder(root_items, "task1- SR")
    task1_items    = list_folder(task1_url)
    preproc_url    = find_folder(task1_items, "Preprocessed")
    preproc_items  = list_folder(preproc_url)

    # Collect subject folders (3-letter codes like ZAB, ZDM, …)
    subject_folders = [
        item for item in preproc_items
        if item["attributes"]["kind"] == "folder"
        and len(item["attributes"]["name"]) == 3
        and item["attributes"]["name"].startswith("Z")
    ]
    print(f"Found {len(subject_folders)} subject folders: "
          f"{[s['attributes']['name'] for s in subject_folders]}\n")

    total_downloaded = 0
    total_skipped    = 0

    for subj_item in subject_folders:
        subj_name = subj_item["attributes"]["name"]
        subj_url  = subj_item["relationships"]["files"]["links"]["related"]["href"]
        subj_files = list_folder(subj_url)

        et_files = [
            f for f in subj_files
            if f["attributes"]["kind"] == "file"
            and f["attributes"]["name"].endswith("_corrected_ET.mat")
        ]

        print(f"{subj_name}: {len(et_files)} ET files found")

        for f in et_files:
            fname     = f["attributes"]["name"]
            dest_path = os.path.join(OUT_DIR, subj_name, fname)
            dl_url    = f["links"]["download"]

            if os.path.exists(dest_path):
                print(f"  ↷ skipping {fname} (already exists)")
                total_skipped += 1
                continue

            print(f"  ↓ {fname}")
            download_file(dl_url, dest_path)
            total_downloaded += 1

        print()

    print(f"Done. Downloaded: {total_downloaded}  Skipped: {total_skipped}")
    print(f"Files saved to: {os.path.abspath(OUT_DIR)}/")

if __name__ == "__main__":
    main()
