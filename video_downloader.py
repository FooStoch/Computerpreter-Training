import argparse
import hashlib
import json
import logging
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import Request, urlopen

logging.basicConfig(
    filename=f"download_{int(time.time())}.log",
    filemode="w",
    level=logging.DEBUG
)
logging.getLogger().addHandler(logging.StreamHandler())

YTDLP = "yt-dlp"
PROJECT_DIR = Path(__file__).resolve().parent
COOKIES_FILE = PROJECT_DIR / "cookies.txt"
OUTPUT_DIR = PROJECT_DIR / "raw_videos"
JSON_FILE = PROJECT_DIR / "collected_data" / "all_target_sign_entries_with_synonyms.json"
DOWNLOAD_INDEX_FILE = "download_index.json"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0 Safari/537.36"
)


def normalize_url(url):
    url = str(url).strip()

    if url.startswith(("www.youtube.com", "youtube.com", "youtu.be")):
        return "https://" + url

    return url


def is_youtube_url(url):
    hostname = (urlparse(normalize_url(url)).hostname or "").lower()
    return hostname in {
        "youtu.be",
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
    }


def get_yt_id(url):
    url = normalize_url(url)
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()

    if hostname == "youtu.be":
        return parsed.path.lstrip("/")

    if hostname in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        if parsed.path == "/watch":
            query = parse_qs(parsed.query)
            return query.get("v", [None])[0]

        if parsed.path.startswith(("/embed/", "/v/", "/shorts/")):
            parts = parsed.path.split("/")
            return parts[2] if len(parts) > 2 else None

    return None


def file_extension_from_url(url):
    suffix = Path(unquote(urlparse(url).path)).suffix.lower()
    if suffix and len(suffix) <= 10:
        return suffix
    return ".bin"


def unique_download_id(url, entry):
    if is_youtube_url(url):
        yt_id = get_yt_id(url)
        if yt_id:
            return yt_id

    video_id = entry.get("video_id")
    if video_id not in (None, ""):
        return f"{entry.get('dataset', 'video').lower()}_{video_id}"

    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"url_{digest}"


def build_output_path(url, entry, output_dir):
    target = str(entry.get("target_sign", "unknown")).lower()
    matched = str(entry.get("matched_sign", target)).lower().replace(" ", "_")
    dataset = str(entry.get("dataset", "dataset")).lower().replace("-", "")
    download_id = unique_download_id(url, entry)
    extension = ".mp4" if is_youtube_url(url) else file_extension_from_url(url)
    filename = f"{dataset}_{target}_{matched}_{download_id}{extension}"
    return output_dir / filename


def check_command(command, name):
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        first_line = (result.stdout or result.stderr).strip().splitlines()[0]
        logging.info(f"{name}: {first_line}")
    except Exception as e:
        logging.error(f"{name} check failed: {e}")
        raise SystemExit(1)


def check_python_version():
    if sys.version_info < (3, 10):
        logging.warning(
            "You are using Python %s.%s. yt-dlp has deprecated Python 3.9. "
            "Install Python 3.10+ soon.",
            sys.version_info.major,
            sys.version_info.minor
        )


def build_ytdlp_command(url, output_path):
    cmd = [
        YTDLP,

        # Avoid the YouTube web client path that can trigger SABR / HTTP 403.
        "--extractor-args", "youtube:player_client=android",

        # Retries help with intermittent YouTube throttling/network weirdness.
        "--retries", "10",
        "--fragment-retries", "10",

        # Slow down a bit so you look less like a bursty scraper.
        "--sleep-requests", "1",
        "--sleep-interval", "2",
        "--max-sleep-interval", "5",

        # Prefer merged MP4 output.
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo*+bestaudio/best",
        "--merge-output-format", "mp4",

        "-o", str(output_path),
    ]

    # Use exported cookies.txt instead of Chrome's live encrypted cookie DB.
    if COOKIES_FILE.exists():
        cmd.extend(["--cookies", str(COOKIES_FILE)])
    else:
        logging.warning(
            "cookies.txt not found. Downloads may fail with bot checks or HTTP 403."
        )

    cmd.append(url)
    return cmd


def load_entries(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)


def group_entries_by_url(entries):
    grouped = defaultdict(list)
    missing_url = 0

    for index, entry in enumerate(entries):
        url = entry.get("url")
        if not url:
            missing_url += 1
            continue

        copied_entry = dict(entry)
        copied_entry["manifest_index"] = index
        grouped[normalize_url(url)].append(copied_entry)

    if missing_url:
        logging.warning(f"Skipped {missing_url} entries without a URL.")

    return grouped


def write_download_index(output_dir, records):
    path = output_dir / DOWNLOAD_INDEX_FILE
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
        f.write("\n")
    logging.info(f"Wrote download index: {path}")


def download_youtube_video(url, output_path):
    yt_id = get_yt_id(url)
    if yt_id is None:
        raise ValueError(f"Invalid YouTube URL: {url}")

    cmd = build_ytdlp_command(url, output_path)
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True
    )

    if result.stdout:
        logging.debug(result.stdout)
    if result.stderr:
        logging.debug(result.stderr)


def download_direct_url(url, output_path, chunk_size=1024 * 1024):
    request = Request(url, headers={"User-Agent": USER_AGENT})

    with urlopen(request, timeout=60) as response:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)


def download_with_retries(url, output_path, downloader, max_attempts=3):
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            downloader(url, output_path)
            return True, None
        except (subprocess.CalledProcessError, HTTPError, URLError, TimeoutError, OSError, ValueError) as e:
            last_error = e
            logging.error(f"Attempt {attempt}/{max_attempts} failed for {url}: {e}")
            if isinstance(e, subprocess.CalledProcessError):
                if e.stderr:
                    logging.error(e.stderr)
                if e.stdout:
                    logging.debug(e.stdout)

            if attempt < max_attempts:
                time.sleep(random.uniform(3.0, 7.0))

    return False, str(last_error)


def build_index_record(url, entries, output_path, status, error=None):
    return {
        "url": url,
        "url_type": "youtube" if is_youtube_url(url) else "direct",
        "output_path": str(output_path),
        "status": status,
        "error": error,
        "entry_count": len(entries),
        "entries": [
            {
                "manifest_index": entry.get("manifest_index"),
                "dataset": entry.get("dataset"),
                "target_sign": entry.get("target_sign"),
                "matched_sign": entry.get("matched_sign"),
                "split": entry.get("split"),
                "video_id": entry.get("video_id"),
                "start_time": entry.get("start_time"),
                "end_time": entry.get("end_time"),
                "start": entry.get("start"),
                "end": entry.get("end"),
                "frame_start": entry.get("frame_start"),
                "frame_end": entry.get("frame_end"),
            }
            for entry in entries
        ],
    }


def download_videos(json_file=JSON_FILE, output_dir=OUTPUT_DIR, limit=None):
    entries = load_entries(json_file)
    grouped = group_entries_by_url(entries)
    output_dir.mkdir(parents=True, exist_ok=True)

    urls = sorted(grouped)
    if limit is not None:
        urls = urls[:limit]

    youtube_count = sum(1 for url in urls if is_youtube_url(url))
    direct_count = len(urls) - youtube_count
    logging.info(f"Loaded {len(entries)} entries from {json_file}.")
    logging.info(f"Found {len(grouped)} unique URLs: {youtube_count} YouTube, {direct_count} direct.")

    index_records = []

    for number, url in enumerate(urls, start=1):
        entries_for_url = grouped[url]
        output_path = build_output_path(url, entries_for_url[0], output_dir)

        if output_path.exists() and output_path.stat().st_size > 0:
            logging.info(f"[{number}/{len(urls)}] Already downloaded: {output_path}")
            index_records.append(build_index_record(url, entries_for_url, output_path, "skipped_existing"))
            continue

        logging.info(f"[{number}/{len(urls)}] Downloading: {url}")
        downloader = download_youtube_video if is_youtube_url(url) else download_direct_url
        success, error = download_with_retries(url, output_path, downloader)

        if success:
            logging.info(f"Saved: {output_path}")
            status = "downloaded"
        else:
            logging.error(f"Failed permanently: {url}")
            status = "failed"

        index_records.append(build_index_record(url, entries_for_url, output_path, status, error))
        time.sleep(random.uniform(1.0, 4.0))

    write_download_index(output_dir, index_records)
    return index_records


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download raw videos from collected MS-ASL/WLASL metadata."
    )
    parser.add_argument(
        "--json-file",
        type=Path,
        default=JSON_FILE,
        help="Collected metadata JSON file to download from.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where raw videos should be saved.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of unique URLs to download for a test run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.info("Starting video downloads.")

    check_python_version()
    check_command([YTDLP, "--version"], "yt-dlp version")
    check_command(["ffmpeg", "-version"], "ffmpeg version")

    records = download_videos(args.json_file, args.output_dir, args.limit)
    counts = Counter(record["status"] for record in records)
    logging.info(f"Download process completed: {dict(counts)}")
