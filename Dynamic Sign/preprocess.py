import argparse
import csv
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


logging.basicConfig(
    filename=f"preprocess_{int(time.time())}.log",
    filemode="w",
    level=logging.INFO,
)
logging.getLogger().addHandler(logging.StreamHandler())

PROJECT_DIR = Path(__file__).resolve().parent
COMPUTERPRETER_DIR = PROJECT_DIR.parent

MANIFEST_FILE = PROJECT_DIR / "collected_data" / "all_target_sign_entries_with_synonyms.json"
RAW_VIDEOS_DIR = PROJECT_DIR / "raw_videos"
DOWNLOAD_INDEX_FILE = RAW_VIDEOS_DIR / "download_index.json"
OUTPUT_DIR = PROJECT_DIR / "videos"
ASL_CITIZEN_DIR = COMPUTERPRETER_DIR / "ASL_Citizen"

VIDEO_EXTENSIONS = (".mp4", ".webm", ".mkv", ".mov", ".swf")
CROP_PADDING = 0.20
MIN_RAW_VIDEO_BYTES = 1024

ASL_CITIZEN_SPLITS = ("train", "val", "test")
TARGET_SIGN_SYNONYMS = {
    "me": ("me",),
    "you": ("you",),
    "he": ("he", "she", "it"),
    "we": ("we",),
    "they": ("they", "them"),
    "mother": ("mother", "mom", "mommy"),
    "father": ("father", "dad", "daddy"),
    "friend": ("friend", "friends"),
    "family": ("family",),
    "yes": ("yes",),
    "no": ("no",),
    "not": ("not", "don't"),
    "please": ("please",),
    "thank you": ("thank you", "thanks"),
    "sorry": ("sorry", "regret"),
    "hello": ("hello",),
    "bye": ("bye", "goodbye"),
    "again": ("again", "repeat"),
    "what": ("what",),
    "why": ("why",),
    "who": ("who",),
    "which": ("which",),
    "name": ("name",),
    "eat": ("eat", "food"),
    "drink": ("drink",),
    "help": ("help", "aid"),
    "need": ("need", "must", "should"),
    "like": ("like",),
    "don't like": ("don't like", "dislike"),
    "go": ("go",),
    "play": ("play", "playing"),
    "want": ("want",),
    "don't want": ("don't want",),
    "think": ("think",),
    "know": ("know",),
    "don't know": ("don't know",),
    "have": ("have",),
    "school": ("school",),
    "home": ("home",),
    "bathroom": ("bathroom", "toilet"),
    "water": ("water",),
    "book": ("book",),
    "work": ("work",),
    "sit": ("sit", "chair"),
    "stop": ("stop",),
    "good": ("good",),
    "bad": ("bad",),
    "happy": ("happy", "joyful"),
    "sad": ("sad",),
    "tired": ("tired",),
    "understand": ("understand", "comprehend"),
    "don't understand": ("don't understand", "not understand", "misunderstand"),
    "more": ("more",),
    "finish": ("finish", "finished", "done"),
    "same": ("same", "same same", "similar"),
    "now": ("now",),
}


def normalize_gloss(value):
    return str(value).strip().lower().replace("’", "'")


def gloss_variants(value):
    normalized = normalize_gloss(value)
    return {
        normalized,
        normalized.replace("'", ""),
        normalized.replace(" ", ""),
        normalized.replace("'", "").replace(" ", ""),
    }


def build_asl_citizen_targets():
    targets = {}
    for target, synonyms in TARGET_SIGN_SYNONYMS.items():
        for synonym in synonyms:
            for variant in gloss_variants(synonym):
                targets[variant.upper()] = target
    return targets


ASL_CITIZEN_TARGETS = build_asl_citizen_targets()


def normalize_url(url):
    url = str(url).strip()
    if url.startswith(("www.youtube.com", "youtube.com", "m.youtube.com", "youtu.be")):
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


def safe_token(value, default="unknown"):
    token = str(value or default).strip().lower()
    token = re.sub(r"[^a-z0-9]+", "_", token)
    return token.strip("_") or default


def unique_download_id(url, entry):
    if is_youtube_url(url):
        yt_id = get_yt_id(url)
        if yt_id:
            return yt_id

    video_id = entry.get("video_id")
    if video_id not in (None, ""):
        return f"{str(entry.get('dataset', 'video')).lower()}_{video_id}"

    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"url_{digest}"


def expected_raw_path(url, entry, raw_videos_dir):
    target = safe_token(entry.get("target_sign"))
    matched = safe_token(entry.get("matched_sign"), target)
    dataset = safe_token(str(entry.get("dataset", "dataset")).replace("-", ""))
    download_id = unique_download_id(url, entry)
    extension = ".mp4" if is_youtube_url(url) else file_extension_from_url(url)
    return raw_videos_dir / f"{dataset}_{target}_{matched}_{download_id}{extension}"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)
        f.write("\n")


def load_download_index(path=DOWNLOAD_INDEX_FILE):
    if not path.exists():
        logging.info(f"Download index not found: {path}")
        return {}

    records = load_json(path)
    lookup = {}
    for record in records:
        url = normalize_url(record.get("url", ""))
        output_path = record.get("output_path")
        if url and output_path:
            lookup[url] = Path(output_path)

    logging.info(f"Loaded {len(lookup)} URL mappings from {path}")
    return lookup


def is_usable_raw_video(path):
    return path.exists() and path.is_file() and path.stat().st_size >= MIN_RAW_VIDEO_BYTES


def resolve_raw_video(entry, download_lookup, raw_videos_dir):
    url = normalize_url(entry.get("url", ""))
    indexed_path = download_lookup.get(url)
    if indexed_path and is_usable_raw_video(indexed_path):
        return indexed_path

    expected_path = expected_raw_path(url, entry, raw_videos_dir)
    if is_usable_raw_video(expected_path):
        return expected_path

    download_id = unique_download_id(url, entry)
    expected_extension = ".mp4" if is_youtube_url(url) else file_extension_from_url(url)
    for candidate in raw_videos_dir.iterdir():
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() != expected_extension:
            continue
        if candidate.stem.endswith(f"_{download_id}") and is_usable_raw_video(candidate):
            return candidate

    return None


def even_floor_expr(expr):
    return f"trunc(({expr})/2)*2"


def ffmpeg_func(name, *args):
    return f"{name}(" + "\\,".join(str(arg) for arg in args) + ")"


def build_msasl_crop_filter(entry, padding=CROP_PADDING):
    box = entry.get("box")
    if not box or len(box) != 4:
        return None

    x_min, y_min, x_max, y_max = [float(value) for value in box]
    box_width = x_max - x_min
    box_height = y_max - y_min

    if box_width <= 0 or box_height <= 0:
        return None

    x_min = max(0.0, x_min - (box_width * padding))
    y_min = max(0.0, y_min - (box_height * padding))
    x_max = min(1.0, x_max + (box_width * padding))
    y_max = min(1.0, y_max + (box_height * padding))

    crop_width = x_max - x_min
    crop_height = y_max - y_min
    if crop_width <= 0 or crop_height <= 0:
        return None

    width_expr = ffmpeg_func("max", 2, even_floor_expr(f"iw*{crop_width:.8f}"))
    height_expr = ffmpeg_func("max", 2, even_floor_expr(f"ih*{crop_height:.8f}"))
    x_expr = ffmpeg_func(
        "min",
        ffmpeg_func("max", 0, f"trunc(iw*{x_min:.8f})"),
        f"iw-{width_expr}",
    )
    y_expr = ffmpeg_func(
        "min",
        ffmpeg_func("max", 0, f"trunc(ih*{y_min:.8f})"),
        f"ih-{height_expr}",
    )
    return f"crop={width_expr}:{height_expr}:{x_expr}:{y_expr}"


def build_wlasl_crop_filter(entry, padding=CROP_PADDING):
    bbox = entry.get("bbox")
    width = entry.get("width")
    height = entry.get("height")

    if not bbox or len(bbox) != 4:
        return None

    x_min, y_min, x_max, y_max = [float(value) for value in bbox]
    box_width = x_max - x_min
    box_height = y_max - y_min
    if box_width <= 0 or box_height <= 0:
        return None

    x_min -= box_width * padding
    y_min -= box_height * padding
    x_max += box_width * padding
    y_max += box_height * padding

    if width and height:
        width = float(width)
        height = float(height)
        x_min = max(0.0, x_min)
        y_min = max(0.0, y_min)
        x_max = min(width, x_max)
        y_max = min(height, y_max)
        crop_width = x_max - x_min
        crop_height = y_max - y_min
        if crop_width <= 0 or crop_height <= 0:
            return None

        width_expr = ffmpeg_func("min", ffmpeg_func("max", 2, even_floor_expr(str(crop_width))), "iw")
        height_expr = ffmpeg_func("min", ffmpeg_func("max", 2, even_floor_expr(str(crop_height))), "ih")
        x_expr = ffmpeg_func("min", ffmpeg_func("max", 0, int(x_min)), f"iw-{width_expr}")
        y_expr = ffmpeg_func("min", ffmpeg_func("max", 0, int(y_min)), f"ih-{height_expr}")
        return f"crop={width_expr}:{height_expr}:{x_expr}:{y_expr}"

    width_expr = ffmpeg_func("min", ffmpeg_func("max", 2, even_floor_expr(str(x_max - x_min))), "iw")
    height_expr = ffmpeg_func("min", ffmpeg_func("max", 2, even_floor_expr(str(y_max - y_min))), "ih")
    x_expr = ffmpeg_func("min", ffmpeg_func("max", 0, max(0, int(x_min))), f"iw-{width_expr}")
    y_expr = ffmpeg_func("min", ffmpeg_func("max", 0, max(0, int(y_min))), f"ih-{height_expr}")
    return f"crop={width_expr}:{height_expr}:{x_expr}:{y_expr}"


def build_crop_filter(entry):
    dataset = str(entry.get("dataset", "")).upper()
    if dataset == "MS-ASL":
        return build_msasl_crop_filter(entry)
    if dataset == "WLASL":
        return build_wlasl_crop_filter(entry)
    return None


def output_manifest_clip_name(entry, manifest_index):
    dataset = safe_token(str(entry.get("dataset", "")).replace("-", ""))
    target = safe_token(entry.get("target_sign"))
    matched = safe_token(entry.get("matched_sign"), target)

    if dataset == "wlasl":
        video_id = safe_token(entry.get("video_id"), f"idx_{manifest_index}")
        instance_id = safe_token(entry.get("instance_id"), "instance")
        return f"wlasl_{target}_{matched}_{video_id}_{instance_id}.mp4"

    return f"{dataset}_{target}_{matched}_{manifest_index}.mp4"


def ffmpeg_cut_and_crop(input_video, output_path, entry):
    crop_filter = build_crop_filter(entry)
    if not crop_filter:
        return "invalid_crop", "Invalid or missing crop metadata."

    start_time = entry.get("start_time")
    end_time = entry.get("end_time")
    if start_time is None:
        return "invalid_time", "Missing start_time."

    cmd = [
        "ffmpeg",
        "-ss",
        str(start_time),
    ]
    if end_time is not None:
        cmd.extend(["-to", str(end_time)])

    cmd.extend([
        "-i",
        str(input_video),
        "-vf",
        crop_filter,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        "-y",
        str(output_path),
    ])

    try:
        logging.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return "created", None
    except FileNotFoundError:
        raise
    except subprocess.CalledProcessError as e:
        return "ffmpeg_failure", e.stderr or str(e)


def process_manifest_videos(manifest_file, raw_videos_dir, output_dir, limit=None):
    entries = load_json(manifest_file)
    download_lookup = load_download_index(raw_videos_dir / "download_index.json")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    counts = Counter()
    selected_entries = list(enumerate(entries))
    if limit is not None:
        selected_entries = selected_entries[:limit]

    logging.info(f"Processing {len(selected_entries)} MS-ASL/WLASL manifest entries.")

    for manifest_index, entry in selected_entries:
        dataset = entry.get("dataset")
        if dataset not in {"MS-ASL", "WLASL"}:
            continue

        raw_video = resolve_raw_video(entry, download_lookup, raw_videos_dir)
        output_path = output_dir / output_manifest_clip_name(entry, manifest_index)

        record = {
            "dataset": dataset,
            "target_sign": entry.get("target_sign"),
            "matched_sign": entry.get("matched_sign"),
            "split": entry.get("split"),
            "source_path": str(raw_video) if raw_video else None,
            "output_path": str(output_path),
            "manifest_index": manifest_index,
            "url": entry.get("url"),
            "video_id": entry.get("video_id"),
            "instance_id": entry.get("instance_id"),
            "start_time": entry.get("start_time"),
            "end_time": entry.get("end_time"),
            "start": entry.get("start"),
            "end": entry.get("end"),
            "frame_start": entry.get("frame_start"),
            "frame_end": entry.get("frame_end"),
        }

        if output_path.exists() and output_path.stat().st_size > 0:
            logging.info(f"Processed clip already exists: {output_path}")
            record["status"] = "skipped_existing"
            counts["skipped_existing"] += 1
            records.append(record)
            continue

        if raw_video is None:
            logging.warning(f"Missing raw video for manifest entry {manifest_index}: {entry.get('url')}")
            record["status"] = "missing_raw"
            counts["missing_raw"] += 1
            records.append(record)
            continue

        status, error = ffmpeg_cut_and_crop(raw_video, output_path, entry)
        record["status"] = status
        if error:
            record["error"] = error

        counts[status] += 1
        records.append(record)

    return records, counts


def load_asl_citizen_rows(asl_citizen_dir):
    rows = []
    splits_dir = asl_citizen_dir / "splits"

    for split in ASL_CITIZEN_SPLITS:
        csv_path = splits_dir / f"{split}.csv"
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row_index, row in enumerate(reader):
                gloss = str(row.get("Gloss", "")).strip().upper()
                normalized_gloss = normalize_gloss(gloss).upper()
                numberless_gloss = re.sub(r"\d+$", "", normalized_gloss)
                target = (
                    ASL_CITIZEN_TARGETS.get(normalized_gloss)
                    or ASL_CITIZEN_TARGETS.get(numberless_gloss)
                )
                if not target:
                    continue
                row["split"] = split
                row["row_index"] = row_index
                row["target_sign"] = target
                row["matched_sign"] = gloss
                rows.append(row)

    return rows


def output_asl_citizen_name(row):
    target = safe_token(row["target_sign"])
    gloss = re.sub(r"[^A-Z0-9]+", "_", row["matched_sign"]).strip("_")
    original = Path(row["Video file"]).name
    return f"aslcitizen_{target}_{gloss}_{original}"


def import_asl_citizen_videos(asl_citizen_dir, output_dir, limit=None):
    rows = load_asl_citizen_rows(asl_citizen_dir)
    if limit is not None:
        rows = rows[:limit]

    source_dir = asl_citizen_dir / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    counts = Counter()

    logging.info(f"Importing {len(rows)} ASL Citizen target videos.")

    for row in rows:
        source_path = source_dir / row["Video file"]
        output_path = output_dir / output_asl_citizen_name(row)

        record = {
            "dataset": "ASL_Citizen",
            "target_sign": row["target_sign"],
            "matched_sign": row["matched_sign"],
            "split": row["split"],
            "source_path": str(source_path),
            "output_path": str(output_path),
            "participant_id": row.get("Participant ID"),
            "asl_lex_code": row.get("ASL-LEX Code"),
            "row_index": row.get("row_index"),
        }

        if output_path.exists() and output_path.stat().st_size > 0:
            logging.info(f"ASL Citizen clip already exists: {output_path}")
            record["status"] = "skipped_existing"
            counts["skipped_existing"] += 1
            records.append(record)
            continue

        if not source_path.exists() or source_path.stat().st_size == 0:
            logging.warning(f"Missing ASL Citizen video: {source_path}")
            record["status"] = "missing_asl_citizen"
            counts["missing_asl_citizen"] += 1
            records.append(record)
            continue

        shutil.copy2(source_path, output_path)
        logging.info(f"Copied ASL Citizen clip: {output_path}")
        record["status"] = "copied"
        counts["copied"] += 1
        records.append(record)

    return records, counts


def preprocess_all(
    manifest_file=MANIFEST_FILE,
    raw_videos_dir=RAW_VIDEOS_DIR,
    output_dir=OUTPUT_DIR,
    asl_citizen_dir=ASL_CITIZEN_DIR,
    limit=None,
    asl_limit=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_records, manifest_counts = process_manifest_videos(
        manifest_file,
        raw_videos_dir,
        output_dir,
        limit=limit,
    )
    asl_records, asl_counts = import_asl_citizen_videos(
        asl_citizen_dir,
        output_dir,
        limit=asl_limit,
    )

    all_records = manifest_records + asl_records
    save_json(output_dir / "preprocess_index.json", all_records)

    total_counts = Counter()
    total_counts.update(manifest_counts)
    total_counts.update(asl_counts)

    logging.info("\nSummary:")
    logging.info(f"Created cropped clips: {manifest_counts.get('created', 0)}")
    logging.info(f"Copied ASL Citizen clips: {asl_counts.get('copied', 0)}")
    logging.info(f"Skipped existing clips: {total_counts.get('skipped_existing', 0)}")
    logging.info(f"Missing raw files: {manifest_counts.get('missing_raw', 0)}")
    logging.info(f"Missing ASL Citizen files: {asl_counts.get('missing_asl_citizen', 0)}")
    logging.info(f"Invalid crop boxes: {manifest_counts.get('invalid_crop', 0)}")
    logging.info(f"Invalid timing: {manifest_counts.get('invalid_time', 0)}")
    logging.info(f"FFmpeg failures: {manifest_counts.get('ffmpeg_failure', 0)}")
    logging.info(f"Wrote preprocess index: {output_dir / 'preprocess_index.json'}")

    by_dataset = defaultdict(Counter)
    for record in all_records:
        by_dataset[record["dataset"]][record["status"]] += 1
    for dataset, counts in sorted(by_dataset.items()):
        logging.info(f"{dataset}: {dict(counts)}")

    return all_records, total_counts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare cropped MS-ASL/WLASL clips and import ASL Citizen clips."
    )
    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=MANIFEST_FILE,
        help="Collected MS-ASL/WLASL metadata JSON.",
    )
    parser.add_argument(
        "--raw-videos-dir",
        type=Path,
        default=RAW_VIDEOS_DIR,
        help="Directory containing downloaded raw videos.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Flat output directory for processed videos.",
    )
    parser.add_argument(
        "--asl-citizen-dir",
        type=Path,
        default=ASL_CITIZEN_DIR,
        help="ASL Citizen dataset directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of MS-ASL/WLASL manifest entries to process.",
    )
    parser.add_argument(
        "--asl-limit",
        type=int,
        default=None,
        help="Optional number of ASL Citizen target rows to import.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.info("Starting preprocessing.")
    logging.info(f"Manifest file: {args.manifest_file}")
    logging.info(f"Raw videos directory: {args.raw_videos_dir}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"ASL Citizen directory: {args.asl_citizen_dir}")
    logging.info(f"Crop padding: {CROP_PADDING:.0%}")

    try:
        preprocess_all(
            manifest_file=args.manifest_file,
            raw_videos_dir=args.raw_videos_dir,
            output_dir=args.output_dir,
            asl_citizen_dir=args.asl_citizen_dir,
            limit=args.limit,
            asl_limit=args.asl_limit,
        )
    except FileNotFoundError as e:
        logging.error(f"Required executable or file not found: {e}")
        raise SystemExit(1)

    logging.info("Preprocessing completed.")


if __name__ == "__main__":
    main()
