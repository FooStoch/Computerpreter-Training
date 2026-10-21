import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


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
TARGET_SIGNS = tuple(TARGET_SIGN_SYNONYMS)

PROJECT_DIR = Path(__file__).resolve().parent
COMPUTERPRETER_DIR = PROJECT_DIR.parent

MSASL_DIR = COMPUTERPRETER_DIR / "MS-ASL"
WLASL_JSON = COMPUTERPRETER_DIR / "WLASL-master" / "start_kit" / "WLASL_v0.3.json"
OUTPUT_DIR = PROJECT_DIR / "collected_data"

MSASL_SPLIT_FILES = {
    "train": "MSASL_train.json",
    "val": "MSASL_val.json",
    "test": "MSASL_test.json",
}
MSASL_SYNONYM_FILE = "MSASL_synonym.json"


def normalize_sign(value):
    """Normalize labels/glosses so matching is exact but case-insensitive."""
    return str(value).strip().lower().replace("’", "'")


def alias_variants(value):
    normalized = normalize_sign(value)
    return {
        normalized,
        normalized.replace("'", ""),
        normalized.replace(" ", ""),
        normalized.replace("'", "").replace(" ", ""),
    }


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)
        f.write("\n")


def build_aliases(msasl_dir=MSASL_DIR, target_signs=TARGET_SIGNS):
    """Map every searched label/synonym back to one of the canonical targets."""
    aliases_by_target = {
        normalize_sign(target): set()
        for target in target_signs
    }
    for target in target_signs:
        normalized_target = normalize_sign(target)
        for synonym in TARGET_SIGN_SYNONYMS[normalized_target]:
            aliases_by_target[normalized_target].update(alias_variants(synonym))

    target_set = set(aliases_by_target)
    synonym_path = msasl_dir / MSASL_SYNONYM_FILE

    for group in load_json(synonym_path):
        normalized_group = set()
        for item in group:
            normalized_group.update(alias_variants(item))
        matched_targets = target_set & normalized_group
        matched_targets.update(
            target
            for target, aliases in aliases_by_target.items()
            if aliases & normalized_group
        )
        for target in matched_targets:
            aliases_by_target[target].update(normalized_group)

    target_by_alias = {}
    for target, aliases in aliases_by_target.items():
        for alias in aliases:
            existing_target = target_by_alias.get(alias)
            if existing_target and existing_target != target:
                raise ValueError(
                    f"Alias {alias!r} maps to both {existing_target!r} and {target!r}."
                )
            target_by_alias[alias] = target

    return {
        "aliases_by_target": {
            target: sorted(aliases)
            for target, aliases in aliases_by_target.items()
        },
        "target_by_alias": target_by_alias,
        "source_json": str(synonym_path),
    }


def tag_record(record, dataset, target_sign, matched_sign, source_json):
    record["dataset"] = dataset
    record["target_sign"] = target_sign
    record["matched_sign"] = matched_sign
    record["matched_via"] = "target" if matched_sign == target_sign else "synonym"
    record["source_json"] = str(source_json)
    return record


def collect_msasl_entries(msasl_dir=MSASL_DIR, aliases=None):
    aliases = aliases or build_aliases(msasl_dir)
    target_by_alias = aliases["target_by_alias"]
    collected = []

    for split, filename in MSASL_SPLIT_FILES.items():
        path = msasl_dir / filename
        entries = load_json(path)

        for entry in entries:
            clean_text = normalize_sign(entry.get("clean_text", ""))
            target_sign = target_by_alias.get(clean_text)
            if not target_sign:
                continue

            record = tag_record(
                dict(entry),
                dataset="MS-ASL",
                target_sign=target_sign,
                matched_sign=clean_text,
                source_json=path,
            )
            record["split"] = split
            collected.append(record)

    return collected


def collect_wlasl_entries(wlasl_json=WLASL_JSON, aliases=None):
    aliases = aliases or build_aliases()
    target_by_alias = aliases["target_by_alias"]
    collected = []
    gloss_entries = load_json(wlasl_json)

    for gloss_entry in gloss_entries:
        gloss = normalize_sign(gloss_entry.get("gloss", ""))
        target_sign = target_by_alias.get(gloss)
        if not target_sign:
            continue

        for instance in gloss_entry.get("instances", []):
            record = tag_record(
                dict(instance),
                dataset="WLASL",
                target_sign=target_sign,
                matched_sign=gloss,
                source_json=wlasl_json,
            )
            record["gloss"] = gloss
            record["clean_text"] = gloss

            fps = record.get("fps")
            frame_start = record.get("frame_start")
            frame_end = record.get("frame_end")
            if fps and frame_start not in (None, -1):
                record["start_time"] = round(frame_start / fps, 3)
            if fps and frame_end not in (None, -1):
                record["end_time"] = round(frame_end / fps, 3)

            collected.append(record)

    return collected


def nested_counts(entries, *keys):
    if not keys:
        return len(entries)

    key = keys[0]
    grouped = defaultdict(list)
    for entry in entries:
        grouped[entry.get(key, "UNKNOWN")].append(entry)

    return {
        group: nested_counts(group_entries, *keys[1:])
        for group, group_entries in sorted(grouped.items())
    }


def collect_all(output_dir=OUTPUT_DIR):
    aliases = build_aliases()
    msasl_entries = collect_msasl_entries(aliases=aliases)
    wlasl_entries = collect_wlasl_entries(aliases=aliases)
    combined_entries = msasl_entries + wlasl_entries

    outputs = {
        "msasl": output_dir / "msasl_target_signs_with_synonyms.json",
        "wlasl": output_dir / "wlasl_target_signs_with_synonyms.json",
        "combined": output_dir / "all_target_sign_entries_with_synonyms.json",
        "summary": output_dir / "collection_summary.json",
    }

    save_json(outputs["msasl"], msasl_entries)
    save_json(outputs["wlasl"], wlasl_entries)
    save_json(outputs["combined"], combined_entries)

    summary = {
        "target_signs": list(TARGET_SIGNS),
        "synonym_source": aliases["source_json"],
        "searched_terms_by_target": aliases["aliases_by_target"],
        "input_files": {
            "MS-ASL": [str(MSASL_DIR / filename) for filename in MSASL_SPLIT_FILES.values()],
            "MS-ASL synonyms": aliases["source_json"],
            "WLASL": str(WLASL_JSON),
        },
        "total_entries": len(combined_entries),
        "total_by_source": dict(Counter(entry["dataset"] for entry in combined_entries)),
        "counts_by_source_target_and_match": nested_counts(
            combined_entries,
            "dataset",
            "target_sign",
            "matched_sign",
        ),
        "counts_by_target_and_match": nested_counts(
            combined_entries,
            "target_sign",
            "matched_sign",
        ),
        "outputs": {name: str(path) for name, path in outputs.items()},
    }

    save_json(outputs["summary"], summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect target sign metadata entries from MS-ASL and WLASL, "
            "including target synonyms listed by MS-ASL. This does not download videos."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where collected JSON files should be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = collect_all(args.output_dir)

    print("No videos were downloaded.")
    print("Target signs:", ", ".join(summary["target_signs"]))
    print("Searched terms by target:")
    for target, terms in summary["searched_terms_by_target"].items():
        print(f"  {target}: {', '.join(terms)}")
    print("Input files:")
    for name, paths in summary["input_files"].items():
        if isinstance(paths, list):
            print(f"  {name}:")
            for path in paths:
                print(f"    {path}")
        else:
            print(f"  {name}: {paths}")
    print("Total entries:", summary["total_entries"])
    print("Total by source:", summary["total_by_source"])
    print("Counts by source/target/matched term:")
    for dataset, target_counts in summary["counts_by_source_target_and_match"].items():
        print(f"  {dataset}: {target_counts}")
    print("Wrote files:")
    for name, path in summary["outputs"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
