import json
import os
from typing import List, Dict, Any

DEFAULT_PATH = os.path.join("src", "graph", "micro_dataset.json")


def prompt_list(prompt: str) -> List[str]:
    raw = input(prompt).strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def prompt_relation() -> Dict[str, Any]:
    rel_type = input("  Relation type [action|association|belonging]: ").strip().lower()
    if not rel_type:
        return {"subject": {"head": "", "modifiers": []}, "relation": {"type": "", "text": ""}, "object": {"head": "", "modifiers": []}}
    rel_text = input("  Relation text/phrase: ").strip()
    subj = input("  Subject entity name: ").strip()
    obj = input("  Object entity name: ").strip()
    return {
        "subject": {"head": subj, "modifiers": []},
        "relation": {"type": rel_type, "text": rel_text},
        "object": {"head": obj, "modifiers": []},
    }


def main(output_path: str = DEFAULT_PATH):
    print("=== Micro-Dataset Creator ===")
    print("Enter sentences and annotate entities and relations. Leave sentence empty to finish.\n")
    dataset: List[Dict[str, Any]] = []

    preload: List[str] = []
    data_dir = os.path.join("src", "graph", "data")
    candidates = [
        os.path.join(data_dir, "sentences.txt"),
        os.path.join(data_dir, "samples.txt"),
        os.path.join(data_dir, "dataset.txt"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path, "r") as f:
                    lines = [ln.strip() for ln in f.readlines()]
                preload = [ln for ln in lines if ln]
                if preload:
                    print(f"Preloaded {len(preload)} sentences from {path}")
                    break
            except Exception:
                pass

    idx = 0
    while True:
        if idx < len(preload):
            sentence = preload[idx]
            print(f"Sentence [{idx+1}/{len(preload)}]: {sentence}")
            use = input("Use this sentence? [Y/n]: ").strip().lower()
            if use == "n":
                sentence = input("Sentence: ").strip()
            idx += 1
        else:
            sentence = input("Sentence: ").strip()
        if not sentence:
            break
        entities = prompt_list("Entities in sentence (comma-separated): ")
        ground_truth: List[Dict[str, Any]] = []
        print("Add ground-truth relations for this sentence. Leave relation type empty to stop.")
        while True:
            r = prompt_relation()
            if not r["relation"]["type"] or not r["subject"]["head"] or not r["object"]["head"]:
                break
            ground_truth.append(r)
            more = input("  Add another relation? [y/N]: ").strip().lower()
            if more != "y":
                break
        if entities:
            print("\nAdd modifiers for entities in this sentence (optional). Leave blank to skip.")
            entity_mods = {}
            for ent in entities:
                mods = prompt_list(f"  Modifiers for '{ent}' (comma-separated, optional): ")
                entity_mods[ent] = mods
            for rel in ground_truth:
                s = rel["subject"]["head"]
                o = rel["object"]["head"]
                rel["subject"]["modifiers"] = entity_mods.get(s, [])
                rel["object"]["modifiers"] = entity_mods.get(o, [])
        dataset.append({
            "sentence": sentence,
            "entities": entities,
            "ground_truth": ground_truth,
        })
        print()

    if not dataset:
        print("No data entered. Exiting without saving.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved {len(dataset)} items to {output_path}")


if __name__ == "__main__":
    main()
