import json
import os
import sys
from typing import List, Dict, Any
from datetime import datetime

DEFAULT_PATHS = [
    os.path.join("src", "graph", "data", "micro_dataset.json"),
    os.path.join("src", "graph", "micro_dataset.json"),
]


def prompt(msg: str) -> str:
    try:
        return input(msg)
    except EOFError:
        return ""


def prompt_list(msg: str) -> List[str]:
    raw = prompt(msg).strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def confirm(msg: str, default_yes: bool = True) -> bool:
    suffix = " [Y/n]: " if default_yes else " [y/N]: "
    ans = prompt(msg + suffix).strip().lower()
    if ans == "":
        return default_yes
    return ans in {"y", "yes"}


def choose_path(arg_path: str | None) -> str:
    if arg_path and os.path.isfile(arg_path):
        return arg_path
    for p in DEFAULT_PATHS:
        if os.path.isfile(p):
            return p
    while True:
        p = prompt("Enter dataset path: ").strip()
        if p and os.path.isfile(p):
            return p
        print("Path not found. Try again.")


def print_item(item: Dict[str, Any], idx: int) -> None:
    print("-" * 80)
    print(f"Item #{idx+1}")
    print(f"Sentence: {item.get('sentence','')}")
    print(f"Entities: {', '.join(item.get('entities', []))}")
    rels = item.get("ground_truth", [])
    if not rels:
        print("Relations: []")
    else:
        print("Relations:")
        for i, r in enumerate(rels, 1):
            s = r.get("subject", {}).get("head", "")
            sm = r.get("subject", {}).get("modifiers", [])
            t = r.get("relation", {}).get("type", "")
            tx = r.get("relation", {}).get("text", "")
            o = r.get("object", {}).get("head", "")
            om = r.get("object", {}).get("modifiers", [])
            print(f"  {i}. {s} [{', '.join(sm)}] --{t}:{tx}--> {o} [{', '.join(om)}]")


def edit_sentence(item: Dict[str, Any]) -> None:
    cur = item.get("sentence", "")
    new = prompt(f"Sentence [{cur}]: ").strip()
    if new:
        item["sentence"] = new


def edit_entities(item: Dict[str, Any]) -> None:
    cur = ", ".join(item.get("entities", []))
    new_list = prompt_list(f"Entities (comma-separated) [{cur}]: ")
    if new_list:
        item["entities"] = new_list


def add_relation(entities: List[str]) -> Dict[str, Any] | None:
    rel_type = prompt("  Relation type [action|association|belonging]: ").strip().lower()
    if not rel_type:
        return None
    rel_text = prompt("  Relation text/phrase: ").strip()
    subj = prompt("  Subject entity name: ").strip()
    obj = prompt("  Object entity name: ").strip()
    if not subj or not obj:
        return None
    return {
        "subject": {"head": subj, "modifiers": []},
        "relation": {"type": rel_type, "text": rel_text},
        "object": {"head": obj, "modifiers": []},
    }


def edit_relation(rel: Dict[str, Any], entities: List[str]) -> None:
    s = rel.get("subject", {}).get("head", "")
    o = rel.get("object", {}).get("head", "")
    t = rel.get("relation", {}).get("type", "")
    tx = rel.get("relation", {}).get("text", "")
    ns = input(f"    Subject entity name [{s}]: ").strip() or s
    no = input(f"    Object entity name [{o}]: ").strip() or o
    nt = input(f"    Relation type [{t}]: ").strip().lower() or t
    ntx = input(f"    Relation text/phrase [{tx}]: ").strip() or tx
    rel["subject"]["head"] = ns
    rel["object"]["head"] = no
    rel["relation"]["type"] = nt
    rel["relation"]["text"] = ntx


def apply_entity_modifiers(relations: List[Dict[str, Any]], entities: List[str]) -> None:
    mapping: Dict[str, List[str]] = {}
    for ent in entities:
        mods = prompt_list(f"  Modifiers for '{ent}' (comma-separated, optional): ")
        mapping[ent] = mods
    for r in relations:
        s = r.get("subject", {}).get("head", "")
        o = r.get("object", {}).get("head", "")
        r["subject"]["modifiers"] = mapping.get(s, [])
        r["object"]["modifiers"] = mapping.get(o, [])


def edit_relations(item: Dict[str, Any]) -> None:
    entities = item.get("entities", [])
    relations: List[Dict[str, Any]] = item.get("ground_truth", [])
    while True:
        print_item(item, -1)
        print("Relation edit menu: [a]dd, [e]dit, [d]elete, [m]odifiers, [q]uit")
        choice = prompt("Choose: ").strip().lower()
        if choice == "a":
            r = add_relation(entities)
            if r:
                relations.append(r)
        elif choice == "e":
            idx = prompt("  Index to edit: ").strip()
            if idx.isdigit():
                k = int(idx) - 1
                if 0 <= k < len(relations):
                    edit_relation(relations[k], entities)
        elif choice == "d":
            idx = prompt("  Index to delete: ").strip()
            if idx.isdigit():
                k = int(idx) - 1
                if 0 <= k < len(relations):
                    relations.pop(k)
        elif choice == "m":
            apply_entity_modifiers(relations, entities)
        elif choice == "q":
            break
        item["ground_truth"] = relations


def edit_item(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    while True:
        print_item(item, idx)
        if confirm("Approve this item?", default_yes=True):
            return item
        print("Edit menu: 1) Sentence  2) Entities  3) Relations  4) Apply entity modifiers  5) Done")
        choice = prompt("Choose: ").strip()
        if choice == "1":
            edit_sentence(item)
        elif choice == "2":
            edit_entities(item)
        elif choice == "3":
            edit_relations(item)
        elif choice == "4":
            apply_entity_modifiers(item.get("ground_truth", []), item.get("entities", []))
        elif choice == "5":
            if confirm("Done editing this item?", default_yes=True):
                return item


def backup_file(path: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{os.path.splitext(path)[0]}.backup_{ts}.json"
    try:
        with open(path, "r") as rf, open(backup, "w") as wf:
            wf.write(rf.read())
        return backup
    except Exception:
        return ""


def main():
    arg_path = sys.argv[1] if len(sys.argv) > 1 else None
    path = choose_path(arg_path)
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("Dataset is not a list. Aborting.")
        return
    print(f"Loaded {len(data)} items from {path}")
    b = backup_file(path)
    if b:
        print(f"Backup saved to {b}")
    updated: List[Dict[str, Any]] = []
    for i, item in enumerate(data):
        updated.append(edit_item(item, i))
    print("Preview of changes:")
    print(json.dumps(updated, indent=2))
    if confirm("Save changes?", default_yes=True):
        with open(path, "w") as f:
            json.dump(updated, f, indent=2)
        print(f"Saved to {path}")
    else:
        print("Discarded changes.")


if __name__ == "__main__":
    main()
