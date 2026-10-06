"""Audit train.jsonl and write a cleaned training file plus cleaning_log.csv.

The rule labeler in label_rules.py matches every dev ticket. On train, a
review of disagreements showed the opposite problem: the original category is
usually right and the rule tagger misses paraphrases. Categories from a
parseable label are therefore kept. Everything else is repaired from the
schema: priority, paise, ISO dates, transaction ids, channel, language, and
needs_human.

Rows that are not tickets (empty text, "test", "asdf") are dropped. Exact
duplicate tickets are dropped after the first copy. Unparseable labels are
rebuilt, using the category string if the truncated JSON still contains one.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

from label_rules import (
    KEYS,
    _amount_paise,
    _category,
    _channel,
    _language,
    _priority,
    _repeat_contact,
    _txn_date,
    _txn_id,
    canonical_json,
    is_garbage,
    label_ticket,
    parse_ticket,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ALLOWED = {
    "payment_failed",
    "refund",
    "fraud",
    "account_access",
    "kyc",
    "offers",
    "other",
}
NULL_CATS = {"account_access", "kyc", "other"}
CHANNELS = {"upi", "card", "netbanking", "wallet"}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_id(value):
    if not isinstance(value, str):
        return None
    m = re.search(r"(?i)\bnp[\s\-]*((?:\d[\s\-]*){10})", value)
    if not m:
        return None
    digits = re.sub(r"\D", "", m.group(1))
    return "NP" + digits if len(digits) == 10 else None


def normalize_date(value):
    if not isinstance(value, str):
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return value
    m = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", value)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    return None


def category_from_raw(raw: str | None) -> str | None:
    if not raw:
        return None
    m = re.search(r'"category"\s*:\s*"([a-z_]+)"', raw)
    if m and m.group(1) in ALLOWED:
        return m.group(1)
    return None


def repair(text: str, gold: dict | None, raw: str | None) -> dict | None:
    received, customer, body = parse_ticket(text)
    if is_garbage(body):
        return None
    if gold and gold.get("category") in ALLOWED:
        category = gold["category"]
        category_source = "original"
    else:
        salvaged = category_from_raw(raw)
        if salvaged:
            category = salvaged
            category_source = "salvaged"
        else:
            category = _category(body)
            category_source = "rules"
    if category in NULL_CATS:
        amount = txn_date = txn_id = channel = None
    else:
        amount = _amount_paise(customer)
        if amount is None and gold and isinstance(gold.get("amount_paise"), int):
            amount = gold["amount_paise"]
        txn_date = _txn_date(customer, received)
        if txn_date is None and gold:
            txn_date = normalize_date(gold.get("txn_date"))
        txn_id = _txn_id(customer)
        if txn_id is None and gold:
            txn_id = normalize_id(gold.get("txn_id")) if isinstance(gold.get("txn_id"), str) else None
        channel = _channel(customer)
        if channel is None and gold and gold.get("channel") in CHANNELS:
            channel = gold["channel"]
    label = {
        "category": category,
        "priority": _priority(category, amount, customer),
        "amount_paise": amount,
        "txn_date": txn_date,
        "txn_id": txn_id,
        "channel": channel,
        "language": _language(customer),
        "needs_human": False,
    }
    label["needs_human"] = label["priority"] == "P1" or _repeat_contact(customer)
    label["_category_source"] = category_source
    return label


def change_list(gold: dict | None, label: dict) -> list[str]:
    changes = []
    for key in KEYS:
        old = None if gold is None else gold.get(key, None)
        new = label[key]
        if gold is None or key not in gold or old != new:
            if gold is not None and key not in gold:
                changes.append(f"missing {key} -> {new!r}")
            else:
                changes.append(f"{key}: {old!r} -> {new!r}")
    return changes


def clean(train_path: Path, out_path: Path, log_path: Path, summary_path: Path) -> dict:
    rows = load_jsonl(train_path)
    seen: dict[str, str] = {}
    log = []
    cleaned = []
    counts = Counter()
    field_fixes = Counter()
    for row in rows:
        text = row["messages"][1]["content"]
        raw = row["messages"][-1]["content"]
        _, _, body = parse_ticket(text)
        if is_garbage(body):
            counts["drop_garbage"] += 1
            log.append((row["id"], "drop", "placeholder or empty customer message"))
            continue
        if text in seen:
            counts["drop_duplicate"] += 1
            log.append((row["id"], "drop", f"exact duplicate of {seen[text]}"))
            continue
        seen[text] = row["id"]
        try:
            gold = json.loads(raw)
            if not isinstance(gold, dict):
                gold = None
                parsed = False
            else:
                parsed = True
        except json.JSONDecodeError:
            gold = None
            parsed = False
        label = repair(text, gold, raw)
        if label is None:
            counts["drop_garbage"] += 1
            log.append((row["id"], "drop", "placeholder or empty customer message"))
            continue
        source = label.pop("_category_source")
        changes = change_list(gold if parsed else None, label)
        # A legacy "prio" key is a change even when the note above already lists priority.
        if parsed and "priority" not in gold:
            counts["legacy_prio"] += 1
        if not parsed:
            counts["rebuilt_json"] += 1
            reason = "unparseable assistant JSON; label rebuilt from the ticket and schema"
            if changes:
                reason += " (" + "; ".join(changes) + ")"
            log.append((row["id"], "fix", reason))
            field_fixes["rebuilt_json"] += 1
        elif changes:
            counts["fix"] += 1
            for key in KEYS:
                if gold.get(key) != label[key] or key not in gold:
                    field_fixes[key] += 1
            note = "schema repair"
            if source == "salvaged":
                note = "category kept from truncated JSON; other fields rebuilt"
            log.append((row["id"], "fix", note + " (" + "; ".join(changes) + ")"))
        else:
            counts["unchanged"] += 1
        new_row = {
            "id": row["id"],
            "messages": [
                row["messages"][0],
                row["messages"][1],
                {"role": "assistant", "content": canonical_json(label)},
            ],
        }
        cleaned.append(new_row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in cleaned:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "action", "reason"])
        writer.writerows(log)
    # Dev check, so a later edit to the rules cannot silently drift.
    dev_path = DATA / "dev.jsonl"
    dev_exact = None
    if dev_path.exists():
        dev_rows = load_jsonl(dev_path)
        hit = 0
        for row in dev_rows:
            gold = json.loads(row["messages"][-1]["content"])
            pred = label_ticket(row["messages"][1]["content"])
            hit += int(pred == gold)
        dev_exact = {"hit": hit, "n": len(dev_rows)}
    summary = {
        "train_rows": len(rows),
        "cleaned_rows": len(cleaned),
        "log_rows": len(log),
        "counts": dict(counts),
        "field_fixes": dict(field_fixes),
        "dev_rule_exact": dev_exact,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_raw_trainable(train_path: Path, out_path: Path) -> int:
    """Drop only rows that cannot be trained on. Labels are otherwise untouched.

    Invalid JSON cannot be a target. Placeholder text is not a ticket.
    The prio/low legacy labels are kept, which is the point of the raw run.
    """
    rows = load_jsonl(train_path)
    kept = []
    for row in rows:
        _, _, body = parse_ticket(row["messages"][1]["content"])
        if is_garbage(body):
            continue
        try:
            gold = json.loads(row["messages"][-1]["content"])
        except json.JSONDecodeError:
            continue
        if not isinstance(gold, dict):
            continue
        kept.append(row)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(kept)


def main():
    train = DATA / "train.jsonl"
    if not train.exists():
        sys.exit(f"missing {train}")
    summary = clean(
        train,
        DATA / "cleaned_train.jsonl",
        ROOT / "cleaning_log.csv",
        ROOT / "cleaning_summary.json",
    )
    raw_n = write_raw_trainable(train, DATA / "raw_trainable.jsonl")
    summary["raw_trainable_rows"] = raw_n
    (ROOT / "cleaning_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
