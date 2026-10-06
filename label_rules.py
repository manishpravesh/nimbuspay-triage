"""Schema rules for NimbusPay triage labels.

Used only to audit and clean train.jsonl. Dev is used as a check that these
rules match SCHEMA.md. Nothing here is applied to model outputs.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP

KEYS = [
    "category",
    "priority",
    "amount_paise",
    "txn_date",
    "txn_id",
    "channel",
    "language",
    "needs_human",
]

_MONTH = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

_HINGLISH = re.compile(
    r"\b("
    r"bhai|mera|mere|meri|mujhe|nahi|nahin|hain|hua|hue|hoon|karo|karun|"
    r"namaste|dhanyavaad|dhanyawad|jaldi|paise|paisa|rupaye|rakam|kaise|"
    r"gaya|gaye|gayi|kya|mein|pehle|baar|yeh|aaj|kal|parso|abhi|aaya|aayi|"
    r"mila|mili|chuka|chuki|dikha|raha|rahi|rahe|wala|wali|karna|thi|tha|"
    r"koi|lekin|hai|ji|band|chuke|chala|chale|kat|diya|nahi|loogut|"
    r"accuont|blcok|tarnsaction|tarnsfer|piase|manag|ktina|kinta|cnacel|"
    r"refnud|prcoess|adderss|udpate|viedo|regsitered|ho\s+gaya|kar\s+diya|"
    r"kar\s+raha|nahi\s+aaya|nahi\s+hua|se\s+hua|ki\s+baat|ka\s+tha"
    r")\b",
    re.I,
)

_GARBAGE = re.compile(
    r"(?i)^(test(\s+test)?(\s+123)?|asdf(\s+asdf)?|lorem ipsum|xxx+|foo|bar)$"
)


def parse_ticket(text: str):
    received = None
    m = re.search(r"Received:\s*(\d{4}-\d{2}-\d{2})", text)
    if m:
        received = datetime.strptime(m.group(1), "%Y-%m-%d").date()
    customer_lines = []
    for line in text.splitlines():
        if line.startswith(">"):
            continue
        customer_lines.append(line)
    customer = "\n".join(customer_lines)
    body_lines = []
    for line in customer.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(("Received:", "Source:", "Subject:", "Regards,", "Regards")):
            continue
        body_lines.append(s)
    body = "\n".join(body_lines).strip()
    return received, customer, body


def is_garbage(body: str) -> bool:
    compact = re.sub(r"\s+", " ", body).strip()
    return compact == "" or _GARBAGE.match(compact) is not None


def _category(body: str) -> str:
    t = body.lower()
    fraud = [
        r"didn'?t make", r"did not make", r"don'?t recogni[sz]e", r"do not recogni[sz]e",
        r"not responsible", r"fraudulent", r"unauthori[sz]ed", r"cash withdraw",
        r"strange withdrawal", r"random charge", r"dont recogni[sz]e",
        r"never seen the name", r"didn'?t take out", r"statement is wrong",
        r"maine nahi kiya", r"mera nahi hai", r"anjaan", r"kisi ne",
        r"koi payment nahi", r"paise nikal", r"piase nikal",
        r"i have a payment that i didn", r"charge i dont",
    ]
    if any(re.search(p, t) for p in fraud):
        return "fraud"
    offers = [
        r"cashback", r"cash back", r"promo", r"coupon", r"scratch", r"referral",
        r"re[aw]{1,2}a?rd", r"rweard", r"joining bonus", r"discount",
    ]
    if any(re.search(p, t) for p in offers):
        return "offers"
    kyc = [
        r"\bkyc\b", r"identit", r"identif", r"aadhaar", r"\bpan\b",
        r"proof of identity", r"\bid verification\b", r"type of id\b",
        r"forms of id\b", r"verify my id\b", r"for the id\b", r"\bverification\b",
    ]
    if any(re.search(p, t) for p in kyc):
        return "kyc"
    refund = [
        r"refund", r"refnud", r"charged twice", r"double charged", r"duplicate (payment|charge)",
        r"charged multiple", r"more than one charge", r"multiple transactions",
        r"returned it", r"return accept", r"changed my mind", r"faulty",
        r"\bcancel", r"cnacel", r"money back", r"put back", r"reversed",
        r"wapas", r"do baar", r"item refund", r"show up twice",
        r"payment show", r"\btwice\b",
    ]
    if any(re.search(p, t) for p in refund):
        return "refund"
    if re.search(r"didn'?t get a pin yet|\bpin yet\b", t):
        return "other"
    access = [
        r"passcode", r"password", r"psasword", r"\bpin\b", r"\botp\b",
        r"sign in", r"log ?in", r"logout", r"loogut", r"code for the app",
        r"account (is )?locked", r"account block", r"account access",
    ]
    if any(re.search(p, t) for p in access):
        return "account_access"
    failed = [
        r"fail", r"didn'?t go through", r"did not go through", r"didn'?t work",
        r"did not work", r"declin", r"unsuccessful", r"not going through",
        r"top[- ]?up", r"topped up", r"couldn'?t make a transfer", r"transfer didn",
        r"transfer failed", r"transfer was unsuccessful", r"could not be completed",
        r"atak", r"atka", r"recharge nahi", r"mile nahi", r"didn'?t reach",
        r"stuck", r"red flag", r"not working", r"didn;t", r"paise kat",
        r"paisa chala", r"amount debit", r"debit ho",         r"unable to finish",
        r"didn'?t process", r"did not process",
        r"didn'?t see the money", r"added to my card",
        r"wasn'?t .{0,40}accepted", r"not accepted", r"\bnot work\b",
        r"couldn'?t (i )?use my card", r"card at a store", r"error is appearing",
        r"make a payment",
    ]
    if any(re.search(p, t) for p in failed):
        return "payment_failed"
    return "other"


def _to_paise(num_str: str, unit: str | None) -> int:
    n = Decimal(num_str.replace(",", "").replace(" ", ""))
    unit = (unit or "").lower()
    if unit == "k":
        n *= 1000
    elif unit in {"lakh", "lakhs"}:
        n *= 100000
    paise = (n * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return int(paise)


_NUM = r"((?<![\d.])(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?)"
_AMOUNT = re.compile(
    rf"(?i)(?:₹|rs\.?|inr)\s*{_NUM}(?:\s*(k|lakhs?|rupees|rupaye|rupaya)\b)?"
    rf"|{_NUM}\s*(k|lakhs?|rupees|rupaye|rupaya)\b"
)


def _strip_distractors(text: str) -> str:
    text = re.sub(r"(?i)\brs\.", "Rs", text)
    money = r"(?:₹|rs\.?|inr)?\s*[\d,.]+(?:\s*(?:k|lakhs?|rupees|rupaye|rupaya))?"
    text = re.sub(
        rf"(?i)(?:my account balance is|account balance is|account mein abhi)\s*{money}(?:\s+right now|\s+pade hain)?",
        " ",
        text,
    )
    text = re.sub(rf"(?i)(?:fyi )?(?:my )?daily limit is\s*{money}", " ", text)
    text = re.sub(r"(?i)bank ref(?:erence)?(?: no)?\.?\s*#?\d+", " ", text)
    text = re.sub(r"(?i)app version(?: is)?\s*\d+(?:\.\d+)*", " ", text)
    text = re.sub(r"(?i)my phone is a [^.\n]*", " ", text)
    text = re.sub(r"(?i)mera phone [^.\n]*", " ", text)
    text = re.sub(
        r"(?i)(i am on the ios app|using the android app|android app use kar raha hoon|"
        r"i am using the android app|i am using app version)",
        " ",
        text,
    )
    text = re.sub(r"(?i)(my earlier ticket number is|pichla ticket number)\s*#?\d+", " ", text)
    text = re.sub(r"#\d+", " ", text)
    return text


def _amount_paise(customer: str) -> int | None:
    cleaned = _strip_distractors(customer)
    found = []
    for m in _AMOUNT.finditer(cleaned):
        if m.group(1):
            found.append(_to_paise(m.group(1), m.group(2)))
        elif m.group(3):
            found.append(_to_paise(m.group(3), m.group(4)))
    if not found:
        return None
    return found[-1]


def _resolve_ymd(year: int | None, month: int, day: int, received: date) -> date | None:
    try:
        if year:
            if year < 100:
                year += 2000
            return date(year, month, day)
        cand = date(received.year, month, day)
        if cand > received:
            cand = date(received.year - 1, month, day)
        return cand
    except ValueError:
        return None


def _txn_date(customer: str, received: date | None) -> str | None:
    if received is None:
        return None
    t = customer
    if re.search(r"(?i)\b(day before yesterday|parso)\b", t):
        return (received - timedelta(days=2)).isoformat()
    if re.search(r"(?i)\b(today|aaj)\b", t):
        return received.isoformat()
    if re.search(r"(?i)\b(yesterday|kal)\b", t):
        return (received - timedelta(days=1)).isoformat()
    m = re.search(r"(?i)\b(\d+)\s+days?\s+ago\b", t)
    if m:
        return (received - timedelta(days=int(m.group(1)))).isoformat()
    m = re.search(r"(?i)\b(\d+)\s+din\s+pehle\b", t)
    if m:
        return (received - timedelta(days=int(m.group(1)))).isoformat()

    found: list[date] = []
    for m in re.finditer(r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b", t):
        d = _resolve_ymd(int(m.group(3)), int(m.group(2)), int(m.group(1)), received)
        if d:
            found.append(d)
    for m in re.finditer(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH})(?:\s+(\d{{4}}))?\b", t, re.I
    ):
        month = _MONTHS[m.group(2).lower()]
        year = int(m.group(3)) if m.group(3) else None
        d = _resolve_ymd(year, month, int(m.group(1)), received)
        if d:
            found.append(d)
    for m in re.finditer(
        rf"\b({_MONTH})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:\s+(\d{{4}}))?\b", t, re.I
    ):
        month = _MONTHS[m.group(1).lower()]
        year = int(m.group(3)) if m.group(3) else None
        d = _resolve_ymd(year, month, int(m.group(2)), received)
        if d:
            found.append(d)
    if not found:
        return None
    # Prefer a date that is not after the ticket arrived.
    found = [d for d in found if d <= received] or found
    return found[-1].isoformat()


def _txn_id(customer: str) -> str | None:
    ids = []
    for m in re.finditer(r"(?i)\bnp[\s\-]*((?:\d[\s\-]*){10})", customer):
        digits = re.sub(r"\D", "", m.group(1))
        if len(digits) == 10:
            ids.append("NP" + digits)
    return ids[0] if ids else None


def _channel(customer: str) -> str | None:
    t = re.sub(
        r"(?i)s(?:c|cr|rc)?ratch card|scratch card|srcatch card|virtual card|"
        r"visa card|mastercard or a visa",
        " ",
        customer,
    )
    if re.search(r"(?i)\bupi\b", t):
        return "upi"
    if re.search(r"(?i)net\s?banking|internet banking", t):
        return "netbanking"
    if re.search(r"(?i)\bwallet\b", t):
        return "wallet"
    if re.search(
        r"(?i)(credit card|debit card|card payment|card transaction|\bcard se\b|"
        r"mode:\s*card|\bvisa\b|\bmastercard\b|\bmy card\b|\bthe card\b|"
        r"\bcard\b)",
        t,
    ):
        return "card"
    return None


def _language(customer: str) -> str:
    return "hi-en" if _HINGLISH.search(customer) else "en"


def _repeat_contact(customer: str) -> bool:
    return re.search(
        r"(?i)(second complaint|third time|fourth time|\b4 times\b|again and again|"
        r"already written|contacted support|keep raising|baar baar|pehle bhi|"
        r"teesri baar|another complaint)",
        customer,
    ) is not None


def _legal(customer: str) -> bool:
    return re.search(
        r"(?i)(\brbi\b|ombudsman|consumer court|lawyer|legal notice|legal action)",
        customer,
    ) is not None


def _priority(category: str, amount: int | None, customer: str) -> str:
    if category == "fraud" or (amount is not None and amount >= 5_000_000) or _legal(customer):
        return "P1"
    if category == "account_access":
        return "P2"
    if category in {"payment_failed", "refund"} and amount is not None and amount >= 500_000:
        return "P2"
    return "P3"


def label_ticket(text: str) -> dict | None:
    """Return the schema label, or None when the ticket has no usable message."""
    received, customer, body = parse_ticket(text)
    if is_garbage(body):
        return None
    category = _category(body)
    if category in {"account_access", "kyc", "other"}:
        amount = txn_date = txn_id = channel = None
    else:
        amount = _amount_paise(customer)
        txn_date = _txn_date(customer, received)
        txn_id = _txn_id(customer)
        channel = _channel(customer)
    priority = _priority(category, amount, customer)
    needs_human = priority == "P1" or _repeat_contact(customer)
    return {
        "category": category,
        "priority": priority,
        "amount_paise": amount,
        "txn_date": txn_date,
        "txn_id": txn_id,
        "channel": channel,
        "language": _language(customer),
        "needs_human": needs_human,
    }


def canonical_json(obj: dict) -> str:
    import json
    ordered = {k: obj[k] for k in KEYS}
    return json.dumps(ordered, ensure_ascii=False, separators=(", ", ": "))
