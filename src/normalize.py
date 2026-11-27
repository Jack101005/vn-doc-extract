"""Normalize field values so predictions and labels can be compared fairly."""
import re
import unicodedata

_NUM = re.compile(r"\d[\d.,]*\d|\d")
_DATE = re.compile(r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{2,4})")
_TIME = re.compile(r"(\d{1,2})\s*[:hH]\s*(\d{2})")


def norm_text(s):
    if s is None:
        return None
    s = unicodedata.normalize("NFC", str(s)).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def norm_total(s):
    """'Tổng tiền: 1,475,000' -> 1475000 ; '45.400' -> 45400 ; '54,000.00' -> 54000"""
    if s is None:
        return None
    nums = _NUM.findall(str(s))
    if not nums:
        return None
    n = nums[-1]
    n = re.sub(r"[.,]\d{1,2}$", "", n)  # drop decimal cents
    digits = re.sub(r"\D", "", n)
    return int(digits) if digits else None


def norm_timestamp(s):
    """'Ngày: 11/08/2020 - 12:55' -> ('2020-08-11', '12:55'). Missing part -> None."""
    if s is None:
        return None
    s = str(s)
    date = None
    m = _DATE.search(s)
    if m:
        d, mo, y = (int(x) for x in m.groups())
        if y < 100:
            y += 2000
        date = f"{y:04d}-{mo:02d}-{d:02d}"
        s = s[:m.start()] + " " + s[m.end():]
    t = _TIME.search(s)
    time = f"{int(t.group(1)):02d}:{t.group(2)}" if t else None
    if date is None and time is None:
        return None
    return (date, time)
