"""De-obfuscation (feature #4): undo common tricks spammers use to dodge filters,
plus the attack generators used by the robustness playground and benchmark.

normalize("FR33 C@SH, c l a i m now") -> "FREE CASH, claim now"
"""
import re
import unicodedata

# Zero-width / invisible characters inserted to break up words
_INVISIBLE = re.compile("[­​-‏⁠-⁤﻿]")

# Cyrillic/Greek letters that look identical to Latin ones
HOMOGLYPHS = {
    "а": "a", "в": "b", "е": "e", "к": "k", "м": "m", "н": "h", "о": "o", "р": "p",
    "с": "c", "т": "t", "у": "y", "х": "x", "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d",
    "ӏ": "l", "ɡ": "g", "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H",
    "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X", "І": "I", "Ј": "J", "Ѕ": "S",
    "α": "a", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "ο": "o", "ρ": "p", "τ": "t",
    "υ": "u", "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
}
_HOMOGLYPH_RE = re.compile("[" + "".join(HOMOGLYPHS) + "]")

LEET = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i", "|": "l"}
_TOKEN = re.compile(r"\S+")
_LOOKS_LIKE_URL = re.compile(r"://|^www\.|\.[a-z]{2,}(/|$)", re.IGNORECASE)

# "f r e e", "f.r.e.e", "f-r-e-e": 3+ single letters joined by the *same*
# separator (the backreference stops "f.r.e.e e.n.t.r.y" merging into one word)
_SPACED = re.compile(r"(?<![A-Za-z])[A-Za-z]([ .\-_*])(?:[A-Za-z]\1)+[A-Za-z](?![A-Za-z])")
# Numbers with units are not leetspeak: 3pm, 10am, 4th, 2nd, 5mins, 100gb
_NUMBER_WITH_UNIT = re.compile(r"^\d+(am|pm|st|nd|rd|th|p|k|m|s|x|min|mins|hr|hrs|gb|mb|km|kg|yr|yrs)$", re.I)
# Short spaced words are only joined if they are known spam words, so ordinary
# SMS text like "u r a star" is left alone
_SHORT_SPAM_WORDS = {"win", "won", "txt", "sms", "fee", "sex", "now", "cash"}


def _deleet_token(token):
    """Fixes leetspeak inside a word ("fr33", "c@sh", "w1nner") but leaves real
    SMS shorthand and numbers alone ("4u", "2nite", "150p", "£500", 0800 numbers)."""
    letters = sum(c.isalpha() for c in token)
    leet = sum(c in LEET for c in token)
    if letters < 2 or leet == 0 or re.search(r"\d{3,}", token) or _LOOKS_LIKE_URL.search(token):
        return token  # URLs are left exactly as written (paypa1.com must stay paypa1.com)
    core = token.strip(".,;:?!\"'()[]")
    if not core or _NUMBER_WITH_UNIT.match(core):
        return token
    fixed = "".join(LEET.get(c, c) for c in core)
    return token.replace(core, fixed, 1)


def _join_spaced(match):
    joined = re.sub(r"[ .\-_*]", "", match.group(0))
    if len(joined) >= 4 or joined.lower() in _SHORT_SPAM_WORDS:
        return joined
    return match.group(0)


def normalize(text):
    """Returns (normalized_text, list of tricks that were undone)."""
    steps = []
    out = unicodedata.normalize("NFKC", text)
    if out != text:
        steps.append("unicode look-alike characters")

    new = _INVISIBLE.sub("", out)
    if new != out:
        steps.append("invisible zero-width characters")
    out = new

    new = _HOMOGLYPH_RE.sub(lambda m: HOMOGLYPHS[m.group(0)], out)
    if new != out:
        steps.append("Cyrillic/Greek look-alike letters")
    out = new

    new = _SPACED.sub(_join_spaced, out)
    if new != out:
        steps.append("spaced-out letters")
        # The message clearly uses the trick, so also join short runs ("Y o u")
        new = _SPACED.sub(lambda m: re.sub(r"[ .\-_*]", "", m.group(0)), new)
    out = new

    new = _TOKEN.sub(lambda m: _deleet_token(m.group(0)), out)
    if new != out:
        steps.append("leetspeak")
    out = new

    return out, steps


# ---------------------------------------------------------------- attacks ---
# Each attack rewrites words of 3+ letters the way real spam does.

def _per_word(text, fn):
    return re.sub(r"[A-Za-z]{3,}", lambda m: fn(m.group(0)), text)


ATTACKS = {
    "leetspeak": lambda t: _per_word(t, lambda w: w.translate(str.maketrans("eoiasEOIAS", "3014$3014$"))),
    "spaced_letters": lambda t: "  ".join(_per_word(w, lambda x: " ".join(x)) for w in t.split()),
    "dotted_letters": lambda t: _per_word(t, lambda w: ".".join(w)),
    "homoglyphs": lambda t: t.translate(str.maketrans("aeopcxyAEOPCX", "аеорсхуАЕОРСХ")),
    "zero_width": lambda t: _per_word(t, lambda w: "​".join(w)),
    "mixed_case": lambda t: "".join(c.upper() if i % 2 else c.lower() for i, c in enumerate(t)),
}

ATTACK_DESCRIPTIONS = {
    "leetspeak": "Letters swapped for look-alike digits/symbols (free → fr33)",
    "spaced_letters": "Letters separated by spaces (free → f r e e)",
    "dotted_letters": "Letters separated by dots (free → f.r.e.e)",
    "homoglyphs": "Latin letters swapped for identical-looking Cyrillic ones",
    "zero_width": "Invisible zero-width characters inserted inside words",
    "mixed_case": "Alternating upper/lower case (control: should not matter)",
}
