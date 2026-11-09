"""Offline phishing-link analysis (feature #8).

Finds URLs and bare domains in a message and checks each one for common
phishing tricks. Nothing is fetched: no network calls, so it is fast, private,
and safe to run on untrusted messages.
"""
import re
from urllib.parse import urlsplit

from classifiers.normalize import HOMOGLYPHS


def HOMOGLYPHS_TO_LATIN(name):
    return "".join(HOMOGLYPHS.get(c, c) for c in name).lower()


URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "rebrand.ly",
    "cutt.ly", "shorturl.at", "tiny.cc", "rb.gy", "bl.ink", "s.id", "t.ly", "v.gd", "lnkd.in",
}
SUSPICIOUS_TLDS = {
    "zip", "mov", "xyz", "top", "tk", "ml", "ga", "cf", "gq", "click", "link", "work", "loan",
    "win", "bid", "country", "kim", "review", "party", "rest", "fit", "icu", "cam", "buzz", "monster",
}
# Two-part public suffixes, so "paypal.co.uk" is treated as one registered domain
SECOND_LEVEL_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "me.uk", "com.au", "net.au", "org.au", "co.in", "org.in",
    "gov.in", "co.jp", "com.br", "co.nz", "co.za", "com.cn", "com.sg", "com.mx",
}
# Brand -> its real registered domains
BRANDS = {
    "paypal": {"paypal.com", "paypal.me"},
    "apple": {"apple.com", "icloud.com"},
    "amazon": {"amazon.com", "amazon.co.uk", "amazon.in", "amazon.de", "amazon.com.au"},
    "microsoft": {"microsoft.com", "live.com", "outlook.com", "office.com"},
    "google": {"google.com", "gmail.com", "youtube.com"},
    "netflix": {"netflix.com"},
    "facebook": {"facebook.com", "fb.com", "meta.com"},
    "instagram": {"instagram.com"},
    "whatsapp": {"whatsapp.com"},
    "linkedin": {"linkedin.com"},
    "dhl": {"dhl.com", "dhl.co.uk"},
    "fedex": {"fedex.com"},
    "royalmail": {"royalmail.com"},
    "hmrc": {"gov.uk"},
    "chase": {"chase.com"},
    "wellsfargo": {"wellsfargo.com"},
    "bankofamerica": {"bankofamerica.com"},
    "hsbc": {"hsbc.com", "hsbc.co.uk"},
    "barclays": {"barclays.co.uk", "barclays.com"},
    "hdfcbank": {"hdfcbank.com"},
    "icicibank": {"icicibank.com"},
    "onlinesbi": {"onlinesbi.sbi", "sbi.co.in"},
    "paytm": {"paytm.com"},
    "coinbase": {"coinbase.com"},
    "binance": {"binance.com"},
}
SENSITIVE_PATH_WORDS = ("login", "signin", "verify", "account", "update", "secure", "password",
                        "confirm", "wallet", "invoice", "unlock", "billing", "kyc")

# Look-alike characters used inside domain names
_DOMAIN_LOOKALIKES = [("rn", "m"), ("vv", "w"), ("0", "o"), ("1", "l"), ("3", "e"), ("5", "s"), ("@", "a")]

_COMMON_TLDS = (
    "com|net|org|info|biz|io|co|uk|us|ru|cn|in|de|fr|me|app|dev|site|online|shop|store|"
    "ly|gl|gd|cc|at|id|to|tv|ai|gg|so|"
    + "|".join(sorted(SUSPICIOUS_TLDS))
)
_URL_RE = re.compile(
    r"(?:https?://|www\.)[^\s<>\"']+"                                   # explicit links
    rf"|\b(?:[a-z0-9-]+\.)+(?:{_COMMON_TLDS})\b(?:/[^\s<>\"']*)?",      # bare domains: paypa1.com/login
    re.IGNORECASE,
)

# Points per finding; risk = min(100, sum)
_POINTS = {
    "ip_host": 45, "lookalike": 60, "brand_in_wrong_domain": 50, "punycode": 40, "at_symbol": 50,
    "shortener": 25, "suspicious_tld": 25, "many_subdomains": 15, "sensitive_path": 15,
    "no_https": 10, "long_url": 10,
}


def _levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def registered_domain(host):
    labels = host.split(".")
    if len(labels) >= 3 and ".".join(labels[-2:]) in SECOND_LEVEL_SUFFIXES:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _unlookalike(name):
    for fake, real in _DOMAIN_LOOKALIKES:
        name = name.replace(fake, real)
    return name


def analyze_url(raw):
    url = raw.rstrip(".,;:!?)]}'\"")
    parts = urlsplit(url if "://" in url else "http://" + url)
    netloc = parts.netloc.lower()
    host = (parts.hostname or "").lower()
    findings = []

    def add(kind, message):
        findings.append({"kind": kind, "message": message, "points": _POINTS[kind]})

    official_site = False
    if "@" in netloc:
        add("at_symbol", "Contains '@': the browser ignores everything before it, so the real site is "
                         f"'{host}'")
    if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host):
        add("ip_host", "Links to a raw IP address instead of a domain name")
    elif host:
        reg = registered_domain(host)
        name = reg.split(".")[0]
        tld = host.rsplit(".", 1)[-1]

        if any(label.startswith("xn--") for label in host.split(".")):
            try:
                shown = host.encode("ascii").decode("idna")
            except UnicodeError:
                shown = host
            add("punycode", f"Internationalised (punycode) domain that displays as '{shown}', "
                            "which can imitate a real site")
            # Compare what the user *sees* (Cyrillic а → a) against brand names
            name = HOMOGLYPHS_TO_LATIN(registered_domain(shown).split(".")[0])
        if reg in URL_SHORTENERS:
            add("shortener", f"URL shortener ({reg}) hides the real destination")
        if tld in SUSPICIOUS_TLDS:
            add("suspicious_tld", f"'.{tld}' is a top-level domain heavily used for spam/phishing")
        if host.count(".") >= 4:
            add("many_subdomains", f"Unusually many subdomains ({host})")

        for brand, official in BRANDS.items():
            if reg in official:
                official_site = True
                break  # it is the real site
            # Not an official domain, yet the name is (or nearly is) the brand:
            # paypa1.com, amaz0n.com, paypal.xyz, Cyrillic pаypal.com
            cleaned = _unlookalike(name)
            if cleaned == brand or (len(brand) >= 5 and _levenshtein(name, brand) == 1):
                add("lookalike", f"'{reg}' imitates {brand} (real site: {sorted(official)[0]})")
                break
            # Brand as a whole word elsewhere in the host: paypal.com.evil.top, dhl-delivery.xyz
            if brand in re.split(r"[.\-]", host):
                add("brand_in_wrong_domain", f"Uses the name '{brand}' but the real domain is '{reg}', "
                                             f"not {sorted(official)[0]}")
                break

    path = (parts.path + "?" + parts.query).lower()
    hits = [w for w in SENSITIVE_PATH_WORDS if w in path]
    if hits and not official_site:
        add("sensitive_path", f"Path asks for sensitive actions ({', '.join(hits)})")
    if url.lower().startswith("http://"):
        add("no_https", "Not encrypted (http, not https)")
    if len(url) > 100:
        add("long_url", f"Very long URL ({len(url)} characters)")

    risk = min(100, sum(f["points"] for f in findings))
    level = "high" if risk >= 50 else "medium" if risk >= 25 else "low"
    return {"url": url, "host": host, "risk": risk, "level": level, "findings": findings}


def analyze_links(text):
    """Returns {"links": [...], "max_risk": int, "level": "none|low|medium|high"}."""
    seen, links = set(), []
    for match in _URL_RE.finditer(text):
        raw = match.group(0)
        if raw.lower() in seen:
            continue
        seen.add(raw.lower())
        result = analyze_url(raw)
        result.update(start=match.start(), end=match.start() + len(result["url"]))
        links.append(result)
    max_risk = max((link["risk"] for link in links), default=0)
    level = "none" if not links else ("high" if max_risk >= 50 else "medium" if max_risk >= 25 else "low")
    return {"links": links, "max_risk": max_risk, "level": level}
