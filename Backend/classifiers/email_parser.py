"""Reads real .eml files (feature #9) and checks the sender for phishing tricks.

The text models were trained on short SMS messages, so for emails the header
and link checks here are often the strongest evidence. They are reported
separately from the text verdict (like the link-safety panel).
"""
import re
from email import policy
from email.parser import BytesParser
from email.utils import getaddresses, parseaddr
from html.parser import HTMLParser

from classifiers.link_analyzer import BRANDS, analyze_url, registered_domain

FREE_MAIL = {"gmail.com", "googlemail.com", "outlook.com", "hotmail.com", "live.com", "yahoo.com", "aol.com",
             "icloud.com", "protonmail.com", "proton.me", "gmx.com", "mail.ru", "yandex.com", "zoho.com"}
RISKY_EXTENSIONS = {"exe", "scr", "js", "vbs", "bat", "cmd", "com", "msi", "jar", "ps1", "hta", "iso", "img",
                    "html", "htm", "docm", "xlsm", "pptm", "lnk", "wsf"}
MAX_TEXT = 5000

_POINTS = {
    "display_name_brand": 50, "display_name_address": 40, "lookalike_sender": 50, "reply_to_mismatch": 25,
    "return_path_mismatch": 10, "spf_fail": 30, "spf_softfail": 15, "dkim_fail": 25, "dmarc_fail": 40,
    "risky_attachment": 40, "double_extension": 50, "link_text_mismatch": 45,
}


class _HTMLText(HTMLParser):
    """HTML -> visible text, plus every <a href> with the text shown for it."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.links, self._skip, self._href, self._anchor = [], [], 0, None, []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head", "title"):
            self._skip += 1
        elif tag == "a":
            self._href, self._anchor = dict(attrs).get("href"), []
        elif tag in ("br", "p", "div", "tr", "li", "h1", "h2", "h3"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head", "title"):
            self._skip = max(0, self._skip - 1)
        elif tag == "a" and self._href:
            self.links.append({"href": self._href, "text": " ".join("".join(self._anchor).split())})
            self._href = None

    def handle_data(self, data):
        if self._skip:
            return
        self.parts.append(data)
        if self._href is not None:
            self._anchor.append(data)

    def text(self):
        return re.sub(r"[ \t\r\f\v]+", " ", "".join(self.parts)).strip()


def _domain(address):
    return address.rsplit("@", 1)[-1].lower().strip(">") if "@" in address else ""


def _reg(domain):
    return registered_domain(domain) if domain else ""


def parse_eml(data: bytes):
    """Parses raw .eml bytes into headers, body text, links and attachments."""
    msg = BytesParser(policy=policy.default).parsebytes(data)

    def header(name):
        value = msg.get(name)
        return str(value).strip() if value is not None else ""

    from_name, from_addr = parseaddr(header("From"))
    reply_name, reply_addr = parseaddr(header("Reply-To"))
    _, return_path = parseaddr(header("Return-Path"))

    html_links, body_format, text = [], "plain", ""
    plain = msg.get_body(preferencelist=("plain",))
    html = msg.get_body(preferencelist=("html",))
    if html is not None:
        parser = _HTMLText()
        try:
            parser.feed(html.get_content())
            html_links = parser.links
            if plain is None:
                text, body_format = parser.text(), "html"
        except (LookupError, ValueError):
            pass
    if plain is not None:
        try:
            text = plain.get_content()
        except (LookupError, ValueError):
            text = plain.get_payload(decode=True).decode("utf-8", "replace")
    if html is not None and plain is not None:
        body_format = "plain + html"

    attachments = []
    for part in msg.iter_attachments():
        name = part.get_filename() or "(unnamed)"
        payload = part.get_payload(decode=True) or b""
        attachments.append({"filename": name, "content_type": part.get_content_type(), "size": len(payload)})

    auth = {}
    for value in msg.get_all("Authentication-Results", []) + msg.get_all("ARC-Authentication-Results", []):
        for key, result in re.findall(r"\b(spf|dkim|dmarc)=(\w+)", str(value), re.I):
            auth.setdefault(key.lower(), result.lower())
    spf_header = header("Received-SPF")
    if "spf" not in auth and spf_header:
        auth["spf"] = spf_header.split()[0].lower()

    return {
        "from": {"name": from_name, "address": from_addr.lower(), "domain": _domain(from_addr)},
        "reply_to": reply_addr.lower(),
        "return_path": return_path.lower(),
        "to": [a for _, a in getaddresses([header("To")]) if a][:10],
        "subject": header("Subject"),
        "date": header("Date"),
        "auth": auth,
        "body_format": body_format,
        "body_text": text.strip(),
        "html_links": html_links,
        "attachments": attachments,
    }


def check_sender(email):
    """Phishing checks on the headers, attachments and HTML links."""
    findings = []

    def add(kind, message):
        findings.append({"kind": kind, "message": message, "points": _POINTS[kind]})

    sender = email["from"]
    from_reg = _reg(sender["domain"])
    name_key = re.sub(r"[^a-z0-9]", "", sender["name"].lower())

    # Display name claims a brand the address doesn't belong to
    for brand, official in BRANDS.items():
        if len(brand) >= 3 and brand in name_key and from_reg not in official:
            where = "a free email service" if from_reg in FREE_MAIL else from_reg or "an unknown domain"
            add("display_name_brand", f"Display name says “{sender['name']}”, but it was sent from {where}, "
                                      f"not {sorted(official)[0]}")
            break

    embedded = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", sender["name"])
    if embedded and embedded.group(0).lower() != sender["address"]:
        add("display_name_address", f"Display name shows the address {embedded.group(0)}, "
                                    f"but the real sender is {sender['address']}")

    if sender["domain"]:
        for f in analyze_url(sender["domain"])["findings"]:
            if f["kind"] in ("lookalike", "brand_in_wrong_domain"):
                add("lookalike_sender", f"Sender domain: {f['message']}")
                break

    if email["reply_to"] and _reg(_domain(email["reply_to"])) not in ("", from_reg):
        add("reply_to_mismatch", f"Replies go to {email['reply_to']}, a different domain from the sender "
                                 f"({sender['domain']})")
    if email["return_path"] and _reg(_domain(email["return_path"])) not in ("", from_reg):
        add("return_path_mismatch", f"Bounce address ({email['return_path']}) is on a different domain; "
                                    "common for newsletters, suspicious with other red flags")

    auth = email["auth"]
    if auth.get("spf") in ("fail", "softfail"):
        add("spf_fail" if auth["spf"] == "fail" else "spf_softfail",
            f"SPF {auth['spf']}: the sending server isn't authorised to send for {sender['domain']}")
    if auth.get("dkim") == "fail":
        add("dkim_fail", "DKIM fail: the email's signature is invalid (it may have been altered or forged)")
    if auth.get("dmarc") == "fail":
        add("dmarc_fail", f"DMARC fail: {sender['domain']}'s own policy says this email is not genuine")

    for a in email["attachments"]:
        parts = a["filename"].lower().rsplit(".", 2)
        ext = parts[-1] if len(parts) > 1 else ""
        if len(parts) == 3 and parts[1] in ("pdf", "doc", "docx", "xls", "xlsx", "jpg", "png", "txt") \
                and ext in RISKY_EXTENSIONS:
            add("double_extension", f"Attachment “{a['filename']}” pretends to be a .{parts[1]} but is a .{ext}")
        elif ext in RISKY_EXTENSIONS:
            add("risky_attachment", f"Attachment “{a['filename']}” is a .{ext} file, which can run code or "
                                    "open a fake login page")

    # Link text shows one site but the link goes somewhere else
    mismatches = []
    for link in email["html_links"]:
        shown = re.search(r"(?:https?://)?(?:www\.)?([a-z0-9-]+(?:\.[a-z0-9-]+)+)", link["text"].lower())
        href_host = analyze_url(link["href"])["host"] if link["href"] else ""
        if shown and href_host and "." in shown.group(1) and _reg(shown.group(1)) != _reg(href_host):
            mismatches.append({"text": link["text"], "href": link["href"]})
    if mismatches:
        m = mismatches[0]
        add("link_text_mismatch", f"Link text shows “{m['text']}” but actually goes to {m['href']}"
                                  + (f" (+{len(mismatches) - 1} more)" if len(mismatches) > 1 else ""))

    risk = min(100, sum(f["points"] for f in findings))
    level = "high" if risk >= 50 else "medium" if risk >= 25 else "low"
    notes = []
    if not auth:
        notes.append("No SPF/DKIM/DMARC results in the headers (normal for emails saved from some clients).")
    return {"risk": risk, "level": level, "findings": findings, "notes": notes, "link_mismatches": mismatches}


def email_text_for_models(email):
    """Subject + body, trimmed to what the text models accept."""
    text = f"{email['subject']}\n\n{email['body_text']}".strip()
    return re.sub(r"\n{3,}", "\n\n", text)[:MAX_TEXT]
