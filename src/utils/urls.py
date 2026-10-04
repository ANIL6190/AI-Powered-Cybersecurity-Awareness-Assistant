"""
URL normalisation and registrable-domain extraction.

The same canonicalisation is applied to every dataset so that collection format
(scheme, leading "www.", trailing slash) cannot act as a label shortcut. In
PhiUSIIL every legitimate URL is "https://www.<domain>" with no path, while
phishing URLs mix http/https, www/no-www and trailing slashes; a character model
on raw URLs would learn that formatting instead of phishing signals.

Registrable domains come from tldextract using its bundled Public Suffix List
snapshot only (no network access), so grouping is reproducible offline.
"""

import ipaddress
import re
import urllib.parse
from functools import lru_cache
from typing import Optional

import tldextract

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_WWW_RE = re.compile(r"^www\d*\.")

# Offline extractor: suffix_list_urls=() forces the bundled snapshot, cache_dir=None avoids disk writes.
_EXTRACT_ICANN = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=False)
_EXTRACT_PRIVATE = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True)


def canonicalize_url(
    url: str,
    strip_scheme: bool = True,
    strip_www: bool = True,
    strip_trailing_slash: bool = True,
) -> str:
    """Lowercase the host, drop scheme / leading www / trailing slash, keep path, query and fragment."""
    if url is None:
        return ""
    s = str(url).strip()
    if not s or s.lower() in {"nan", "none"}:
        return ""

    scheme_match = _SCHEME_RE.match(s)
    rest = s[scheme_match.end():] if scheme_match else s
    if not strip_scheme and scheme_match:
        prefix = scheme_match.group(0).lower()
    else:
        prefix = ""

    # Split host from the remainder at the first "/", "?" or "#"
    m = re.match(r"^([^/?#]*)(.*)$", rest, flags=re.DOTALL)
    host, tail = (m.group(1), m.group(2)) if m else (rest, "")
    host = host.lower()
    if strip_www:
        host = _WWW_RE.sub("", host)
    if strip_trailing_slash:
        tail = tail.rstrip("/")
    return f"{prefix}{host}{tail}"


def extract_host(url: str) -> str:
    """Host part of a URL (no scheme, credentials or port), lowercased."""
    s = str(url).strip()
    if not _SCHEME_RE.match(s):
        s = "http://" + s
    try:
        host = urllib.parse.urlsplit(s).hostname or ""
    except ValueError:
        host = ""
    return host.lower().rstrip(".")


def host_only(canonical_url: str) -> str:
    """Host of an already-canonical URL, with leading www removed."""
    return _WWW_RE.sub("", extract_host(canonical_url))


def has_path(canonical_url: str) -> bool:
    """True when anything beyond the host remains (path, query or fragment)."""
    return bool(re.search(r"[/?#].", canonical_url))


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


@lru_cache(maxsize=None)
def registrable_domain(host: str, include_private: bool = False) -> str:
    """
    eTLD+1 of a host (e.g. 'a.b.example.co.uk' -> 'example.co.uk').
    Falls back to the full host for IP literals or hosts with no known suffix,
    so every row always has a non-empty group key.
    """
    host = (host or "").lower().strip(".")
    if not host:
        return ""
    if _is_ip(host):
        return host
    extractor = _EXTRACT_PRIVATE if include_private else _EXTRACT_ICANN
    ext = extractor(host)
    domain: Optional[str] = getattr(ext, "top_domain_under_public_suffix", None) or ext.registered_domain
    return domain or host


def url_representation(canonical_url: str, mode: str = "canonical_url") -> str:
    """Model input string for the chosen representation."""
    if mode == "canonical_url":
        return canonical_url
    if mode == "host_only":
        return host_only(canonical_url)
    raise ValueError(f"Unknown url.representation '{mode}' (expected 'canonical_url' or 'host_only').")
