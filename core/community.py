"""The community model catalogue, read out of OpenModelDB's HTML.

`openmodeldb.info` publishes no JSON API — `/api/v1/models` answers HTTP 404 —
so the listing is scraped with `selectolax`. Every selector is a module
constant with a comment naming the markup it targets, so a layout change is a
one-line fix rather than a treasure hunt.

Three facts about the live site shape this module, all re-checked on 2026-09-26
against the untouched page:

* **The listing is served from the site root.** `https://openmodeldb.info/models`
  answers HTTP 404, which is why `OPENMODELDB_URL` is the root.
* **The server sends twelve cards and 659 empty `placeholder` tiles.** The grid
  is rendered for the full model count and every tile past the twelfth is an
  empty `<div>` the client hydrates from its own payload. Dropping them is
  correct, not a parse failure: a placeholder carries no name, no scale and no
  author, so `parse_openmodeldb` skips it for exactly the reason it skips an
  empty element. Twelve is how many models the listing can yield server-side,
  not how many OpenModelDB has.
* **A card carries no download link at all.** The file lives on the model's own
  page, behind a single `a[type="button"]`. This is a deliberate limit, not an
  oversight: the Community tab opens the model page in a browser rather than
  scraping one detail page per model, because N detail pages would drift at
  least as fast as the listing does and would turn a browsing tab into 671 HTTP
  requests. The curated Models tab is the download path; the Community tab is a
  browser. `direct_url` is already wired for the case where a page does put the
  link on the card, and is filled in only for hosts that serve file bytes to a
  plain GET: GitHub raw and Hugging Face do, Google Drive, MediaFire, Mega and
  the rest answer with an interstitial page, and saving one of those as a
  `.pth` is the exact failure `ModelFileTooSmall` exists to catch.

`fetch_openmodeldb` treats a page that yields zero cards as a hard error. That
is the drift signal the scheduled CI job keys on: the committed snapshot proves
the parser still parses, but a snapshot pins the very markup the selectors
target and so cannot notice the site moving underneath them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from selectolax.parser import HTMLParser

from .download import USER_AGENT
from .errors import DownloadError

OPENMODELDB_URL = "https://openmodeldb.info/"

# The grid tile. Matched on the un-hashed part of the class, because the build
# hash after it (`__mGDelq` on 2026-09-26) changes on every deploy. The 659
# `placeholder` siblings match it as well and are dropped below for having no
# name, scale or author to read — they are empty tiles, not models.
_CARD = 'div[class*="modelCard"]'
# The model name, and the one link a card is guaranteed to carry.
_TITLE = 'a[class*="__name"]'
# The badge row over the thumbnail: architecture first, then the "2x" pill.
_ARCH = 'div[class*="__topTags"] div[class*="__tagBase"]'
# The scale pill is the accent-coloured badge; the architecture badge is the
# muted one beside it.
_SCALE = 'div[class*="__topTags"] div[class*="__tagBase"][class*="bg-accent"]'
# "by <a href="/users/<name>">", the only author link inside a card.
_AUTHOR = 'a[href*="/users/"]'
# The category chips under the description; the top badges are not tags.
_TAGS = 'div[class*="__tagRow"] a'
# The download button a model page renders. A listing card has none today; it
# is here so a page that does carry one is read rather than ignored.
_DOWNLOAD = 'a[type="button"]'

# Always used with `fullmatch`: a badge that merely contains a number is an
# architecture like "16xESRGAN", not a scale.
_SCALE_RE = re.compile(r"(\d+)\s*x", re.IGNORECASE)

_HOSTS = {
    "github.com": "github",
    "huggingface.co": "huggingface",
    "drive.google.com": "drive",
}

# Hosts whose link is the file itself. Everything else is a landing page.
_DIRECT_HOSTS = frozenset({"github", "huggingface"})


@dataclass(frozen=True)
class CommunityModel:
    """One model as the Community tab shows it.

    Attributes:
        name: The model's display name, as the site spells it.
        architecture: The badge above the thumbnail, e.g. `Compact`.
        scale: The model's factor, read from the badge beside the architecture.
        author: The username shown after "by".
        tags: The category chips, not the architecture and scale badges.
        page_url: Absolute link to the model's page on the site.
        direct_url: The file itself, when a host serves bytes to a plain GET,
            and `None` otherwise — a Drive or MediaFire link is a page to open,
            not a file to fetch.
        host: `github`, `huggingface`, `drive`, or `other`, which also covers a
            card that carries no download link at all.
    """

    name: str
    architecture: str
    scale: int
    author: str
    tags: tuple[str, ...]
    page_url: str
    direct_url: str | None
    host: str


def host_for(url: str) -> str:
    """Classify a download host as `github`, `huggingface`, `drive` or `other`.

    `other` also covers a card with no download link at all, which is what a
    listing scrape produces today.
    """
    netloc = urlparse(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return _HOSTS.get(netloc, "other")


def _scale_of(card: Any) -> int | None:
    """The card's scale, or `None` when it has none.

    A card without a scale is skipped by the caller rather than assumed to be
    4x: the scale decides whether the model can be used at all, and a wrong
    default would offer a 2x or 16x model as a 4x one. The badge has to *be*
    the scale, not merely contain a number, so an architecture called
    `16xESRGAN` is not read as a 16x scale.
    """
    node = card.css_first(_SCALE)
    if node is None:
        return None
    match = _SCALE_RE.fullmatch(node.text(strip=True))
    return int(match.group(1)) if match else None


def _architecture_of(card: Any) -> str:
    """The architecture badge: the first one that is not the scale pill."""
    for badge in card.css(_ARCH):
        text: str = badge.text(strip=True)
        if text and not _SCALE_RE.fullmatch(text):
            return text
    return ""


def parse_openmodeldb(
    html: str, base_url: str = OPENMODELDB_URL
) -> list[CommunityModel]:
    """Read a model listing out of a page, resolving links against `base_url`.

    Cards without a name or without a scale are dropped, so an unrelated
    element that happens to match `_CARD` costs nothing.
    """
    models: list[CommunityModel] = []
    for card in HTMLParser(html).css(_CARD):
        title = card.css_first(_TITLE)
        if title is None:
            continue
        name = title.text(strip=True)
        page_href = title.attributes.get("href")
        scale = _scale_of(card)
        if not name or not page_href or scale is None:
            continue

        architecture = _architecture_of(card)
        author = card.css_first(_AUTHOR)
        download = card.css_first(_DOWNLOAD)
        download_href = (
            download.attributes.get("href") if download is not None else None
        )
        link = urljoin(base_url, download_href) if download_href else None
        host = host_for(link) if link else "other"

        models.append(
            CommunityModel(
                name=name,
                architecture=architecture,
                scale=scale,
                author=author.text(strip=True) if author else "",
                tags=tuple(
                    node.text(strip=True)
                    for node in card.css(_TAGS)
                    if node.text(strip=True)
                ),
                page_url=urljoin(base_url, page_href),
                direct_url=link if host in _DIRECT_HOSTS else None,
                host=host,
            )
        )
    return models


def fetch_openmodeldb(
    url: str = OPENMODELDB_URL, timeout: float = 30.0
) -> list[CommunityModel]:
    """Fetch a listing and parse it, or raise `DownloadError`.

    A page that yields no cards raises too. That is the drift alarm, and it is
    the condition the weekly CI job reports: the snapshot in `tests/` cannot see
    the live site change, so this call is the only check that can.
    """
    try:
        response = requests.get(
            url, timeout=timeout, headers={"User-Agent": USER_AGENT}
        )
    except requests.RequestException as exc:
        raise DownloadError(f"Could not reach {url}: {exc}") from exc
    if response.status_code != 200:
        raise DownloadError(f"{url} answered HTTP {response.status_code}")
    # No charset in the header, so requests falls back to apparent_encoding;
    # the page is UTF-8 and a wrong guess would mangle every model name.
    models = parse_openmodeldb(response.text, base_url=url)
    if not models:
        raise DownloadError(
            f"{url} yielded no model cards — the page layout has changed"
        )
    return models
