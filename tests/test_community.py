"""The OpenModelDB parser, against a recorded snapshot of the real page.

`fetch_openmodeldb` is deliberately not called here: a pull-request suite must
not depend on somebody else's website. The committed snapshot proves the parser
still parses the markup it was written against; it cannot notice the site
changing underneath it, because a snapshot pins exactly what the selectors
target. That is the gap the weekly CI job in Phase 7 fills — it calls the live
fetch and opens an issue, never a pull request.

**When that job must open an issue.** The live listing serves twelve cards
server-side on 2026-09-26, so a naive "fewer than ten entries" rule sits one
model away from firing on every run, and a job that cries wolf is a job nobody
reads. The rule is therefore:

* Open an issue when the parse yields **zero** cards — the selectors have
  stopped matching, which is unambiguous drift.
* Otherwise, record the count. Open an issue only when it falls **below half
  of the smallest count ever recorded** — six today. A site that reorganises
  itself and serves ten or eleven cards is still serving a working listing, and
  that belongs in the job log, not an issue tracker.
* Re-baseline deliberately, in a pull request, once a change is understood.

The count is never compared against a number written here; it is compared
against what the job itself has seen, so the threshold cannot silently drift
out of date.

One honest limitation, repeated so the tests below are not read as promising
more than they deliver: a listing card carries no download link — the file lives
on the model's own page — so both snapshot entries have `direct_url=None`, and
the Community tab opens the model page in a browser. The fragments composed in
`_card` for the host-classification tests are the real ones: the card wrapper is
the snapshot's own markup, the download anchor is the button a model page
renders. That is the only way to reach the classification honestly.
"""

from __future__ import annotations

from pathlib import Path

from core.community import CommunityModel, parse_openmodeldb

FIXTURE = Path(__file__).parent / "fixtures" / "openmodeldb_page1.html"

# The download button as a model page renders it, copied from
# https://openmodeldb.info/models/2x-90s-Sonic on 2026-09-26 with the href as
# the only thing changed.
_DOWNLOAD_BUTTON = (
    '<a rel="noopener noreferrer" target="_blank" class="inline-flex h-20 w-full '
    'cursor-pointer items-center rounded-l-lg border-0 bg-accent-600" href="{href}" '
    'type="button">Download</a>'
)


def _card(
    *,
    name: str = "A model",
    scale: str | None = "2x",
    architecture: str = "Compact",
    href: str = "/models/a-model",
    download: str | None = None,
) -> str:
    """A model card in the live listing's markup, with the scale pill optional."""
    badges = [
        f'<div class="model-card-module-scss-module__mGDelq__tagBase '
        f'backdrop-blur-sm bg-fade-900/70 text-fade-100">{architecture}</div>'
    ]
    if scale is not None:
        badges.append(
            '<div class="model-card-module-scss-module__mGDelq__tagBase '
            f'backdrop-blur-sm bg-accent-600 text-white">{scale}</div>'
        )
    return (
        '<div class="model-card-module-scss-module__mGDelq__modelCard '
        'model-card-module-scss-module__mGDelq__overflowHidden"><div class='
        '"model-card-module-scss-module__mGDelq__inner"><div class='
        '"model-card-module-scss-module__mGDelq__topTags">'
        + "".join(badges)
        + '</div><a class="model-card-module-scss-module__mGDelq__name block '
        f'text-base font-semibold leading-snug text-ink line-clamp-2" href="{href}">'
        f"{name}</a>"
        '<div class="truncate text-sm text-ink-muted">by <a class="font-medium '
        'text-accent-text hover:underline" href="/users/someone">someone</a></div>'
        + (download or "")
        + '<div class="model-card-module-scss-module__mGDelq__tagRow text-xs">'
        '<a class="editable-tags-module-scss-module__Bz5F5W__tag" '
        'href="/?t=anime">Anime</a></div></div></div>'
    )


def test_snapshot_reads_the_recorded_models() -> None:
    """Every field the Community tab shows comes out of the real markup."""
    models = parse_openmodeldb(FIXTURE.read_text(encoding="utf-8"))

    assert models == [
        CommunityModel(
            name="90s Sonic 2x (Small)",
            architecture="Compact",
            scale=2,
            author="pokepress",
            tags=("Cartoon", "Restoration", "Video Frame"),
            page_url="https://openmodeldb.info/models/2x-90s-Sonic",
            direct_url=None,
            host="other",
        ),
        CommunityModel(
            name="90s Sonic 2x (Large)",
            architecture="RealPLKSR",
            scale=2,
            author="pokepress",
            tags=("Cartoon", "Restoration", "Video Frame"),
            page_url="https://openmodeldb.info/models/2x-90s-Sonic-LG",
            direct_url=None,
            host="other",
        ),
    ]


def test_a_listing_card_has_no_download_to_offer() -> None:
    """The truth the Community tab has to render: open the page, do not fetch."""
    first, _ = parse_openmodeldb(FIXTURE.read_text(encoding="utf-8"))

    assert first.direct_url is None
    assert first.host == "other"
    assert first.page_url.endswith("/models/2x-90s-Sonic")


def test_a_github_download_is_served_directly() -> None:
    """A raw file link is what the downloader can fetch without a browser."""
    link = "https://github.com/Phhofm/models/raw/main/2xHFA2kAVCSRFormer_light.pth"
    (model,) = parse_openmodeldb(_card(download=_DOWNLOAD_BUTTON.format(href=link)))

    assert model.direct_url == link
    assert model.host == "github"


def test_a_huggingface_download_is_served_directly() -> None:
    link = "https://huggingface.co/ai-forever/RealESRGAN/resolve/main/model.pth"
    (model,) = parse_openmodeldb(_card(download=_DOWNLOAD_BUTTON.format(href=link)))

    assert model.direct_url == link
    assert model.host == "huggingface"


def test_a_drive_download_is_only_a_page_to_open() -> None:
    """Google Drive cannot hand a file to a plain GET, so it is not direct."""
    link = "https://drive.google.com/file/d/1AbCdEfGhIjKlMnOp/view"
    (model,) = parse_openmodeldb(_card(download=_DOWNLOAD_BUTTON.format(href=link)))

    assert model.direct_url is None
    assert model.host == "drive"
    assert model.page_url == "https://openmodeldb.info/models/a-model"


def test_a_landing_page_host_is_neither_direct_nor_drive() -> None:
    """MediaFire answers with a countdown page; saving it as a model is the bug."""
    link = "https://www.mediafire.com/file/d9wyxoux6e9kllv/90s_Sonic_2x.pth/file"
    (model,) = parse_openmodeldb(_card(download=_DOWNLOAD_BUTTON.format(href=link)))

    assert model.direct_url is None
    assert model.host == "other"


def test_a_card_without_a_scale_is_skipped() -> None:
    """An assumed 4x would offer a 2x or 16x model as something it is not."""
    assert parse_openmodeldb(_card(scale=None)) == []


def test_a_page_without_cards_yields_nothing() -> None:
    """The drift signal: the fetch turns this empty result into an error."""
    assert parse_openmodeldb("<html><body><p>Nothing here</p></body></html>") == []


def test_links_resolve_against_the_page_that_was_fetched() -> None:
    """A listing reached through a mirror must not stamp the canonical host on it."""
    model = parse_openmodeldb(
        _card(
            href="/models/a-model",
            download=_DOWNLOAD_BUTTON.format(href="/dl/a-model.pth"),
        ),
        base_url="https://mirror.example/listing",
    )[0]

    assert model.page_url == "https://mirror.example/models/a-model"
    assert model.host == "other"


def test_architecture_and_author_survive_a_card_without_them() -> None:
    """A card missing its badges is still a model, and empty is not a guess."""
    (model,) = parse_openmodeldb(
        '<div class="model-card-module-scss-module__mGDelq__modelCard">'
        '<div class="model-card-module-scss-module__mGDelq__topTags">'
        '<div class="model-card-module-scss-module__mGDelq__tagBase bg-accent-600">4x'
        "</div></div>"
        '<a class="model-card-module-scss-module__mGDelq__name" href="/models/bare">'
        "Bare</a></div>"
    )

    assert model.name == "Bare"
    assert model.scale == 4
    assert model.architecture == ""
    assert model.author == ""
    assert model.tags == ()
