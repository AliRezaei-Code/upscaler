"""The live OpenModelDB check — deselected in pull requests, run on a schedule.

`tests/test_community.py` parses a committed snapshot, which pins the *old*
markup: the selectors and the fixture always agree, so that test cannot detect
the site changing. This one is the only test in the suite that talks to
openmodeldb.info, and it answers the other question — does the parser still
read the page as it is served today.

It is marked `network` and excluded from the PR workflow on purpose. It runs in
a weekly `schedule:` job, and the workflow opens an issue (never a pull
request) when it fails, because a site changing its markup is not something a
branch can fix.

**The threshold is deliberately loose.** The listing serves about a dozen
server-rendered cards today, and a hard "fewer than 10" rule would open an issue
every time the site changes its pagination or drops a model. The test fails
only when the parse collapses — fewer than a quarter of the cards a working
page serves, or none at all — which is the signature of the layout changing
rather than of the catalogue shrinking.
"""

from __future__ import annotations

import pytest

from core.community import OPENMODELDB_URL, fetch_openmodeldb

pytestmark = pytest.mark.network

#: A working page serves at least this many cards today. Anything below a
#: quarter of it is a layout change, not a smaller catalogue.
MINIMUM_CARDS = 3


def test_the_live_listing_still_parses() -> None:
    models = fetch_openmodeldb()
    assert len(models) >= MINIMUM_CARDS, (
        f"{OPENMODELDB_URL} yielded {len(models)} model cards, which is below the "
        f"{MINIMUM_CARDS} a working page serves. The selectors in core.community "
        "probably need updating; the committed snapshot in tests/fixtures is the "
        "old markup and cannot detect this."
    )
    for model in models:
        assert model.name
        assert model.scale >= 1
        assert model.page_url.startswith("https://")


def test_the_live_listing_names_do_not_contain_markup() -> None:
    for model in fetch_openmodeldb():
        assert "<" not in model.name, f"a title came back as markup: {model.name!r}"
        assert model.architecture == "" or "<" not in model.architecture
