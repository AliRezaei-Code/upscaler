"""The Briefcase entry point for the Toga front-end.

Briefcase requires the app's `sources` to include a package named after the
app, so `upscaler` is that package. It is a name and nothing else: the Toga
front-end is `ui_toga`, the logic is `core`, and everything here is a
re-export. The indirection exists for the packaging tool, not for the design —
which is why it is three lines and why nothing imports *from* it except
Briefcase.
"""

from __future__ import annotations

from ui_toga.app import Upscaler, main

__all__ = ["Upscaler", "main"]
