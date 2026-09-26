"""The model catalogue and the check that a model file is really a model.

`verify_model` exists because a failed download is indistinguishable from a
model by name: `RealESRGAN_x4plus_anime_6B.pth` on the machine this was built
on is nine bytes long and contains the literal string `Not Found`. Handing
that to spandrel produces a traceback about tensors, not about the download.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .errors import ModelFileTooSmall, ModelLoadError
from .paths import catalogue_path, models_dir

# Below this, a `.pth`/`.safetensors`/`.onnx` cannot be a super-resolution
# checkpoint: the smallest model in the catalogue is 2.4 MB, and every real one
# is over 1 MB by a wide margin.
MIN_MODEL_BYTES = 1_048_576

_REQUIRED_FIELDS = (
    "id",
    "name",
    "architecture",
    "scale",
    "url",
    "filename",
    "size_bytes",
    "sha256",
    "license",
    "source",
    "notes",
    "tags",
)


@dataclass(frozen=True)
class ModelEntry:
    """One catalogue entry, as the Models tab shows it."""

    id: str
    name: str
    architecture: str
    scale: int
    url: str
    filename: str
    size_bytes: int
    sha256: str | None
    license: str
    source: str
    notes: str
    tags: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, object], index: int) -> ModelEntry:
        """Build an entry, refusing anything the schema does not guarantee.

        A catalogue with a missing field is a packaging accident, and the app
        must say so at load time rather than raise `KeyError` later inside a
        download callback.
        """
        missing = [field for field in _REQUIRED_FIELDS if field not in raw]
        if missing:
            raise ModelLoadError(
                f"models/catalogue.json entry {index} is missing {', '.join(missing)}"
            )
        unknown = sorted(set(raw) - set(_REQUIRED_FIELDS) - {"$comment"})
        if unknown:
            raise ModelLoadError(
                f"models/catalogue.json entry {index} has unknown field(s): "
                f"{', '.join(unknown)}"
            )
        url = str(raw["url"])
        if not url.startswith("https://"):
            raise ModelLoadError(
                f"models/catalogue.json entry {raw['id']} has a non-https url: {url}"
            )
        digest = raw["sha256"]
        if digest is not None and not isinstance(digest, str):
            raise ModelLoadError(
                f"models/catalogue.json entry {raw['id']} has a non-string sha256"
            )
        scale = raw["scale"]
        if not isinstance(scale, int) or scale < 1:
            raise ModelLoadError(
                f"models/catalogue.json entry {raw['id']} has a bad scale: {scale!r}"
            )
        size = raw["size_bytes"]
        if not isinstance(size, int) or size < 0:
            raise ModelLoadError(
                f"models/catalogue.json entry {raw['id']} has a bad "
                f"size_bytes: {size!r}"
            )
        tags = raw["tags"]
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ModelLoadError(
                f"models/catalogue.json entry {raw['id']} has bad tags: {tags!r}"
            )
        return cls(
            id=str(raw["id"]),
            name=str(raw["name"]),
            architecture=str(raw["architecture"]),
            scale=scale,
            url=url,
            filename=str(raw["filename"]),
            size_bytes=size,
            sha256=digest,
            license=str(raw["license"]),
            source=str(raw["source"]),
            notes=str(raw["notes"]),
            tags=tuple(tags),
        )


_catalogue_cache: list[ModelEntry] | None = None


def load_catalogue(*, refresh: bool = False) -> list[ModelEntry]:
    """Read `models/catalogue.json`, caching the parse.

    Pass `refresh=True` after downloading a new catalogue. Ids are unique and
    enforced: two entries sharing an id would make the Models tab's selection
    ambiguous.
    """
    global _catalogue_cache
    if _catalogue_cache is not None and not refresh:
        return _catalogue_cache
    path = catalogue_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelLoadError(f"Model catalogue not found at {path}") from exc
    except json.JSONDecodeError as exc:
        raise ModelLoadError(
            f"Model catalogue at {path} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("models"), list):
        raise ModelLoadError(f"Model catalogue at {path} has no 'models' list")
    entries = [
        ModelEntry.from_dict(entry, index)
        for index, entry in enumerate(raw["models"], start=1)
        if isinstance(entry, dict)
    ]
    seen: set[str] = set()
    for entry in entries:
        if entry.id in seen:
            raise ModelLoadError(f"Model catalogue has a duplicate id: {entry.id}")
        seen.add(entry.id)
    _catalogue_cache = entries
    return entries


def entry_for(model_path: Path) -> ModelEntry | None:
    """The catalogue entry for a file, matched by filename.

    `None` for a model picked through "Browse local…": it has no catalogue
    entry, and inventing one would attach a licence and a hash that belong to
    a different file.
    """
    target = model_path.name.lower()
    for entry in load_catalogue():
        if entry.filename.lower() == target:
            return entry
    return None


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """SHA-256 of a file, read in chunks so a 2.4 GB model does not sit in RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_cache_path() -> Path:
    return models_dir() / ".hashes.json"


def load_hash_cache() -> dict[str, dict[str, object]]:
    """The on-disk hash cache; an unreadable or corrupt cache is simply empty."""
    try:
        raw = json.loads(_hash_cache_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {key: value for key, value in raw.items() if isinstance(value, dict)}


def store_hash(path: Path, digest: str) -> None:
    """Remember a computed hash, keyed by size and mtime.

    Keying on both means a re-downloaded file never reuses a stale digest.
    """
    stat = path.stat()
    cache = load_hash_cache()
    cache[str(path)] = {
        "sha256": digest,
        "size": stat.st_size,
        "mtime": int(stat.st_mtime),
    }
    target = _hash_cache_path()
    target.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def cached_hash(path: Path) -> str | None:
    """The remembered hash for this exact file, if the cache has one."""
    record = load_hash_cache().get(str(path))
    if record is None:
        return None
    stat = path.stat()
    if record.get("size") != stat.st_size or record.get("mtime") != int(stat.st_mtime):
        return None
    digest = record.get("sha256")
    return digest if isinstance(digest, str) else None


def verify_model(path: Path, entry: ModelEntry | None) -> None:
    """Raise unless `path` is a model file the app can load.

    Three failures, in the order a user meets them: the file is not there, the
    file is a failed download, the file is not the one the catalogue describes.
    """
    if not path.is_file():
        raise ModelLoadError(f"Model file not found: {path}")
    size = path.stat().st_size
    if size < MIN_MODEL_BYTES:
        raise ModelFileTooSmall(
            f"{path.name} is only {size} bytes — this is a failed download, "
            "not a model. Delete it and download again."
        )
    if entry is None or entry.sha256 is None:
        return
    known = cached_hash(path)
    digest = known or hash_file(path)
    if known is None:
        store_hash(path, digest)
    if digest != entry.sha256:
        raise ModelLoadError(
            f"{path.name} does not match the catalogue: expected sha256 "
            f"{entry.sha256}, got {digest}"
        )
