from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from core import models as models_module
from core.errors import ModelFileTooSmall, ModelLoadError
from core.models import (
    MIN_MODEL_BYTES,
    ModelEntry,
    cached_hash,
    entry_for,
    hash_file,
    load_catalogue,
    load_hash_cache,
    store_hash,
    verify_model,
)
from core.paths import catalogue_path

SHA256_RE = re.compile(r"[0-9a-f]{64}")
# The 9-byte `Not Found` file this app exists partly to catch, if it is still there.
REAL_TRUNCATED_MODEL = Path(
    "/media/ali0rez/ext4-Linux/mosaferan-mahtab/RealESRGAN_x4plus_anime_6B.pth"
)


@pytest.fixture(autouse=True)
def data_dir_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep `models/.hashes.json` out of the real data directory."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    models_module._catalogue_cache = None
    yield
    models_module._catalogue_cache = None


# --- the catalogue itself -----------------------------------------------------


def test_catalogue_ships_with_the_app() -> None:
    assert catalogue_path().is_file()


def test_catalogue_has_the_expected_entries() -> None:
    entries = load_catalogue()
    assert len(entries) >= 14
    assert {entry.id for entry in entries} >= {
        "realesrgan-x4plus",
        "realesrgan-x2plus",
        "realesrnet-x4plus",
        "esrgan-srx4-df2kost-official",
        "realesr-general-x4v3",
        "realesr-general-wdn-x4v3",
        "realesr-animevideov3",
        "realesrgan-x4plus-anime-6b",
        "realesrgan-x4plus-anime-6b-netd",
        "realesrganv2-animevideo-x4",
        "realesrganv2-animevideo-x2",
        "swinir-lite-x4",
        "swinir-realsr-x4-gan",
        "aurasr-v2",
    }


def test_catalogue_ids_are_unique() -> None:
    entries = load_catalogue()
    assert len({entry.id for entry in entries}) == len(entries)


def test_catalogue_urls_are_https() -> None:
    for entry in load_catalogue():
        assert entry.url.startswith("https://"), entry.id
        assert entry.filename, entry.id


def test_catalogue_filenames_match_their_urls() -> None:
    for entry in load_catalogue():
        assert entry.url.endswith(entry.filename) or "huggingface.co" in entry.url, (
            entry.id
        )


def test_catalogue_hashes_are_hex_or_absent() -> None:
    for entry in load_catalogue():
        if entry.sha256 is not None:
            assert SHA256_RE.fullmatch(entry.sha256), entry.id


def test_catalogue_sizes_are_plausible() -> None:
    for entry in load_catalogue():
        assert entry.size_bytes > MIN_MODEL_BYTES, entry.id


def test_catalogue_declares_a_licence_and_a_source() -> None:
    for entry in load_catalogue():
        assert entry.license.strip(), entry.id
        assert entry.source.startswith("https://"), entry.id


def test_catalogue_scales_are_the_ones_the_models_actually_do() -> None:
    scales = {entry.id: entry.scale for entry in load_catalogue()}
    assert scales["realesrgan-x2plus"] == 2
    assert scales["realesrganv2-animevideo-x2"] == 2
    assert scales["realesr-general-x4v3"] == 4


def test_every_architecture_is_one_spandrel_knows() -> None:
    import spandrel

    known = {a.architecture.id for a in spandrel.MAIN_REGISTRY.architectures()}
    for entry in load_catalogue():
        assert entry.architecture in known, f"{entry.id}: {entry.architecture}"


# --- schema failures ----------------------------------------------------------


def _raw() -> dict[str, object]:
    return json.loads(catalogue_path().read_text(encoding="utf-8"))


def test_catalogue_rejects_a_missing_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _raw()
    del raw["models"][0]["license"]  # type: ignore[index]
    broken = tmp_path / "catalogue.json"
    broken.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(models_module, "catalogue_path", lambda: broken)
    with pytest.raises(ModelLoadError, match="missing license"):
        load_catalogue(refresh=True)


def test_catalogue_rejects_an_unknown_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _raw()
    raw["models"][0]["download_mirror"] = "https://example.invalid"  # type: ignore[index]
    broken = tmp_path / "catalogue.json"
    broken.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(models_module, "catalogue_path", lambda: broken)
    with pytest.raises(ModelLoadError, match="unknown field"):
        load_catalogue(refresh=True)


def test_catalogue_rejects_a_duplicate_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _raw()
    raw["models"].append(dict(raw["models"][0]))  # type: ignore[attr-defined,index]
    broken = tmp_path / "catalogue.json"
    broken.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(models_module, "catalogue_path", lambda: broken)
    with pytest.raises(ModelLoadError, match="duplicate id"):
        load_catalogue(refresh=True)


def test_catalogue_rejects_a_non_https_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _raw()
    raw["models"][0]["url"] = "http://example.invalid/model.pth"  # type: ignore[index]
    broken = tmp_path / "catalogue.json"
    broken.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(models_module, "catalogue_path", lambda: broken)
    with pytest.raises(ModelLoadError, match="non-https"):
        load_catalogue(refresh=True)


def test_catalogue_rejects_broken_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = tmp_path / "catalogue.json"
    broken.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(models_module, "catalogue_path", lambda: broken)
    with pytest.raises(ModelLoadError, match="not valid JSON"):
        load_catalogue(refresh=True)


def test_catalogue_reports_a_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        models_module, "catalogue_path", lambda: tmp_path / "absent.json"
    )
    with pytest.raises(ModelLoadError, match="not found"):
        load_catalogue(refresh=True)


def test_entry_from_dict_accepts_every_shipped_entry() -> None:
    for index, raw in enumerate(_raw()["models"], start=1):  # type: ignore[union-attr]
        assert isinstance(ModelEntry.from_dict(raw, index), ModelEntry)


# --- entry_for ----------------------------------------------------------------


def test_entry_for_matches_a_catalogue_model(tmp_path: Path) -> None:
    entry = entry_for(tmp_path / "realesr-general-x4v3.pth")
    assert entry is not None
    assert entry.id == "realesr-general-x4v3"
    assert entry.scale == 4


def test_entry_for_is_case_insensitive(tmp_path: Path) -> None:
    assert entry_for(tmp_path / "REALESR-GENERAL-X4V3.PTH") is not None


def test_entry_for_returns_none_for_an_unknown_model(tmp_path: Path) -> None:
    assert entry_for(tmp_path / "my_own_model.pth") is None


# --- verify_model -------------------------------------------------------------


def test_a_nine_byte_download_is_rejected(tmp_path: Path) -> None:
    # The exact artefact on the machine this was built on: the literal nine
    # bytes of `Not Found`, saved under a real model's filename.
    path = tmp_path / "RealESRGAN_x4plus_anime_6B.pth"
    path.write_bytes(b"Not Found")
    with pytest.raises(ModelFileTooSmall) as excinfo:
        verify_model(path, entry_for(path))
    assert str(excinfo.value) == (
        "RealESRGAN_x4plus_anime_6B.pth is only 9 bytes — this is a failed "
        "download, not a model. Delete it and download again."
    )


def test_the_real_truncated_model_is_rejected() -> None:
    if not REAL_TRUNCATED_MODEL.is_file():
        pytest.skip(f"{REAL_TRUNCATED_MODEL} is not on this machine")
    assert REAL_TRUNCATED_MODEL.stat().st_size < MIN_MODEL_BYTES
    with pytest.raises(ModelFileTooSmall):
        verify_model(REAL_TRUNCATED_MODEL, entry_for(REAL_TRUNCATED_MODEL))


def test_a_missing_model_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError, match="not found"):
        verify_model(tmp_path / "absent.pth", None)


def test_a_wrong_hash_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "realesr-general-x4v3.pth"
    path.write_bytes(b"x" * MIN_MODEL_BYTES)
    with pytest.raises(ModelLoadError, match="does not match the catalogue"):
        verify_model(path, entry_for(path))


def test_the_right_hash_passes(tmp_path: Path) -> None:
    entry = next(e for e in load_catalogue() if e.id == "realesr-general-x4v3")
    path = tmp_path / "realesr-general-x4v3.pth"
    path.write_bytes(b"y" * MIN_MODEL_BYTES)
    import hashlib

    patched = ModelEntry(
        **{**entry.__dict__, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    )
    verify_model(path, patched)


def test_a_model_with_no_entry_is_only_size_checked(tmp_path: Path) -> None:
    path = tmp_path / "somebody-elses-model.pth"
    path.write_bytes(b"z" * MIN_MODEL_BYTES)
    verify_model(path, entry_for(path))


# --- the hash cache -----------------------------------------------------------


def test_hash_is_cached_and_reused(tmp_path: Path) -> None:
    path = tmp_path / "realesr-general-x4v3.pth"
    path.write_bytes(b"q" * MIN_MODEL_BYTES)
    digest = hash_file(path)
    assert cached_hash(path) is None
    store_hash(path, digest)
    assert cached_hash(path) == digest
    assert load_hash_cache()[str(path)]["sha256"] == digest


def test_a_changed_file_invalidates_the_cached_hash(tmp_path: Path) -> None:
    path = tmp_path / "realesr-general-x4v3.pth"
    path.write_bytes(b"q" * MIN_MODEL_BYTES)
    store_hash(path, hash_file(path))
    path.write_bytes(b"r" * (MIN_MODEL_BYTES + 1))
    assert cached_hash(path) is None


def test_an_unreadable_hash_cache_is_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = models_module.models_dir() / ".hashes.json"
    cache.write_text("{{{ not json", encoding="utf-8")
    assert load_hash_cache() == {}


def test_hash_file_reads_in_chunks(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    payload = bytes(range(256)) * 10_000
    path.write_bytes(payload)
    import hashlib

    assert hash_file(path, chunk_size=1024) == hashlib.sha256(payload).hexdigest()
