# Upscaler

A cross-platform GUI video upscaler. It wraps a toolkit-agnostic upscaling
core in two native desktop frontends — **PySide6** (primary) and **Toga** —
ships installers for Ubuntu, Windows and macOS, and drives every GPU backend
that still exists for each platform.

It replaces the hard-coded script workflow: instead of editing
`WORK_DIR`, `MODEL_PATH`, `GPU_IDS`, `SCALE`, `FPS` and `OUTPUT_VIDEO` at the
top of a file and restarting a multi-hour job, every one of those is a field
in a form, and an interrupted run resumes where it stopped.

- One core (`core/`), two frontends (`ui_pyside/`, `ui_toga/`) — no logic in
  either UI: both build a `JobConfig` and drive `UpscalerApp`.
- A browsable catalogue of open-source upscaling models with licences and
  sizes, plus a community browser for [OpenModelDB](https://openmodeldb.info).
- Multi-GPU: frames are round-robined across every selected device, each in
  its own process, with CPU PNG I/O overlapped against GPU inference.
- **Audio is preserved.** The encode step maps the source audio stream; a
  upscaled video that is silently mute is not a result.
- Resumable. Frame counts, per-device progress and ETA survive a stop, a
  crash, and a restart.

## Status

Alpha. The core, both frontends and all three packaging paths are implemented;
**no release has been published**, because the first public release is gated on
running `scripts/verify_gpu.py` against real NVIDIA, AMD and Apple hardware.
See [Release gate](#release-gate).

## Backend matrix

Every row is either implemented or explicitly unavailable. Rows that are *not*
achievable today are listed as such rather than quietly dropped.

| OS | GPU vendor | Backend | Requirements / limits |
| --- | --- | --- | --- |
| Linux | NVIDIA | `CUDAExecutionProvider` | `onnxruntime-gpu==1.26.0` (CUDA 12.8 line) + `torch==2.7.1+cu126`; compute capability < 7.0 is forced to FP32 (FP16 is slower on Pascal) |
| Linux | AMD | `MIGraphXExecutionProvider` | **Requires a host ROCm 7.x install and Python 3.12.** `ROCmExecutionProvider` was removed in ONNX Runtime 1.23+; MIGraphX takes a plain `.onnx` and caches its compilation. Wheels are on `repo.radeon.com`, not PyPI. With no ROCm host the device is listed unusable: *"AMD GPU acceleration on Linux requires ROCm 7.x; see the Runtimes tab"* |
| Linux | Intel | `torch.xpu` | No ONNX Runtime Intel EP exists. The Intel Extension for PyTorch is archived and end-of-life, so it is never used |
| Linux | any / none | `CPUExecutionProvider` | Always available; `scale` inference and tiling keep it usable on large frames |
| Windows | NVIDIA, AMD, Intel | `DmlExecutionProvider` | `onnxruntime-directml==1.24.4` over DirectX 12. Memory-pattern and parallel-execution optimisations must be disabled (the app sets `enable_mem_pattern=False`, `ORT_SEQUENTIAL`); op set ceiling is opset 20, and the exporter targets opset 17 |
| Windows | any / none | `CPUExecutionProvider` | Fallback when no DirectX 12 device is present |
| macOS 12+ | Apple | `CoreMLExecutionProvider` | `ModelFormat: MLProgram`, which requires macOS 12 or newer. The supported-op table is the contract: the exporter asserts every emitted op against it, and **rewrites the graph when PyTorch emits something the table cannot run** — see below |
| macOS < 12 | Apple | `CoreMLExecutionProvider` (`NeuralNetwork` format) | One automatic retry, then the chain continues |
| macOS 14+, Apple silicon | Apple | torch `mps` | Only when CoreML is unavailable; PyTorch requires macOS 14.0+ and an MPS-capable device |
| macOS | any | `CPUExecutionProvider` | Final fallback |

The ONNX Runtime CUDA/DirectML/CoreML providers all **silently fall back to
CPU** when their runtime cannot load, so the app never trusts a provider list
it did not verify: `scripts/verify_gpu.py` compares every device's output
against a CPU reference and fails when the max absolute difference exceeds
`2e-3`.

### Two exporter facts, measured rather than assumed

`RealESRGAN_x4plus` is the most-supported model in the world, and this app
still has to fix its exported graph twice before Apple's provider will take it.
Both facts were measured on the pinned toolchain (torch 2.7.1, ONNX Runtime
1.22) rather than taken from documentation:

1. **PyTorch exports `pixel_shuffle` as `DepthToSpace(mode="CRD")`** — at every
   opset from 11 to 20, static spatial dims or not. The CoreML EP's MLProgram
   table supports "Only DCR mode DepthToSpace"; its NeuralNetwork column
   accepts CRD but only for a fixed input shape. So a CRD node either fails on
   the fast path or forces every macOS export onto the slower fallback. The
   exporter rewrites each CRD node into the equivalent
   `Reshape → Transpose(perm=[0,1,4,2,5,3]) → Reshape`, which is numerically
   identical — a test asserts `array_equal` against the unrewritten graph.
2. **A `Constant` node is not in the CoreML table either.** The real
   `RealESRGAN_x4plus` graph contains one, and the allow-list check refused the
   model's own export. `Constant` values are folded into initializers, which
   are not nodes at all and which every provider handles.

Exports are cached per checkpoint hash, frame size, opset and an
`EXPORT_REVISION` counter, so a change to the exporter cannot leave a stale
graph behind. A cached export at 288×384 of the 23-block x4plus model takes
minutes on CPU the first time and is instant afterwards.

## Installing from a release

| Platform | Artefact | Notes |
| --- | --- | --- |
| Ubuntu | `upscaler_<ver>_amd64-cpu.deb` | ~0.5 GB, CPU-only, works out of the box |
| Ubuntu | `upscaler_<ver>_amd64-cuda.deb` | ~0.1 GB; the Runtimes tab downloads the CUDA stack on demand, with exact sizes shown before you commit |
| Windows | `upscaler_<ver>_windows-amd64-setup.exe` | Inno Setup, DirectML bundled |
| macOS | `upscaler_<ver>_macos.dmg` | macOS 12.0 or newer, CoreML bundled |

An offline CUDA `.deb` (~3.5 GB) is produced locally by
`packaging/linux/build-cuda-deb.sh` and is **never uploaded**: GitHub rejects
any release asset of 2 GiB or more.

## Development

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[cpu,dev]"
```

Toga's Linux backend needs PyGObject, which is **source-distribution only** on
PyPI. On a machine without `libgirepository1.0-dev` (no root here, so this
repo's dev environment) install it without dependencies and let the system
provide `gi`:

```bash
uv pip install --python .venv/bin/python "toga==0.5.6" "toga-gtk==0.5.6" --no-deps
```

On Linux, `toga-gtk` is the packaged default because the Toga docs describe the
GTK 3 backend as the more mature of the two; the `.deb` declares
`Depends: python3-gi, gir1.2-gtk-3.0, python3-gi-cairo` and lets `apt` supply
it. `toga-qt` exists and needs no PyGObject, but is documented as an
early-stage backend, so it is not the default.

### Running

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m ui_pyside.main   # Qt frontend
.venv/bin/python -m ui_toga.main                               # Toga frontend (needs gi)
```

### Checks

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/mypy core/ ui_pyside/ ui_toga/
.venv/bin/pytest -m "not gpu"          # headless suite
.venv/bin/pytest -m gpu                # real accelerator only
.venv/bin/pytest -m network            # live OpenModelDB drift check
```

`mypy` runs over the frontends too — they are type-checked, not exempt. The
only excluded file is `ui_pyside/worker_bridge.py`, because PySide6's
`Signal`/`Slot` decorator typing is not strict-clean.

### Device probe

```bash
.venv/bin/python -c "from core.devices import probe_all; print(probe_all())"
```

The probe reports a GPU that is **present but whose driver does not answer**
as `DEGRADED` (usable, FP32 forced, VRAM and compute capability unknown)
rather than as healthy or absent. Reporting a wedged GPU as usable would start
a 55,000-frame-per-device job that can never finish.

## How a job runs

1. Validate the configuration and the model file. A 9-byte `Not Found`
   response saved as `.pth` is rejected before a single frame is touched.
2. Probe the input with `ffprobe`, resolving the frame count from
   `nb_read_frames` (`-count_frames`), then `nb_frames`, then
   `duration × r_frame_rate` — Matroska and many MP4s report `nb_frames=N/A`.
3. Disk preflight against a **calibrated** estimate (0.30 bytes per output
   pixel, 0.25 per input pixel), not the naive uncompressed-RGB figure which
   over-estimates by 12–27× and would reject a legitimate 165,303-frame job.
4. Extract frames into `work_dir/frames_in`, skipping when they are already there.
5. Resume: existing `frames_out/frame_%08d.png` files are skipped, and a work
   directory holding frames from a *different* video is refused rather than
   silently mixed.
6. Round-robin the remaining frames across the selected devices.
7. Each device runs in its own spawned process with a threaded reader/writer,
   so CPU PNG I/O overlaps GPU inference (measured at ~38% on the original
   script's workload).
8. Verify that no frame is missing **before** encoding: ffmpeg's `image2`
   demuxer stops at the first gap and would silently produce a short video.
9. Encode with audio, write to `<output>.part.mp4` in the output's own
   directory, then `os.replace` it, so a failed encode can never destroy the
   previous file.

## Release gate

`scripts/verify_gpu.py` is the local gate. It prints each device's vendor,
compute capability and backend, exports the real `RealESRGAN_x4plus.pth` at a
known size, runs it through the real ONNX session *and* the torch backend, and
compares both against a CPU reference, failing when the max absolute
difference exceeds `2e-3`. A synthetic `conv2d` is not sufficient: it passes
in exactly the situation where the ONNX Runtime CUDA EP silently falls back
to CPU while torch still works.

The first public release is published as a **prerelease** until a maintainer
has run that gate on real Windows and macOS hardware.

## Licence

MIT — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for third-party
components (PySide6 is LGPL-3.0, Toga is BSD-3). Model weights keep the
licences listed per entry in `models/catalogue.json`.
