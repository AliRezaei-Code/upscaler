# Release checklist

`.github/workflows/release.yml` assembles the release body from this file. One
line is read out of it by name — the `- Manual GPU gate:` line under "Manual
gates" — and copied verbatim into the release, so a release cannot claim a gate
that was not run: the line either says it was run, and says on what, or the
release job fails.

Everything else here is for the person cutting the release.

## Automatic gates, on every tag

- `ci.yml` — `ruff check`, `ruff format --check`, `mypy --strict` and the test
  matrix (Linux, Windows, macOS × Python 3.11, 3.12). `release.yml` repeats all
  of it on the tag, because `needs` cannot cross a workflow boundary and a
  release must not be publishable from a commit that never type-checked.
- Every artefact is unpacked and driven by `scripts/verify_install.py` before it
  is uploaded. The exit code per target is copied into the release body.
- Every asset is checked against GitHub's 2 GiB per-asset limit before the
  release exists, so an over-sized artefact fails the run instead of the upload.

## Manual gates

- Manual GPU gate: **not run** — no NVIDIA, AMD or Intel accelerator and no Windows or macOS machine was available. This is the line `release.yml` copies into the release body; replace it with what was actually run, and with the hardware, e.g. `**run** on an RTX 4070 (Linux, sm_89) and a Radeon 7800 (Linux, ROCm 7.2), plus Windows 11 and macOS 14 machines; all four backends agreed with the CPU reference within 2e-3`.

### What "the manual GPU gate" means

`scripts/verify_gpu.py` is the gate, and it is the only thing in this repository
that can tell a working GPU from a silent fallback. ONNX Runtime accepts a
provider it cannot create, runs the graph on the CPU, and returns a correct
answer; the CUDA execution provider falls back to the CPU whenever
`libcublas`/`libcudnn` cannot be loaded, which is every time the CUDA stack is
missing, mismatched or one minor version away. A synthetic `conv2d` passes in
exactly that situation, because it also runs correctly — on the CPU. So the
gate runs the real exported `RealESRGAN_x4plus` graph on every device, compares
it with a CPU reference of the same graph, and fails when the session is not
using the provider it asked for.

Before a release leaves prerelease, all of this has to have happened on real
hardware:

1. A machine with each accelerator backend this release claims, with the
   matching runtime installed through the Runtimes tab. At minimum one of
   NVIDIA, AMD and Intel on Linux.
2. A real Windows machine, for `DmlExecutionProvider`. It is the only provider
   that covers NVIDIA, AMD and Intel at once on Windows, and it has two settings
   the rest of the code does not need: `enable_mem_pattern = False` and
   `ORT_SEQUENTIAL`. Nothing on a CI runner can exercise it.
3. A real macOS machine, for `CoreMLExecutionProvider` (macOS 12+) and the
   `mps` fallback (macOS 14+, Apple silicon). `macos-13` in CI is x86_64 and
   has no CoreML provider in the runner image, so it proves nothing here.
4. A real end-to-end job per platform: a real clip in, a real output out, with
   the audio stream intact and the output dimensions four times the input.

Each of those is a manual action. There is no runner that can do any of them,
and a green CI run is not a substitute for any of them.

## Not releasable yet

- **The Toga front-end.** It is packaged by Briefcase from
  `[tool.briefcase]`, not by `release.yml`, and it is not in the release matrix:
  the Linux `.deb` needs `python3-gi` and `gir1.2-gtk-3.0` from apt, and the
  Windows build needs PyGObject, which ships no wheel and is a source
  distribution. The PySide6 front-end is the supported one until those are
  solved. Adding a Briefcase target to the matrix is a separate piece of work,
  not a flag on this one.
- **The offline CUDA `.deb`** (`packaging/linux/build-cuda-deb.sh`). It is a
  local convenience build and is never uploaded: at roughly 3.5 GB it is over
  GitHub's 2 GiB per-asset limit, and the release job would refuse it.

## Cutting a release

1. Update `version` in `pyproject.toml`. Every artefact name and every
   `Depends:` line comes from it; nothing hard-codes a version.
2. Run the full suite and the GPU gate locally: `python -m pytest`, then
   `python scripts/verify_gpu.py`.
3. Commit the version bump, open a pull request, let the `ci` check go green.
4. Update the `- Manual GPU gate:` line above if anything has been verified
   since the last release, and commit that.
5. Tag and push: `git tag v0.1.0 && git push origin v0.1.0`. The push is what
   triggers both workflows; nothing is published by hand.
6. Check the published release: it must be a prerelease until the manual gate
   above has been run on real Windows and real macOS hardware, and the body
   must show a verification row for all four targets.
