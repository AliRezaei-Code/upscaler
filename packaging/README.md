# Packaging

How every artefact is built, on which platform it has to be built, what it
weighs, and the two rules that are not negotiable.

## The artefacts

| Artefact | Built by | Host | Built on | Expected size |
|---|---|---|---|---|
| `upscaler_<ver>_amd64-cpu.deb` | `packaging/linux/build-deb.sh --fat` | Linux x86-64 | Ubuntu 22.04+ | **486 MiB, measured** |
| `upscaler_<ver>_amd64-cuda.deb` | `packaging/linux/build-deb.sh --slim` | Linux x86-64 | Ubuntu 22.04+ | ~0.15 GiB, estimated |
| `upscaler_<ver>-cuda1_amd64-cuda-offline.deb` | `packaging/linux/build-cuda-deb.sh` | Linux x86-64 | Ubuntu 22.04+ | ~3.5 GiB, estimated. **Never published.** |
| `upscaler_<ver>_windows-amd64-setup.exe` | `packaging/windows/build-exe.ps1` | Windows x64 | windows-latest | ~0.35 GiB, estimated |
| `upscaler_<ver>_macos.dmg` | `packaging/macos/build-dmg.sh` | macOS | macos-13 | ~0.4 GiB, estimated |
| Toga packages | `briefcase package`, from `pyproject.toml` | see below | see below | — |

`<ver>` is the `version` in `pyproject.toml`. Every build reads it from there
through a TOML parser; none of them has a second copy to forget to bump.

**Cross-building is not attempted.** PyInstaller's Windows and macOS
bootloaders, `codesign`, and `create-dmg` are all platform-specific; a Linux
host that "successfully" produced a `.dmg` would have produced one nobody can
open. `build-dmg.sh` refuses to run anywhere but macOS and says why.

## The two rules

### 1. No release asset may be 2 GiB or more

GitHub's documented limit is 2 GiB (2147483648 bytes) per file attached to a
release. Two consequences:

* The published Linux `.deb`s are CPU-fat and slim. The offline CUDA build is
  not published — `build-cuda-deb.sh` prints the rule in capitals before it
  starts and checks the finished artefact's size against the limit afterwards.
* The Windows and macOS images fit. A future accelerator runtime added to them
  would not, and the check has to run again before the tag is pushed.

### 2. The Toga front-end needs a PyGObject built from source

`toga-gtk` 0.5.6 does `from gi.events import GLibEventLoop`, and `gi.events`
arrived in **PyGObject 3.50**. Ubuntu 22.04 ships 3.42.1 and Ubuntu 24.04 ships
3.48, so **neither has it** and a Toga Linux package built against apt's
`python3-gi` cannot start. The package must carry a PyGObject built from source,
which needs `libgirepository-2.0-dev` and `libcairo2-dev` at build time, and must
declare the GTK **runtime** libraries:

```
Depends: libgtk-3-0, libglib2.0-0, gir1.2-gtk-3.0, python3-gi-cairo
```

— that is, `gir1.2-gtk-3.0` and the GLib runtime, **not** `python3-gi`. The
PyGObject module itself is bundled.

## Briefcase, and the system Python

The PySide6 and Toga front-ends are both packaged, by different tools:
PyInstaller for PySide6 (whose widget toolkit freezes cleanly) and Briefcase for
Toga (whose toolkit backends do not).

Briefcase's Linux builder uses the **system** `python3`, and this project
requires `>= 3.11`. That is a hard failure, not a warning:

```
The version of Python being used to run Briefcase ('3.12.12') is not the system python3 ('3.10.12').
```

| Distribution | System Python | Toga `.deb` |
|---|---|---|
| Ubuntu 22.04 | 3.10.12 | **Not buildable** as configured |
| Ubuntu 24.04 | 3.12 | Buildable, once PyGObject ≥ 3.50 is supplied |
| Debian 12 | 3.11 | Buildable, once PyGObject ≥ 3.50 is supplied |
| Fedora 39+ | 3.12 | Buildable, once PyGObject ≥ 3.50 is supplied |
| macOS | 3.9–3.12 depending on the release | Unaffected: Briefcase bundles its own Python on macOS |
| Windows | n/a | Unaffected: the same |

**macOS is not affected by any of this.** Briefcase's macOS backend installs a
standalone Python rather than using the system one, and the two Linux
constraints — the system Python's version and the GTK backend's PyGObject
requirement — are both Linux-only.

Until PyGObject ≥ 3.50 is bundled, the Toga front-end is releasable on macOS
and Windows and not on Linux. The PySide6 `.deb`s above are unaffected: they
never import `gi`.

## Building the Linux packages

```sh
# fat: bundles the ONNX Runtime CPU provider and the CPU build of PyTorch
packaging/linux/build-deb.sh --fat

# slim: bundles no inference engine; the Runtimes tab installs one
packaging/linux/build-deb.sh --slim

# offline CUDA, never published
packaging/linux/build-cuda-deb.sh
```

Both write to `dist/` and print the artefact's path and size as their last two
lines on stdout. Progress goes to stderr, so `dist=$(packaging/linux/build-deb.sh --fat | head -1 | cut -d' ' -f2)`
works.

`--dry-run` prints every command without running any of them, which is the
cheapest way to see what a build would do:

```sh
packaging/linux/build-deb.sh --fat --dry-run
```

`--skip-install` reuses the dependency tree already under `build/linux-<flavour>/`,
which is worth about four minutes of downloads.

### Why the dependencies are installed into their own directory

Each build creates `build/linux-<flavour>/site` and installs the base
dependencies (minus `toga*`) plus the flavour's extra into it with
`pip install --target`, then freezes from there. Two reasons:

* PyPI's `torch==2.7.1` on Linux x86-64 is the **CUDA** wheel plus the whole
  `nvidia-*` stack. A build that took it literally would be several gigabytes of
  files labelled "cpu". The build adds `https://download.pytorch.org/whl/cpu` as
  an extra index — a local version such as `2.7.1+cpu` sorts above the public
  `2.7.1`, so pip takes it — and then *asserts* that no `nvidia_*` package
  arrived before PyInstaller is allowed to run.
* `toga*` is dropped because `toga-gtk` requires `pygobject>=3.55.1`, a
  source-only distribution needing root and the GObject introspection headers.
  Installing it would fail the build on every Linux machine, to ship bytes the
  Qt front-end never loads.

### What is in the package

```
/usr/bin/upscaler                                   the launcher
/usr/lib/upscaler/upscaler                          the frozen executable
/usr/lib/upscaler/_internal/…                       the bundle's libraries and data
/usr/lib/upscaler/bin/ffmpeg                        the ffmpeg the app encodes with
/usr/share/applications/upscaler.desktop            the desktop entry
/usr/share/icons/hicolor/512x512/apps/upscaler.png  the icon
```

The launcher derives its own prefix from its location rather than hard-coding
`/usr`, so a tree unpacked with `dpkg-deb -x` anywhere is runnable. It also
prepends the bundle to `LD_LIBRARY_PATH` — a system `libstdc++` older than the
one the bundle was built against loads first without it, and the executable
dies with a bare "symbol lookup error" that names nothing.

Both flavours bundle an ffmpeg, because the application cannot encode without
one. Only the slim build declares `Depends: ffmpeg`; the fat build relies on its
own copy, and without apt's `ffprobe` `core.ffmpeg.probe()` takes its documented
ffmpeg-only fallback.

## Installing a `.deb` without root

```sh
dpkg-deb -x dist/upscaler_0.1.0_amd64-cpu.deb /tmp/upscaler-root
/tmp/upscaler-root/usr/bin/upscaler
```

`dpkg -i` needs root; `dpkg-deb -x` does not, and the launcher works from any
prefix. With root:

```sh
sudo dpkg -i dist/upscaler_0.1.0_amd64-cpu.deb
```

The desktop entry and the icon are installed, so the app appears in the launcher
and the app grid after a normal install.

## Building the Windows installer

```powershell
packaging\windows\build-exe.ps1
packaging\windows\build-exe.ps1 -Version 0.1.0 -DryRun
packaging\windows\build-exe.ps1 -InnoPath "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
```

Needs Python 3.11+ on `PATH` and Inno Setup's `iscc.exe`, which the script
looks for on `PATH`, then under `%ProgramFiles(x86)%\Inno Setup 6\`, and names
in the failure message if it finds neither. It installs nothing itself.

The bundle is the `dml` extra — `onnxruntime-directml`, the one execution
provider that reaches NVIDIA, AMD and Intel alike on Windows over DirectX 12 —
and the build **fails** if the resulting ONNX Runtime has no
`DmlExecutionProvider` in it. `core/backends/onnx_backend.py` already sets
`enable_mem_pattern=False` and `ORT_SEQUENTIAL` for it, because DirectML refuses
the defaults.

> **Consequence, stated plainly:** the Windows installer carries no PyTorch,
> because the `dml` extra does not include it and `onnxruntime` and
> `onnxruntime-directml` cannot both be installed. `.onnx` models run on
> DirectML; a `.pth` checkpoint does not, and the user installs the CPU runtime
> from the Runtimes tab to get one. If that trade is wrong for a release, the
> fix is to add `torch` to the requirement list `Get-Requirements` produces —
> torch does not conflict with DirectML, so it can be added alongside it — at a
> cost of roughly 200 MB.

The `.iss` is generated from the build parameters, so the output name and the
version have exactly one home. `LICENSE` travels with the binary because Qt is
LGPLv3.

## Building the macOS image

```sh
packaging/macos/build-dmg.sh
packaging/macos/build-dmg.sh --dry-run     # from a macOS host
```

Needs `create-dmg` on `PATH` (`brew install create-dmg`) and the Xcode command
line tools for `PlistBuddy`. The script stops on a non-Darwin host before it has
done anything.

It bundles `onnxruntime` — whose stock macOS wheel already carries the
`CoreMLExecutionProvider` — and asserts that provider is present, plus `torch`
for the `mps` fallback. `LSMinimumSystemVersion` is set to **12.0** in the
bundle's `Info.plist`, because `MLProgram`, the CoreML model format the backend
asks for, is refused below it, and the script reads the value back and fails if
the edit did not take.

This script does not sign or notarise. A build that will be distributed has to
be signed with a Developer ID and notarised, or Gatekeeper refuses it on any
machine but the one that built it.

## The smoke test

`scripts/verify_install.py` is what every packaging job runs against the
artefact it just built, and what a maintainer runs before publishing. It needs
no display and no GPU, and it never touches the real data directory.

```sh
# against a .deb unpacked anywhere
.venv/bin/python scripts/verify_install.py --app-dir /tmp/verify-root

# against a PyInstaller --onedir bundle, or a dist/ directory holding one
.venv/bin/python scripts/verify_install.py --app-dir dist/upscaler
.venv/bin/python scripts/verify_install.py --app-dir dist

# against the built executable
.venv/bin/python scripts/verify_install.py --executable dist/upscaler/upscaler

# a one-frame job instead of the ten-frame default
.venv/bin/python scripts/verify_install.py --app-dir /tmp/verify-root --frames 1

# on a machine whose GPUs are known to be down
.venv/bin/python scripts/verify_install.py --app-dir /tmp/verify-root \
    --allow-unreachable-devices
```

It runs three checks and prints a line per observation:

1. **The bundle launches, and opens exactly one window.** The frozen executable
   is started with `QT_QPA_PLATFORM=offscreen` against a throwaway `$HOME` and
   is expected to still be running when the observation window closes. The
   window count comes from the application's own log: `RuntimesTab` writes one
   `resolved onnxruntime:` line per construction, and a tab is only ever built by
   a `MainWindow`, which only exists once a `QApplication` has — so a second line
   *is* a second window, which is the failure `multiprocessing.freeze_support()`
   prevents. On Linux the count is corroborated by reading `/proc` for further
   processes running the same executable, and by `xdotool` when there is both a
   window server and that tool. Anything not measured is reported as not
   measured.
2. **A real job runs on every device the probe can see** — extract, upscale, the
   completeness check, encode — on a throwaway work directory, once per device,
   each in its own process so a wedged accelerator can be killed rather than
   waited on. A clip with audio is used on purpose: a silent output is the bug
   this project exists to fix, and a job whose source has audio and whose output
   does not **fails**.
3. **The artefact's own execution providers are reported.** The GPU providers
   are separate shared objects under `onnxruntime/capi/`, so whether one is in
   the bundle is what says whether an accelerator can work at all.

Exit codes: **0** every check passed, **1** a check failed with the reason
named on stderr, **2** the command line was wrong.

A device the app lists as usable but that this build cannot reach is a
failure by default — the fat CPU build on a machine with an NVIDIA card is
*correct* and cannot reach the GPU, and that is worth saying out loud rather
than hiding. `--allow-unreachable-devices` downgrades it to a report; it is for
a machine whose GPUs are known to be down, not for making a broken bundle pass.

The source clip and the model are both optional. Without `--source` a real clip
is built with ffmpeg (`testsrc2` plus a sine tone). Without `--model` a real 2x
ONNX graph is written and the run says so: it proves the engine runs, not that
it sharpens anything.

## What was measured, and what was not

Measured on this machine, Ubuntu 22.04, 3 × Tesla P40 (all currently wedged),
Python 3.12.12, PyInstaller 6.22.3:

* `upscaler_0.1.0_amd64-cpu.deb` builds and is **486 MiB (509,503,900 bytes)**,
  with an `Installed-Size` of 1,616,292 KiB.
* Unpacked with `dpkg-deb -x` and launched, the frozen application stays up, and
  its startup log resolves `onnxruntime` and `torch` to files **inside the
  bundle**, not to the build machine's site-packages.
* The fat build ships `libonnxruntime_providers_shared.so` and no accelerator
  provider, which is the correct outcome for a CPU build and the reason the
  Runtimes tab exists.
* A one-frame job on the CPU device produces a 640×480 output with an audio
  stream, from a 320×240 one-frame source.

Not measured here, because the host is Linux: the Windows installer, the macOS
image, the slim `.deb` and the offline CUDA `.deb`. Their sizes in the table
above are estimates, and the two 2 GiB checks in `build-deb.sh`,
`build-cuda-deb.sh` and `build-exe.ps1` are what enforce the rule when they are
built for real.
