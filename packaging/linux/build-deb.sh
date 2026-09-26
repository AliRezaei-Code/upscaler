#!/usr/bin/env bash
#
# Build the Debian package for the PySide6 front-end.
#
#   packaging/linux/build-deb.sh --fat     CPU runtime bundled, ~0.5 GiB
#   packaging/linux/build-deb.sh --slim    no runtime bundled,  ~0.1 GiB
#
# Progress goes to stderr and the artefact's path and size go to stdout, so a
# caller can read the path out of a build without scraping the log.
#
# The two builds differ in exactly one thing — which wheels are in the payload —
# so everything else (the PyInstaller invocation, the launcher, the control
# file) is written once here and reached from build-cuda-deb.sh by exporting
# UPSCALER_FLAVOUR rather than by keeping a second copy of the build.
#
# Why the dependency tree is installed into a throwaway site directory instead
# of being frozen from the repository's own .venv: the flavour has to be exact.
# A "fat" build frozen from a venv that happens to have a CUDA torch in it is a
# fat build nobody asked for, and PyPI's `torch==2.7.1` on Linux x86-64 resolves
# to the *CUDA* wheel plus the whole nvidia-* stack — a multi-gigabyte package
# for something labelled "cpu". So the dependency set is read from
# pyproject.toml and installed here with the PyTorch CPU index added, and the
# result is checked for nvidia-* packages before PyInstaller is allowed to run.
#
# Why `toga*` is dropped from that dependency set: the frozen executable is the
# Qt front-end and never imports it, and `toga-gtk` requires
# `pygobject>=3.55.1`, a source-only distribution that needs root and the
# GObject introspection headers to build. Installing it would fail this build on
# every Linux machine, to ship bytes the app never loads. The Toga front-end is
# packaged by Briefcase from pyproject.toml instead — see packaging/README.md.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "$(dirname -- "$SCRIPT_DIR")/.." && pwd -P)

SCRIPT_NAME=$(basename -- "${BASH_SOURCE[0]}")

# --- what the build produces -------------------------------------------------

readonly PACKAGE_NAME="upscaler"
readonly PAYLOAD_NAME="upscaler"           # /usr/lib/upscaler
readonly LAUNCHER_NAME="upscaler"          # /usr/bin/upscaler
readonly DESKTOP_ID="upscaler"
readonly ICON_SIZE=512                     # matches packaging/icon.png
readonly MAINTAINER="AliRezaei-Code <https://github.com/AliRezaei-Code/upscaler>"
readonly HOMEPAGE="https://github.com/AliRezaei-Code/upscaler"

#: The PyTorch wheel index each flavour's torch has to come from. A local
#: version (`2.7.1+cpu`) sorts *above* the public `2.7.1`, so pip prefers it
#: when both indexes are offered — which is what makes "install the cpu extra"
#: mean "the CPU-only build" on a machine with a CUDA mirror configured.
declare -A TORCH_INDEX=(
    [fat]="https://download.pytorch.org/whl/cpu"
    [slim]="https://download.pytorch.org/whl/cpu"
    [cuda]="https://download.pytorch.org/whl/cu126"
)

#: Which `pyproject.toml` extra each flavour installs. `slim` has none on
#: purpose: the whole point is that it ships no inference engine, and the extra
#: it would otherwise take is the one that carries torch.
declare -A FLAVOUR_EXTRA=(
    [fat]="cpu"
    [slim]=""
    [cuda]="cuda"
)

#: Pinned so two builds of one commit produce the same bundle, and so a
#: PyInstaller major bump is a visible change in this file.
readonly PYINSTALLER_PIN="pyinstaller==6.22.3"

#: Qt 6's XCB platform plugin dlopen()s each of these when a window is created,
#: and the failure mode is a process that exits having shown nothing. `libgtk-3-0`
#: is listed because the desktop entry lives in the same hicolor theme the GTK
#: file manager's thumbnailer uses; the app itself needs none of it.
readonly QT_DEPENDS=(
    libgl1
    libegl1
    libxkbcommon0
    libxkbcommon-x11-0
    libx11-xcb1
    libxcb-cursor0
    libxcb-glx0
    libxcb-icccm4
    libxcb-image0
    libxcb-keysyms1
    libxcb-randr0
    libxcb-render-util0
    libxcb-shape0
    libxcb-sync1
    libxcb-util1
    libxcb-xfixes0
    libxcb-xinerama0
    libxcb-xkb1
    libdbus-1-3
    libfontconfig1
    libfreetype6
)

# --- options -----------------------------------------------------------------

# The UPSCALER_* variables are how build-cuda-deb.sh drives this script; see
# the `cuda` note below the argument loop.
FLAVOUR=${UPSCALER_FLAVOUR:-}
FLAVOUR_FLAG=""
FLAVOUR_COUNT=0
OUT_DIR=${UPSCALER_OUT_DIR:-}
VERSION=${UPSCALER_VERSION:-}
EXTRA_INDEX=""
DRY_RUN=${UPSCALER_DRY_RUN:-0}
SKIP_INSTALL=${UPSCALER_SKIP_INSTALL:-0}
KEEP_PAYLOAD=${UPSCALER_KEEP_PAYLOAD:-0}

usage() {
    cat <<'USAGE'
Usage: build-deb.sh --fat|--slim [options]

Build the Debian package for the PySide6 front-end.

Flavour (exactly one is required):
  --fat               Bundle the CPU inference stack (ONNX Runtime plus the
                      CPU build of PyTorch). Runs on a freshly installed system
                      with nothing else to download. Expect ~0.5 GiB.
  --slim              Bundle no inference engine at all; the Runtimes tab
                      installs one on first use. Expect ~0.1 GiB.

Options:
  --out-dir DIR       Where the .deb is written. Default: <repo>/dist
  --version V         Package version. Default: the `version` in pyproject.toml.
  --extra-index-url U Additional package index for the dependency install.
                      Default: the PyTorch index for the chosen flavour.
  --skip-install      Reuse the site directory already in build/, skipping the
                      dependency install and the build venv.
  --keep-payload      Leave the staged payload tree behind for inspection.
  --dry-run           Print every command without running any of them.
  -h, --help          This message.

Both flavours bundle an ffmpeg binary, because the app cannot encode without
one. Only the slim build declares an apt dependency on ffmpeg; the fat build
uses its own copy and, without apt's ffprobe, core.ffmpeg.probe() takes its
documented ffmpeg-only fallback.
USAGE
}

die() {
    printf '%s: %s\n' "$SCRIPT_NAME" "$1" >&2
    exit 1
}

# Progress and diagnostics. stderr, so that a function's stdout stays a single
# meaningful value and a caller can pipe the artefact path out of the build.
note() {
    printf '%s\n' "$*" >&2
}

# Print a command and run it, or print it and run nothing.
run() {
    if ((DRY_RUN)); then
        printf '+' >&2
        printf ' %q' "$@" >&2
        printf '\n' >&2
        return 0
    fi
    "$@"
}

# Create a directory through `run`, so a dry run still shows the layout.
mkdirs() {
    run mkdir -p -- "$@"
}

while (($#)); do
    case $1 in
        --fat)
            FLAVOUR_FLAG=fat
            FLAVOUR_COUNT=$((FLAVOUR_COUNT + 1))
            ;;
        --slim)
            FLAVOUR_FLAG=slim
            FLAVOUR_COUNT=$((FLAVOUR_COUNT + 1))
            ;;
        --out-dir)
            (($# >= 2)) || die "--out-dir needs a directory"
            OUT_DIR=$2
            shift
            ;;
        --out-dir=*)
            OUT_DIR=${1#*=}
            ;;
        --version)
            (($# >= 2)) || die "--version needs a value"
            VERSION=$2
            shift
            ;;
        --version=*)
            VERSION=${1#*=}
            ;;
        --extra-index-url)
            (($# >= 2)) || die "--extra-index-url needs a URL"
            EXTRA_INDEX=$2
            shift
            ;;
        --extra-index-url=*)
            EXTRA_INDEX=${1#*=}
            ;;
        --skip-install)
            SKIP_INSTALL=1
            ;;
        --keep-payload)
            KEEP_PAYLOAD=1
            ;;
        --dry-run)
            DRY_RUN=1
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            printf '%s: unknown argument %s\n\n' "$SCRIPT_NAME" "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

# `cuda` is not a flag: build-cuda-deb.sh sets UPSCALER_FLAVOUR=cuda and runs this
# script, so the offline build is this code path rather than a fork of it. A
# flavour flag on the command line still wins over the environment.
if ((FLAVOUR_COUNT > 1)); then
    die "--fat and --slim are mutually exclusive; pick one"
fi
if ((FLAVOUR_COUNT == 1)); then
    FLAVOUR=$FLAVOUR_FLAG
fi

case $FLAVOUR in
    fat | slim | cuda) ;;
    "")
        printf '%s: one of --fat or --slim is required\n\n' "$SCRIPT_NAME" >&2
        usage >&2
        exit 2
        ;;
    *)
        die "unknown flavour '$FLAVOUR'"
        ;;
esac

case $DRY_RUN in
    0 | 1) ;;
    *) die "UPSCALER_DRY_RUN must be 0 or 1, got '$DRY_RUN'" ;;
esac

# --- toolchain ---------------------------------------------------------------

need() {
    command -v "$1" >/dev/null 2>&1 || die "$1 is not on PATH: $2"
}

need dpkg-deb "it ships with the dpkg package, which is the point of this script"
need python3 "any Python 3.11+; the repository's .venv is preferred"
need du "coreutils"
need sed "coreutils"
need cp "coreutils"
need tar "coreutils; dpkg-deb needs no tar but the payload does not either"

# The build venv is created from the repository's own interpreter when it
# exists, so the bundle is frozen against the Python the project is tested on
# rather than against whatever python3 the distribution happens to ship.
if [[ -x $REPO_ROOT/.venv/bin/python ]]; then
    BASE_PYTHON=$REPO_ROOT/.venv/bin/python
else
    BASE_PYTHON=$(command -v python3)
fi

# --- read the project --------------------------------------------------------

# Query pyproject.toml through a real TOML parser, because the version has
# exactly one home and a second copy in this file is a second thing to forget.
pyproject() {
    "$BASE_PYTHON" - "$REPO_ROOT/pyproject.toml" "$1" "${2:-}" <<'PYTHON'
import re
import sys
import tomllib

with open(sys.argv[1], "rb") as handle:
    project = tomllib.load(handle)

query = sys.argv[2]
if query == "version":
    print(project["project"]["version"])
elif query == "requirements":
    # The base dependencies with the Toga front-end removed, then the flavour's
    # extra when it has one. See the header comment for why toga cannot be
    # installed here, and why `slim` names no extra.
    names: list[str] = []
    wanted = sys.argv[3]
    for requirement in project["project"]["dependencies"]:
        names.append(requirement)
    if wanted:
        for requirement in project["project"]["optional-dependencies"][wanted]:
            names.append(requirement)
    for requirement in names:
        name = re.split(r"[\s\[<>=!~;(]", requirement, maxsplit=1)[0]
        if name.lower().startswith("toga"):
            continue
        print(requirement)
else:
    raise SystemExit(f"unknown pyproject query {query!r}")
PYTHON
}

if [[ -z $VERSION ]]; then
    VERSION=$(pyproject version)
fi
[[ -n $VERSION ]] || die "could not read a version from $REPO_ROOT/pyproject.toml"

[[ -n $EXTRA_INDEX ]] || EXTRA_INDEX=${TORCH_INDEX[$FLAVOUR]}

readonly EXTRA_NAME=${FLAVOUR_EXTRA[$FLAVOUR]:-}

# The artefact name. The offline CUDA build carries a Debian version revision
# rather than a different package name, so dpkg sees it as a distinct version of
# the same application — which is what makes "install one, not both" expressible
# without a circular Conflicts field.
DEB_SUFFIX=""
case $FLAVOUR in
    fat) DEB_SUFFIX="amd64-cpu" ;;
    slim) DEB_SUFFIX="amd64-cuda" ;;
    cuda)
        DEB_SUFFIX="amd64-cuda-offline"
        VERSION="$VERSION-cuda1"
        ;;
esac
readonly DEB_SUFFIX
readonly DEB_NAME="${PACKAGE_NAME}_${VERSION}_${DEB_SUFFIX}.deb"

if [[ -z $OUT_DIR ]]; then
    OUT_DIR=$REPO_ROOT/dist
elif [[ $OUT_DIR != /* ]]; then
    OUT_DIR=$REPO_ROOT/$OUT_DIR
fi

# --- the build tree ----------------------------------------------------------

# Persistent on purpose: the dependency install is ~600 MB of downloads and is
# by far the slowest step, so --skip-install reuses it.
BUILD_DIR=$REPO_ROOT/build/linux-$FLAVOUR
readonly BUILD_DIR
readonly VENV_DIR="$BUILD_DIR/venv"
readonly SITE_DIR="$BUILD_DIR/site"
readonly DIST_DIR="$BUILD_DIR/dist"
readonly WORK_DIR="$BUILD_DIR/work"
readonly SPEC_DIR="$BUILD_DIR/spec"

STAGE=$(mktemp -d -t upscaler-deb-XXXXXX)
readonly STAGE
cleanup() {
    if ((KEEP_PAYLOAD)); then
        note "payload kept at $STAGE"
        return 0
    fi
    # STAGE came from mktemp -d and holds this script's own value, so the path
    # is never assembled from anything a caller supplied.
    rm -rf -- "$STAGE"
}
trap cleanup EXIT

describe() {
    note "flavour     : $FLAVOUR"
    note "version     : $VERSION  (from pyproject.toml unless --version was given)"
    note "python      : $BASE_PYTHON ($("$BASE_PYTHON" -V 2>&1))"
    note "extra index : $EXTRA_INDEX"
    note "build tree  : $BUILD_DIR"
    note "artefact    : $OUT_DIR/$DEB_NAME"
    ((DRY_RUN)) && note "DRY RUN     : every command below is printed, none is run"
    return 0
}

# --- step 1: the dependency tree --------------------------------------------

# Is a distribution installed in the bundle's site directory? The dist-info
# glob matches the *whole* distribution name: truncating it at the first `.`,
# `_` or `-` turned `spandrel_extra_arches` into `spandrel`, which answered
# "yes" for every build, and PyInstaller then failed the hidden import with
# "Hidden import 'spandrel_extra_arches' not found".
site_has() {
    if [[ -e "$SITE_DIR/$1" || -e "$SITE_DIR/$1.py" ]]; then
        return 0
    fi
    local metadata
    metadata=$(compgen -G "$SITE_DIR/$1"'-'*.dist-info 2>/dev/null || true)
    [[ -n $metadata ]]
}

install_dependencies() {
    local -a requirements=()
    local requirement
    while IFS= read -r requirement; do
        requirements+=("$requirement")
    done < <(pyproject requirements "$EXTRA_NAME")

    if ((${#requirements[@]} == 0)); then
        die "pyproject.toml yielded no requirements for the [$EXTRA_NAME] extra"
    fi

    note "step 1/5    : installing ${#requirements[@]} requirements into $SITE_DIR"
    if [[ -n $EXTRA_NAME ]]; then
        note "              (base dependencies minus toga*, plus the [$EXTRA_NAME] extra)"
    else
        note "              (base dependencies minus toga*; this flavour installs no"
        note "               inference extra, which is what makes it the slim build)"
    fi

    if [[ -e $SITE_DIR ]]; then
        [[ $SITE_DIR == "$BUILD_DIR"/* ]] || die "refusing to remove $SITE_DIR"
        run rm -rf -- "$SITE_DIR"
    fi
    run mkdir -p -- "$SITE_DIR"

    if [[ ! -x $VENV_DIR/bin/python ]]; then
        run "$BASE_PYTHON" -m venv "$VENV_DIR"
        run "$VENV_DIR/bin/python" -m pip install --disable-pip-version-check \
            --quiet --upgrade pip
    fi
    run "$VENV_DIR/bin/python" -m pip install --disable-pip-version-check \
        --quiet --upgrade "$PYINSTALLER_PIN"
    run "$VENV_DIR/bin/python" -m pip install --disable-pip-version-check \
        --upgrade --target "$SITE_DIR" --extra-index-url "$EXTRA_INDEX" \
        "${requirements[@]}"
}

# PyPI's `torch==2.7.1` on Linux x86-64 is the CUDA wheel and drags in the whole
# nvidia-* stack, so a build that quietly got one would be several gigabytes of
# files labelled "cpu". This is the check that stops that, and it is a check
# rather than a comment because nothing else would report it.
assert_expected_runtime() {
    ((DRY_RUN)) && return 0
    local -a cuda_packages=()
    local entry
    for entry in "$SITE_DIR"/nvidia_*.dist-info; do
        [[ -e $entry ]] || continue
        cuda_packages+=("$(basename -- "$entry")")
    done
    if [[ $FLAVOUR == cuda ]]; then
        note "              CUDA packages installed, as intended: ${#cuda_packages[@]}"
        return 0
    fi
    if ((${#cuda_packages[@]} > 0)); then
        die "the [$FLAVOUR] build installed ${#cuda_packages[@]} CUDA torch packages (${cuda_packages[*]}). That means $EXTRA_INDEX did not supply a CPU wheel, and the package would be several gigabytes of files labelled \"cpu\". Re-run with --extra-index-url https://download.pytorch.org/whl/cpu, or use build-cuda-deb.sh for a build that is meant to carry the CUDA stack."
    fi
    if [[ $FLAVOUR == fat ]] && [[ ! -d $SITE_DIR/onnxruntime && ! -d $SITE_DIR/torch ]]; then
        die "the [fat] build installed no inference runtime at all: neither onnxruntime nor torch is in $SITE_DIR"
    fi
    return 0
}

# --- step 2: PyInstaller -----------------------------------------------------

run_pyinstaller() {
    local -a args=(
        --noconfirm --clean
        --name "$PACKAGE_NAME"
        --distpath "$DIST_DIR"
        --workpath "$WORK_DIR"
        --specpath "$SPEC_DIR"
        --paths "$SITE_DIR"
        --paths "$REPO_ROOT"
        # models/catalogue.json is data, not a package, and core.paths looks for
        # it under sys._MEIPASS.
        --add-data "$REPO_ROOT/models:models"
        # No --icon here: PyInstaller applies it only on Windows and macOS and
        # warns "Ignoring icon" on Linux, which reads like a missing icon. A
        # Linux package takes its icon from the hicolor theme, and
        # upscaler.desktop.in's Icon= resolves to the PNG installed below.
        # No --collect-all here, which is a change from the plan, and a measured
        # one. On PyInstaller 6.22.3, `--collect-all spandrel|cv2|selectolax`
        # logs "skipping data collection for module ... as it is not a package"
        # for each and collects nothing, because PyInstaller resolves those three
        # as bare modules rather than packages. Static analysis already finds all
        # 43 of spandrel's `spandrel.architectures.*` modules, every `cv2`
        # submodule, and cv2's `cv2.abi3.so`. The hidden imports below are what
        # actually does the work.
        --hidden-import onnxruntime
        --hidden-import spandrel
        --hidden-import cv2
        --hidden-import core.app
        --hidden-import core.pipeline
        --hidden-import core.backends.onnx_backend
        --hidden-import core.backends.torch_backend
        --hidden-import ui_pyside.window
        # The standard library's own dead weight: none of it is imported by the
        # app, and every kilobyte here is a kilobyte in the .deb.
        --exclude-module tkinter
        --exclude-module unittest
        --exclude-module pydoc
        --exclude-module doctest
        --exclude-module pdb
        --exclude-module lib2to3
    )
    if site_has spandrel_extra_arches; then
        args+=(
            --hidden-import spandrel_extra_arches
            --collect-all spandrel_extra_arches
        )
    else
        note "              spandrel_extra_arches is not installed, so the bundle will not"
        note "              offer the architectures from that optional package. Every"
        note "              architecture spandrel ships itself — the Real-ESRGAN family"
        note "              among them — is included."
    fi
    if [[ $FLAVOUR == slim ]]; then
        # The whole point of the slim build: no torch, no onnxruntime. The app
        # still starts, because every module that imports them does so inside a
        # function, and the Runtimes tab puts a real runtime on sys.path.
        args+=(--exclude-module torch --exclude-module onnxruntime)
    fi
    args+=("$REPO_ROOT/ui_pyside/main.py")

    note "step 2/5    : freezing ui_pyside/main.py"
    run "$VENV_DIR/bin/python" -m PyInstaller "${args[@]}"
    ((DRY_RUN)) && return 0
    [[ -x "$DIST_DIR/$PAYLOAD_NAME/$PACKAGE_NAME" ]] ||
        die "PyInstaller finished but produced no $DIST_DIR/$PAYLOAD_NAME/$PACKAGE_NAME"
    return 0
}

# --- step 3: the payload -----------------------------------------------------

# The ffmpeg the app will use. `imageio_ffmpeg` ships a static 7.0.2 binary and
# is deliberately not a runtime dependency, so it is consulted here at build
# time; the one on PATH is the fallback, and if neither exists the build says so
# loudly rather than shipping a package that silently cannot encode.
resolve_ffmpeg() {
    local candidate=""
    local probe
    for probe in "$REPO_ROOT/.venv/bin/python" "$VENV_DIR/bin/python"; do
        [[ -x $probe ]] || continue
        candidate=$("$probe" -c \
            'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' \
            2>/dev/null || true)
        [[ -n $candidate ]] && break
    done
    if [[ -z $candidate ]]; then
        candidate=$(command -v ffmpeg || true)
    fi
    if [[ -z $candidate ]]; then
        printf '              no ffmpeg found. Install imageio-ffmpeg in the build environment\n' >&2
        printf '              or put ffmpeg on PATH; without one this package can only encode\n' >&2
        printf '              after the user installs one.\n' >&2
    else
        printf '              ffmpeg: %s\n' "$candidate" >&2
    fi
    printf '%s' "$candidate"
}

stage_payload() {
    local libdir=$STAGE/usr/lib/$PAYLOAD_NAME
    local ffmpeg
    ffmpeg=$(resolve_ffmpeg)

    note "step 3/5    : staging the payload under $STAGE/usr"
    run rm -rf -- "$STAGE/usr"
    mkdirs \
        "$libdir" \
        "$STAGE/usr/bin" \
        "$STAGE/usr/share/applications" \
        "$STAGE/usr/share/icons/hicolor/${ICON_SIZE}x${ICON_SIZE}/apps"

    run cp -a -- "$DIST_DIR/$PAYLOAD_NAME/." "$libdir/"
    run chmod 0755 -- "$libdir/$PACKAGE_NAME"

    if [[ -n $ffmpeg ]]; then
        run mkdir -p -- "$libdir/bin"
        run cp -a -- "$ffmpeg" "$libdir/bin/ffmpeg"
        run chmod 0755 -- "$libdir/bin/ffmpeg"
    fi

    write_launcher "$STAGE/usr/bin/$LAUNCHER_NAME"
    write_desktop_entry "$STAGE/usr/share/applications/$DESKTOP_ID.desktop"
    run cp -a -- "$REPO_ROOT/packaging/icon.png" \
        "$STAGE/usr/share/icons/hicolor/${ICON_SIZE}x${ICON_SIZE}/apps/$DESKTOP_ID.png"
}

# The launcher derives its prefix from its own location rather than hard-coding
# /usr, so a tree unpacked with `dpkg-deb -x` anywhere is runnable. That is how
# a package is smoke-tested on a machine with no root, and how a user runs it
# from a USB stick.
write_launcher() {
    local path=$1
    if ((DRY_RUN)); then
        printf '+ write %s  # the /usr/bin/%s launcher\n' "$path" \
            "$LAUNCHER_NAME" >&2
        return 0
    fi
    cat >"$path" <<LAUNCHER
#!/bin/sh
# Generated by packaging/linux/build-deb.sh — do not edit.
#
# The prefix comes from this script's own location rather than a hard-coded
# /usr, so a tree unpacked with \`dpkg-deb -x\` anywhere on the filesystem is
# runnable. That is the only way to smoke-test a package on a machine with no
# root, and the release check does exactly that.
set -eu

self=\$0
# Follow symlinks, so a launcher reached through /usr/local/bin still finds the
# bundle it was installed with.
while [ -L "\$self" ]; do
    link=\$(readlink "\$self")
    case \$link in
        /*) self=\$link ;;
        *) self=\$(dirname -- "\$self")/\$link ;;
    esac
done
bundledir=\$(cd -- "\$(dirname -- "\$self")/../lib/$PAYLOAD_NAME" && pwd -P)

if [ ! -x "\$bundledir/$PACKAGE_NAME" ]; then
    printf '%s: %s/%s is missing or not executable; this package is broken\n' \\
        "$LAUNCHER_NAME" "\$bundledir" "$PACKAGE_NAME" >&2
    exit 1
fi

# The bundled shared libraries come first. A system libstdc++ older than the one
# the bundle was built against loads first without this, and the executable dies
# with a bare "symbol lookup error" that names nothing useful.
LD_LIBRARY_PATH="\$bundledir\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH
# The bundle's own ffmpeg, so the app can encode before the user installs one.
PATH="\$bundledir/bin\${PATH:+:\$PATH}"
export PATH

exec "\$bundledir/$PACKAGE_NAME" "\$@"
LAUNCHER
    # A here-document that expands at build time turns a typo in a variable name
    # into a broken package that only fails once a user runs it.
    sh -n "$path" ||
        die "the generated launcher is not valid sh; refusing to ship it"
    chmod 0755 "$path"
}

write_desktop_entry() {
    local path=$1
    local template=$SCRIPT_DIR/$DESKTOP_ID.desktop.in
    [[ -f $template ]] || die "missing $template"
    if ((DRY_RUN)); then
        printf '+ write %s  # %s with @VERSION@ and @ICON@ substituted\n' \
            "$path" "$(basename -- "$template")" >&2
        return 0
    fi
    # A PEP 440 version contains none of the characters `sed` treats specially
    # in a replacement, and the only substitution is a version and an icon id.
    sed -e "s/@VERSION@/$VERSION/g" -e "s/@ICON@/$DESKTOP_ID/g" \
        "$template" >"$path"
    chmod 0644 "$path"
}

# --- step 4: DEBIAN/control --------------------------------------------------

control_file() {
    local -a depends_list=("libgtk-3-0" "${QT_DEPENDS[@]}")
    if [[ $FLAVOUR == slim ]]; then
        # Slim bundles its own ffmpeg, but the app also wants ffprobe: without
        # it core.ffmpeg.probe() takes a full decode pass per source. Asking apt
        # for the package gets a maintained pair instead.
        depends_list+=(ffmpeg)
    fi
    local depends
    depends=$(IFS=,; printf '%s' "${depends_list[*]}")

    local summary
    local -a description
    case $FLAVOUR in
        fat)
            summary="GPU-accelerated video upscaler (bundled CPU runtime)"
            description=(
                "Upscale a video with an open-source super-resolution model, on"
                "every GPU in the machine, resumably."
                ""
                "This is the fat build: the ONNX Runtime CPU provider and the CPU"
                "build of PyTorch are bundled, so the application runs on a"
                "freshly installed system with nothing else to download. GPU"
                "acceleration is added from the Runtimes tab."
            )
            ;;
        slim)
            summary="GPU-accelerated video upscaler (runtime installed on demand)"
            description=(
                "Upscale a video with an open-source super-resolution model, on"
                "every GPU in the machine, resumably."
                ""
                "This is the slim build: no inference engine is bundled. The"
                "Runtimes tab resolves the exact download size from PyPI before"
                "anything is fetched, and the application restarts into whichever"
                "runtime you install."
            )
            ;;
        cuda)
            summary="GPU-accelerated video upscaler (bundled CUDA runtime, offline)"
            description=(
                "Upscale a video with an open-source super-resolution model, on"
                "every GPU in the machine, resumably."
                ""
                "This is the offline CUDA build: the CUDA 12.8 ONNX Runtime and"
                "the matching PyTorch are bundled, so it needs no network at"
                "all. It is several gigabytes and is not published as a release"
                "asset."
            )
            ;;
        *)
            die "unknown flavour $FLAVOUR"
            ;;
    esac

    local installed_size=0
    ((DRY_RUN)) || installed_size=$(du -sk "$STAGE" | cut -f1)

    printf 'Package: %s\n' "$PACKAGE_NAME"
    printf 'Version: %s\n' "$VERSION"
    printf 'Source: %s\n' "$PACKAGE_NAME"
    printf 'Section: utils\n'
    printf 'Priority: optional\n'
    printf 'Architecture: amd64\n'
    printf 'Installed-Size: %s\n' "$installed_size"
    printf 'Maintainer: %s\n' "$MAINTAINER"
    printf 'Homepage: %s\n' "$HOMEPAGE"
    printf 'Depends: %s\n' "$depends"
    printf 'Description: %s\n' "$summary"
    local line
    for line in "${description[@]}"; do
        if [[ -z $line ]]; then
            printf ' .\n'
        else
            printf ' %s\n' "$line"
        fi
    done
}

write_control() {
    note "step 4/5    : writing DEBIAN/control"
    mkdirs "$STAGE/DEBIAN"
    if ((DRY_RUN)); then
        printf '+ write %s  # the package control file\n' "$STAGE/DEBIAN/control" >&2
        control_file | sed -e 's/^/+   /' >&2
        return 0
    fi
    control_file >"$STAGE/DEBIAN/control"
    # dpkg-deb warns loudly about a group- or world-writable control file, and
    # some repositories refuse a package built that way.
    chmod 0755 "$STAGE/DEBIAN"
    chmod 0644 "$STAGE/DEBIAN/control"
}

# --- step 5: the package -----------------------------------------------------

build_package() {
    note "step 5/5    : building the package"
    mkdirs "$OUT_DIR"
    # --root-owner-group: dpkg-deb otherwise records the invoking uid as the
    # owner, which makes the package unusable by root and unverifiable without
    # fakeroot. gzip rather than the default compression: xz spends minutes on
    # half a gigabyte of shared objects that are already compressed.
    run dpkg-deb --root-owner-group -Zgzip --build "$STAGE" "$OUT_DIR/$DEB_NAME"
}

report() {
    local artefact=$OUT_DIR/$DEB_NAME
    local size="not built (dry run)"
    if [[ -f $artefact ]]; then
        size=$("$BASE_PYTHON" - "$artefact" <<'PYTHON'
import os
import sys

size = os.path.getsize(sys.argv[1])
if size >= 1024**3:
    print(f"{size / 1024**3:.2f} GiB ({size} bytes)")
else:
    print(f"{size / 1024**2:.0f} MiB ({size} bytes)")
PYTHON
        )
    fi
    printf 'artefact: %s\n' "$artefact"
    printf 'size:     %s\n' "$size"
    if [[ -f $artefact ]]; then
        printf '\ninstall without root:\n'
        printf '  dpkg-deb -x %s /tmp/upscaler-root\n' "$artefact"
        printf '  /tmp/upscaler-root/usr/bin/%s\n' "$LAUNCHER_NAME"
    fi
}

# --- go ----------------------------------------------------------------------

describe

if ((!SKIP_INSTALL)); then
    install_dependencies
    assert_expected_runtime
else
    note "step 1/5    : skipped, reusing $SITE_DIR"
    assert_expected_runtime
fi

run_pyinstaller
stage_payload
write_control
build_package
report
