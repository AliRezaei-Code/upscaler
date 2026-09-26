#!/usr/bin/env bash
#
# Build the macOS disk image for the PySide6 front-end.
#
#   packaging/macos/build-dmg.sh [--out-dir DIR] [--version V] [--dry-run]
#
# PyInstaller --windowed produces the .app; `create-dmg` wraps it in a
# compressed image with the usual drag-to-Applications layout.
#
# Bundled runtime: `onnxruntime` — whose stock macOS wheel already carries the
# CoreML execution provider, so there is no separate CoreML package to fetch —
# plus `torch`, because `mps` is the documented fallback for machines whose
# CoreML path is refused. Both come from the base dependencies and the `cpu`
# extra, so this build installs exactly the same dependency set as the Linux fat
# build and PyInstaller freezes it identically.
#
# Cross-building is not attempted. PyInstaller's macOS bootloader, the codesign
# and notarisation steps and `create-dmg` are all macOS-only, and a Linux host
# that "succeeded" would produce an image nobody can open. So this script stops
# on a non-Darwin host before it has done anything.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "$(dirname -- "$SCRIPT_DIR")/.." && pwd -P)

SCRIPT_NAME=$(basename -- "${BASH_SOURCE[0]}")

#: `MLProgram`, the CoreML model format this build asks for, is refused below
#: macOS 12. `core/backends/onnx_backend.py` retries with `NeuralNetwork`, so
#: this is the real floor rather than a policy choice.
readonly MACOS_MINIMUM_VERSION="12.0"
readonly BUNDLE_ID="org.alirezaei.upscaler"
readonly PYINSTALLER_PIN="pyinstaller==6.22.3"

OUT_DIR=""
VERSION=""
DRY_RUN=0
SKIP_INSTALL=0

usage() {
    cat <<'USAGE'
Usage: build-dmg.sh [options]

Build the macOS .dmg for the PySide6 front-end. macOS hosts only.

Options:
  --out-dir DIR       Where the .dmg is written. Default: <repo>/dist
  --version V         Version in the file name. Default: the `version` in
                      pyproject.toml.
  --skip-install      Reuse the site directory already in build/.
  --dry-run           Print every command without running any of them.
  -h, --help          This message.

The image is about 0.6 GiB. The minimum macOS version is 12.0, because the
CoreML MLProgram model format is refused below it.

This script does not sign or notarise the application. A build that will be
distributed has to be signed with a Developer ID and notarised, or Gatekeeper
will refuse it on any machine but the one that built it. The release check runs
the application from the bundle, so an unsigned build is fine there and not fine
in a user's hands.
USAGE
}

die() {
    printf '%s: %s\n' "$SCRIPT_NAME" "$1" >&2
    exit 1
}

note() {
    printf '%s\n' "$*" >&2
}

run() {
    if ((DRY_RUN)); then
        printf '+' >&2
        printf ' %q' "$@" >&2
        printf '\n' >&2
        return 0
    fi
    "$@"
}

while (($#)); do
    case $1 in
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
        --skip-install)
            SKIP_INSTALL=1
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

# --- the host, before anything else ------------------------------------------

if [[ $(uname -s) != Darwin ]]; then
    die "this script only runs on macOS. PyInstaller's macOS bootloader, codesign and create-dmg are all macOS-only, and a cross-built image would be one nobody can open. Build the .dmg on a macOS runner with: $0"
fi

if ((DRY_RUN)); then
    # The plan and the failure message are the two things a non-macOS author
    # needs from this script, and a dry run has to be able to show them from
    # anywhere.
    note "DRY RUN: the host check is the one thing a dry run cannot verify."
fi

for tool in sw_vers /usr/libexec/PlistBuddy create-dmg; do
    if ! command -v "$tool" >/dev/null 2>&1 && [[ ! -x $tool ]]; then
        case $tool in
            create-dmg)
                die "create-dmg is not on PATH. Install it with: brew install create-dmg"
                ;;
            /usr/libexec/PlistBuddy)
                die "$tool is missing; it ships with the Xcode command line tools (xcode-select --install)"
                ;;
            *)
                die "$tool is missing"
                ;;
        esac
    fi
done

# --- the project -------------------------------------------------------------

if [[ -x $REPO_ROOT/.venv/bin/python ]]; then
    BASE_PYTHON=$REPO_ROOT/.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
    BASE_PYTHON=$(command -v python3)
else
    die "no Python 3.11+ found; create $REPO_ROOT/.venv first"
fi

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
    # The base dependencies without the Toga front-end — `toga-gtk` needs a
    # source-built PyGObject — plus the `cpu` extra, whose `onnxruntime` is the
    # macOS wheel carrying the CoreML execution provider.
    names = list(project["project"]["dependencies"])
    names += list(project["project"]["optional-dependencies"][sys.argv[3]])
    for requirement in names:
        name = re.split(r"[\s\[<>=!~;(]", requirement, maxsplit=1)[0]
        if not name.lower().startswith("toga"):
            print(requirement)
else:
    raise SystemExit(f"unknown pyproject query {query!r}")
PYTHON
}

[[ -n $VERSION ]] || VERSION=$(pyproject version)
[[ -n $VERSION ]] || die "could not read a version from $REPO_ROOT/pyproject.toml"

OUT_DIR=${OUT_DIR:-$REPO_ROOT/dist}
[[ $OUT_DIR = /* ]] || OUT_DIR=$REPO_ROOT/$OUT_DIR

BUILD_DIR=$REPO_ROOT/build/macos
readonly BUILD_DIR
readonly VENV_DIR="$BUILD_DIR/venv"
readonly SITE_DIR="$BUILD_DIR/site"
readonly DIST_DIR="$BUILD_DIR/dist"
readonly WORK_DIR="$BUILD_DIR/work"
readonly SPEC_DIR="$BUILD_DIR/spec"
readonly APP_DIR="$DIST_DIR/upscaler.app"
readonly DMG_PATH="$OUT_DIR/upscaler_${VERSION}_macos.dmg"

STAGE=$(mktemp -d -t upscaler-dmg-XXXXXX)
readonly STAGE
cleanup() {
    rm -rf -- "$STAGE"
}
trap cleanup EXIT

# --- step 1: the dependency tree --------------------------------------------

install_dependencies() {
    local -a requirements=()
    local requirement
    while IFS= read -r requirement; do
        requirements+=("$requirement")
    done < <(pyproject requirements cpu)

    note "step 1/5    : installing ${#requirements[@]} requirements into $SITE_DIR"
    if [[ -e $SITE_DIR ]]; then
        [[ $SITE_DIR == "$BUILD_DIR"/* ]] || die "refusing to remove $SITE_DIR"
        run rm -rf -- "$SITE_DIR"
    fi
    run mkdir -p -- "$SITE_DIR"
    if [[ ! -x $VENV_DIR/bin/python ]]; then
        run "$BASE_PYTHON" -m venv "$VENV_DIR"
    fi
    run "$VENV_DIR/bin/python" -m pip install --disable-pip-version-check \
        --quiet --upgrade "$PYINSTALLER_PIN"
    run "$VENV_DIR/bin/python" -m pip install --disable-pip-version-check \
        --upgrade --target "$SITE_DIR" "${requirements[@]}"

    # The CoreML execution provider is compiled into the stock macOS
    # `onnxruntime` wheel, so if it is missing, the `cpu` extra installed the
    # wrong wheel and the image would silently fall back to the CPU.
    if ((DRY_RUN)); then
        return 0
    fi
    local providers
    providers=$(PYTHONPATH=$SITE_DIR "$VENV_DIR/bin/python" -c \
        'import onnxruntime; print(",".join(onnxruntime.get_available_providers()))')
    note "              onnxruntime providers: $providers"
    case $providers in
        *CoreMLExecutionProvider*) ;;
        *)
            die "the installed onnxruntime does not offer CoreMLExecutionProvider (it offers: $providers). The macOS wheel of 'onnxruntime' carries the CoreML provider; a build without it cannot use the GPU and this image would say it can."
            ;;
    esac
}

# --- step 2: PyInstaller -----------------------------------------------------

run_pyinstaller() {
    note "step 2/5    : freezing ui_pyside/main.py"
    run "$VENV_DIR/bin/python" -m PyInstaller \
        --noconfirm --clean \
        --name upscaler \
        --distpath "$DIST_DIR" \
        --workpath "$WORK_DIR" \
        --specpath "$SPEC_DIR" \
        --paths "$SITE_DIR" \
        --paths "$REPO_ROOT" \
        --add-data "$REPO_ROOT/models:models" \
        --windowed \
        --collect-all spandrel \
        --collect-all cv2 \
        --collect-all selectolax \
        --hidden-import onnxruntime \
        --hidden-import spandrel \
        --hidden-import cv2 \
        --hidden-import core.app \
        --hidden-import core.pipeline \
        --hidden-import core.backends.onnx_backend \
        --hidden-import core.backends.torch_backend \
        --hidden-import ui_pyside.window \
        --exclude-module tkinter \
        --exclude-module unittest \
        --exclude-module pydoc \
        --exclude-module doctest \
        --exclude-module pdb \
        --exclude-module lib2to3 \
        "$REPO_ROOT/ui_pyside/main.py"
    ((DRY_RUN)) && return 0
    [[ -d $APP_DIR ]] ||
        die "PyInstaller finished but produced no $APP_DIR"
    return 0
}

# --- step 3: the .app's Info.plist ------------------------------------------

write_info_plist() {
    local plist=$APP_DIR/Contents/Info.plist
    note "step 3/5    : writing Info.plist"
    if ((DRY_RUN)); then
        printf '+ write %s  # LSMinimumSystemVersion %s, CFBundleIdentifier %s\n' \
            "$plist" "$MACOS_MINIMUM_VERSION" "$BUNDLE_ID" >&2
        return 0
    fi
    [[ -f $plist ]] || die "PyInstaller produced no $plist"
    # Set/Add rather than a wholesale rewrite: PyInstaller's own plist carries
    # the executable name and the version, and replacing it loses both.
    /usr/libexec/PlistBuddy -c "Set :LSMinimumSystemVersion $MACOS_MINIMUM_VERSION" \
        "$plist" 2>/dev/null ||
        /usr/libexec/PlistBuddy -c "Add :LSMinimumSystemVersion string $MACOS_MINIMUM_VERSION" \
            "$plist"
    /usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier $BUNDLE_ID" "$plist" 2>/dev/null ||
        /usr/libexec/PlistBuddy -c "Add :CFBundleIdentifier string $BUNDLE_ID" "$plist"
    /usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $VERSION" "$plist" 2>/dev/null ||
        /usr/libexec/PlistBuddy -c "Add :CFBundleShortVersionString string $VERSION" "$plist"
    local minimum
    minimum=$(/usr/libexec/PlistBuddy -c 'Print :LSMinimumSystemVersion' "$plist")
    [[ $minimum == "$MACOS_MINIMUM_VERSION" ]] ||
        die "LSMinimumSystemVersion is $minimum after the edit, not $MACOS_MINIMUM_VERSION"
    note "              LSMinimumSystemVersion: $minimum"
}

# --- step 4: the image --------------------------------------------------------

build_dmg() {
    note "step 4/5    : building the disk image"
    run mkdir -p -- "$OUT_DIR"
    if ((DRY_RUN)); then
        run create-dmg \
            --volname "Upscaler" \
            --window-size 660 400 \
            --icon-size 110 \
            --icon "Upscaler" 180 200 \
            --app-drop-link 480 200 \
            --no-internet-enable \
            "$DMG_PATH" \
            "$APP_DIR" \
            /Applications
        return 0
    fi
    # create-dmg needs a staging directory with the .app and an /Applications
    # symlink, or it silently produces an image with no way to drag anywhere.
    local staging=$STAGE/dmg
    run mkdir -p -- "$staging"
    run cp -a -- "$APP_DIR" "$staging/Upscaler.app"
    run ln -s /Applications "$staging/Applications"
    run create-dmg \
        --volname "Upscaler" \
        --window-size 660 400 \
        --icon-size 110 \
        --icon "Upscaler" 180 200 \
        --app-drop-link 480 200 \
        --no-internet-enable \
        "$DMG_PATH" \
        "$staging" \
        /Applications
}

report() {
    local size="not built (dry run)"
    if [[ -f $DMG_PATH ]]; then
        size=$("$BASE_PYTHON" - "$DMG_PATH" <<'PYTHON'
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
    printf 'artefact: %s\n' "$DMG_PATH"
    printf 'size:     %s\n' "$size"
    printf 'minimum:  macOS %s\n' "$MACOS_MINIMUM_VERSION"
    if ((DRY_RUN == 0)); then
        printf '\nthis image is not signed or notarised. Verify the bundle with:\n'
        printf '  .venv/bin/python scripts/verify_install.py --app-dir %s\n' "$APP_DIR"
    fi
}

# --- go ----------------------------------------------------------------------

note "version     : $VERSION  (from pyproject.toml unless --version was given)"
note "python      : $BASE_PYTHON ($("$BASE_PYTHON" -V 2>&1))"
note "macos       : $(sw_vers -productVersion 2>/dev/null || echo unknown)"
note "build tree  : $BUILD_DIR"
note "artefact    : $DMG_PATH"
((DRY_RUN)) && note "DRY RUN     : every command below is printed, none is run"
note ""

if ((SKIP_INSTALL)); then
    note "step 1/5    : skipped, reusing $SITE_DIR"
else
    install_dependencies
fi
run_pyinstaller
write_info_plist
build_dmg

note ""
note "step 5/5    : done"
report
