#!/usr/bin/env bash
#
# Build the offline fat CUDA package. Never publish it.
#
#   packaging/linux/build-cuda-deb.sh [--out-dir DIR] [--version V] [--dry-run]
#
# This is the fat build with the `cuda` extra instead of the `cpu` one: the CUDA
# 12.8 build of ONNX Runtime plus the matching PyTorch, and the whole nvidia-*
# stack they pull in. It is several gigabytes, it works on a machine with no
# network at all, and it is the artefact a user with a metered connection wants.
#
# It is also the artefact GitHub will not accept. A release asset must be under
# 2 GiB, and this is not, so the build says so in capitals before it starts and
# again after it finishes. Nothing in the release workflow runs this script.
#
# The build itself is `build-deb.sh` with UPSCALER_FLAVOUR=cuda, so the PyInstaller
# invocation, the launcher, the desktop entry and the control file have one home.
# What this script adds is the guard rail and the two facts that only make sense
# here: the Debian version revision that keeps the package distinct from the
# published ones, and the warning.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "$(dirname -- "$SCRIPT_DIR")/.." && pwd -P)
readonly SCRIPT_NAME=$(basename -- "${BASH_SOURCE[0]}")

readonly BUILD_DEB="$SCRIPT_DIR/build-deb.sh"

#: GitHub's documented per-asset ceiling. Quoted in the warning below, and
#: checked against the finished artefact so the number is never a guess.
readonly GITHUB_ASSET_LIMIT_BYTES=2147483648

usage() {
    cat <<'USAGE'
Usage: build-cuda-deb.sh [options]

Build the offline fat CUDA package. NOT FOR PUBLISHING.

Options:
  --out-dir DIR       Where the .deb is written. Default: <repo>/dist
  --version V         Base version. Default: the `version` in pyproject.toml.
                      `-cuda1` is appended, so dpkg sees this as a different
                      version of the same application rather than a second
                      package that would conflict with it.
  --extra-index-url U Additional package index for the dependency install.
  --skip-install      Reuse the site directory already in build/.
  --keep-payload      Leave the staged payload tree behind for inspection.
  --dry-run           Print every command without running any of them.
  -h, --help          This message.

The artefact is roughly 3.5 GB. GitHub rejects any release asset of 2 GiB or
more, so this file must never be attached to a release.
USAGE
}

warn() {
    printf '%s\n' "$*" >&2
}

die() {
    printf '%s: %s\n' "$SCRIPT_NAME" "$1" >&2
    exit 1
}

shout() {
    {
        printf '\n'
        printf '================================================================\n'
        printf '%s\n' "$@"
        printf '================================================================\n'
    } >&2
}

OUT_DIR=""
VERSION=""
DRY_RUN=0
ARGS=()

while (($#)); do
    case $1 in
        --out-dir)
            (($# >= 2)) || die "--out-dir needs a directory"
            OUT_DIR=$2
            ARGS+=(--out-dir "$2")
            shift
            ;;
        --out-dir=*)
            OUT_DIR=${1#*=}
            ARGS+=("$1")
            ;;
        --version)
            (($# >= 2)) || die "--version needs a value"
            VERSION=$2
            ARGS+=(--version "$2")
            shift
            ;;
        --version=*)
            VERSION=${1#*=}
            ARGS+=("$1")
            ;;
        --extra-index-url)
            (($# >= 2)) || die "--extra-index-url needs a URL"
            ARGS+=("$1" "$2")
            shift
            ;;
        --extra-index-url=* | --skip-install | --keep-payload | --dry-run)
            if [[ $1 == --dry-run ]]; then
                DRY_RUN=1
            fi
            ARGS+=("$1")
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

[[ -f $BUILD_DEB ]] || die "$BUILD_DEB is missing; this script only delegates to it"
[[ -x $BUILD_DEB ]] || die "$BUILD_DEB is not executable"

shout \
    "THIS BUILD PRODUCES AN ARTEFACT THAT MUST NOT BE UPLOADED TO A GITHUB RELEASE." \
    "" \
    "GitHub's limit is 2 GiB (2147483648 bytes) per release asset, and this" \
    "package is several gigabytes: it bundles the CUDA 12.8 build of ONNX" \
    "Runtime, the matching PyTorch, and the whole nvidia-* stack they depend" \
    "on. The upload would be rejected." \
    "" \
    "The published Linux artefacts are upscaler_<ver>_amd64-cpu.deb and" \
    "upscaler_<ver>_amd64-cuda.deb, both built by build-deb.sh. This one is for" \
    "local installation and for handing to someone on a stick."

UPSCALER_FLAVOUR=cuda
UPSCALER_OUT_DIR=$OUT_DIR
UPSCALER_VERSION=$VERSION
UPSCALER_DRY_RUN=$DRY_RUN
export UPSCALER_FLAVOUR
if [[ -n $OUT_DIR ]]; then
    export UPSCALER_OUT_DIR
fi
if [[ -n $VERSION ]]; then
    export UPSCALER_VERSION
fi
export UPSCALER_DRY_RUN

warn ""
warn "$SCRIPT_NAME: delegating to $BUILD_DEB with UPSCALER_FLAVOUR=cuda"
warn ""

"$BUILD_DEB" "${ARGS[@]+"${ARGS[@]}"}"

if ((DRY_RUN)); then
    shout "DRY RUN: NO PACKAGE WAS BUILT AND NONE EXISTS TO UPLOAD."
    exit 0
fi

# The artefact name build-deb.sh derived from the flavour, so there is still one
# place that decides it.
BASE_VERSION=$VERSION
if [[ -z $BASE_VERSION ]]; then
    BASE_VERSION=$(sed -n 's/^version = "\(.*\)"$/\1/p' "$REPO_ROOT/pyproject.toml" \
        | head -1)
fi
[[ -n $BASE_VERSION ]] || die "could not read a version from pyproject.toml"
ARTEFACT="$REPO_ROOT/dist/upscaler_${BASE_VERSION}-cuda1_amd64-cuda-offline.deb"
if [[ -n $OUT_DIR && $OUT_DIR = /* ]]; then
    ARTEFACT="$OUT_DIR/upscaler_${BASE_VERSION}-cuda1_amd64-cuda-offline.deb"
fi

[[ -f $ARTEFACT ]] || die "expected the build to produce $ARTEFACT, and it is not there"

SIZE=$(wc -c <"$ARTEFACT")
warn ""
warn "$SCRIPT_NAME: $ARTEFACT"
warn "$SCRIPT_NAME: $SIZE bytes ($(awk -v s="$SIZE" 'BEGIN{printf "%.2f", s/1073741824}') GiB)"

if ((SIZE >= GITHUB_ASSET_LIMIT_BYTES)); then
    shout \
        "DO NOT UPLOAD THIS FILE TO A GITHUB RELEASE." \
        "" \
        "$ARTEFACT is $SIZE bytes." \
        "GitHub's documented limit is $GITHUB_ASSET_LIMIT_BYTES bytes (2 GiB) per" \
        "release asset, and this is over it. The upload will be rejected." \
        "" \
        "Install it locally instead:"
        "  sudo dpkg -i $ARTEFACT"
else
    shout \
        "WARNING: this artefact is under GitHub's 2 GiB limit, which is not" \
        "expected for a CUDA build. Check that the cuda extra really was" \
        "installed before treating it as publishable."
fi

warn ""
warn "$SCRIPT_NAME: build finished"
