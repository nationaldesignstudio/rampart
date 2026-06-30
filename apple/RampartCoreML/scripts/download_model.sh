#!/usr/bin/env bash
set -euo pipefail

REPO="${RAMPART_COREML_REPO:-nationaldesignstudio/rampart}"
RELEASE_TAG="${RAMPART_COREML_RELEASE_TAG:-0.1.3}"
ASSET_NAME="${RAMPART_COREML_ASSET_NAME:-rampart-coreml-artifacts.zip}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_DOWNLOAD_URL="https://github.com/${REPO}/releases/download/${RELEASE_TAG}/${ASSET_NAME}"
DOWNLOAD_URL="${RAMPART_COREML_URL:-${DEFAULT_DOWNLOAD_URL}}"
TMP_DIR="$(mktemp -d)"

cleanup() {
    rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

if ! command -v curl >/dev/null 2>&1; then
    echo "error: curl is required to download the model artifacts." >&2
    exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
    echo "error: unzip is required to extract the model artifacts." >&2
    exit 1
fi

archive_path="${TMP_DIR}/${ASSET_NAME}"

echo "Downloading Rampart Core ML artifacts..."
echo "${DOWNLOAD_URL}"
curl --fail --location --show-error --output "${archive_path}" "${DOWNLOAD_URL}"

echo "Extracting into ${REPO_ROOT}/artifacts..."
unzip -q -o "${archive_path}" -d "${REPO_ROOT}"

required_paths=(
    "artifacts/RampartTokenClassifier.mlpackage"
    "artifacts/rampart-hf/vocab.txt"
    "artifacts/rampart-hf/config.json"
)

for relative_path in "${required_paths[@]}"; do
    if [[ ! -e "${REPO_ROOT}/${relative_path}" ]]; then
        echo "error: expected ${relative_path} after extraction." >&2
        exit 1
    fi
done

echo "Model artifacts are ready:"
printf '  %s\n' "${required_paths[@]}"
