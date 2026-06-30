# Release

## Verify

```bash
bun run build
bun test
bun run type-check
bun run eval:public:strict
bun run export:huggingface:verify
npm pack --dry-run --cache /tmp/npm-cache-pii-filter
```

Equivalent single command:

```bash
bun run verify:public
```

`eval:public:strict` enforces minimum thresholds (private recall ≥ 95%, public
retention ≥ 95%) on the public chat-case suite rather than requiring zero leaks;
it fails on a regression below either floor. The thresholds are pinned in
`eval/run-public-eval.ts`. The headline OpenPII numbers come from `eval/bench`.

## npm

```bash
npm publish --access restricted
```

The package publishes compiled ESM from `dist/`, TypeScript declarations, source,
examples, eval fixtures, docs, and the browser model files.

## Hugging Face

Target repo: `nationaldesignstudio/rampart` (private).

With `hf` authenticated:

```bash
hf repos create nationaldesignstudio/rampart --type model --private
bun run publish:huggingface
```

Manual Git LFS flow:

```bash
bun run export:huggingface:verify
cd hf-export
git init
git lfs install
git add .
git commit -m "Add Rampart model"
git remote add origin https://huggingface.co/nationaldesignstudio/rampart
git push origin main
```

The export contains:

- `README.md` with Hugging Face model metadata.
- ONNX model and tokenizer files.
- `.gitattributes` for ONNX Git LFS tracking.
- `LICENSE`, `MODEL_CARD.md`, `WHITEPAPER.md`, and `examples/basic-chat.ts`.

## Core ML

The release workflow also builds the Apple runtime artifact from the released
ONNX model and attaches two files to the GitHub Release:

- `rampart-coreml-artifacts.zip`
- `rampart-coreml-artifacts.zip.sha256`

The ZIP contains the generated Core ML package plus the tokenizer vocabulary and
config files required by the Swift runtime:

```text
artifacts/RampartTokenClassifier.mlpackage
artifacts/rampart-hf/vocab.txt
artifacts/rampart-hf/config.json
```

Swift package users download this release asset automatically through
`RampartCoreMLClassifier.downloaded()`. The same artifact can be generated
locally from the upstream ONNX model:

```bash
cd apple/RampartCoreML
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-convert.txt
python scripts/convert_rampart_to_coreml.py
```

## Automated release (CI)

Publishing both channels is wired into GitHub Actions
(`.github/workflows/release.yml`). To cut a release:

1. Bump `version` in `package.json` and land it on `main`.
2. Publish a GitHub Release whose tag matches that version.

The workflow re-runs `verify:public` + an `npm pack` dry run as a shared gate,
then publishes to npm (`npm publish --access restricted`), Hugging Face
(`export:huggingface:verify` → `hf upload`), and the Core ML GitHub Release
artifact as independent jobs. `npm` rejects republishing an existing version, so
the tag/release version is what ships.

Run it manually with `workflow_dispatch` to rehearse — it defaults to a dry run
(builds and verifies, publishes nothing).

Required repository secrets:

| Secret | Used by | Purpose |
| --- | --- | --- |
| `NPM_TOKEN` | npm job | npm automation token with publish rights to `@nationaldesignstudio` |
| `HF_TOKEN` | Hugging Face job | Hugging Face write token for `nationaldesignstudio/rampart` |
| `ANTHROPIC_API_KEY` | Claude review/assistant workflows | Powers `claude-code-review.yml` and `claude.yml` |

The manual `npm publish` / Git-LFS flows above remain the fallback if you ever
need to publish from a workstation.

## GitHub

This repository is the standalone open-source home for Rampart. The GitHub
workflow (`.github/workflows/ci.yml`) runs `verify:public` and an npm pack dry
run on every push and pull request. `.github/workflows/release.yml` ships to npm
and Hugging Face on a published release.
