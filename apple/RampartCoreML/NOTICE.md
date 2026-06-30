# Attribution Notice

This repository contains a Core ML conversion and Swift integration for the
Rampart PII token-classification model.

## Upstream Model

- Upstream model: `nationaldesignstudio/rampart`
- Upstream URL: https://huggingface.co/nationaldesignstudio/rampart
- Upstream package: https://www.npmjs.com/package/@nationaldesignstudio/rampart
- Upstream license: Creative Commons Attribution 4.0 International
  (`CC-BY-4.0`)

The upstream model card states that the model weights, deterministic layer, and
TypeScript evaluation harness are released under CC BY 4.0.

## Changes In This Repository

This repository adapts the upstream Rampart artifacts for local Apple-platform
use by:

- downloading the released ONNX/tokenizer/config artifacts locally;
- converting the released ONNX token-classifier weights into a Core ML
  `.mlpackage`;
- adding Swift package code for WordPiece tokenization, Core ML inference, and
  a local conversation guard;
- adding a deterministic recognizer for Rampart's structured classes: SSN,
  credit card, email, URL, and IP address;
- adding structured-span pre-masking, raw-offset projection, default redaction
  policy handling, stable placeholders, and reveal for Apple clients;
- adding a CLI, parity validation scripts, tests, and a minimal iOS example app.

The generated Core ML package and downloaded Hugging Face artifacts retain the
same CC BY 4.0 license and attribution requirements as the upstream model.
