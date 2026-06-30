#!/usr/bin/env python3
"""Validate Rampart ONNX/Core ML parity on real text fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


DEFAULT_MODEL_DIR = Path("artifacts/rampart-hf")
DEFAULT_COREML_PATH = Path("artifacts/RampartTokenClassifier.mlpackage")
DEFAULT_SEQUENCE_LENGTH = 512
MAX_ABS_DIFF_THRESHOLD = 0.25
MEAN_ABS_DIFF_THRESHOLD = 0.02

@dataclass(frozen=True)
class Fixture:
    name: str
    text: str


FIXTURES = [
    Fixture(
        name="name_city_state",
        text="My name is Clara and I live in Berkeley, California.",
    ),
    Fixture(
        name="multi_part_person_name",
        text="My full name is Clara Maria Rivera Lopez.",
    ),
    Fixture(
        name="email_phone",
        text=(
            "Email sarah.rivera@example.com and alex.maria.lopez@sub.example.co.uk "
            "or call 415-555-0199."
        ),
    ),
    Fixture(
        name="url",
        text="Go to http://foo.bar/baz or www.foo.bar.",
    ),
    Fixture(
        name="tax_routing_bank_account",
        text=(
            "TIN 98-7654321 EIN 12-3456789 ITIN 912-70-1234 "
            "IBAN GB82WEST12345698765432 routing 011000015 account 123456789012"
        ),
    ),
    Fixture(
        name="bank_account_continuation",
        text="bank account 17 6175 7758 20 247 9.",
    ),
    Fixture(
        name="government_passport_license",
        text="Government ID A1234567, passport C12345678, drivers license D1234567.",
    ),
    Fixture(
        name="government_id_continuation",
        text="my document id is ID-987-654-321.",
    ),
    Fixture(
        name="passport_continuation",
        text="passport P 123 456 7.",
    ),
    Fixture(
        name="street_address",
        text="Alex Rivera lives at 221B Baker Street Apt 4 in London.",
    ),
    Fixture(
        name="fractional_address",
        text="Mailing address 12 1/2 Main Street Apt 3, New York, NY 10001.",
    ),
    Fixture(
        name="city_state_multi_token",
        text="I live in New York City, New York 10001.",
    ),
    Fixture(
        name="zip_plus_four",
        text="zip 10001-1234.",
    ),
    Fixture(
        name="routing_hyphen_license_spaced",
        text="Routing number 021-000-021 and drivers license D 123 456 7.",
    ),
    Fixture(
        name="multiline_mixed_pii",
        text=(
            "Patient: Maria Lopez\n"
            "Email: maria.lopez@example.com\n"
            "Address: 742 Evergreen Terrace, Springfield, IL 62704\n"
            "Phone: 312-555-0134"
        ),
    ),
    Fixture(
        name="accent_lowercase_name",
        text="josé müller moved from berlin to lisbon.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--onnx-path", type=Path, default=None)
    parser.add_argument("--coreml-path", type=Path, default=DEFAULT_COREML_PATH)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def load_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    with config_path.open() as f:
        return json.load(f)


def load_tokenizer(model_dir: Path) -> Tokenizer:
    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Missing tokenizer: {tokenizer_path}")
    return Tokenizer.from_file(str(tokenizer_path))


def encode_fixture(
    tokenizer: Tokenizer,
    fixture: Fixture,
    sequence_length: int,
    pad_token_id: int,
) -> dict[str, Any]:
    encoding = tokenizer.encode(fixture.text)
    active_length = min(len(encoding.ids), sequence_length)

    input_ids = np.full((1, sequence_length), pad_token_id, dtype=np.int64)
    attention_mask = np.zeros((1, sequence_length), dtype=np.int64)
    token_type_ids = np.zeros((1, sequence_length), dtype=np.int64)

    input_ids[0, :active_length] = encoding.ids[:active_length]
    attention_mask[0, :active_length] = 1
    token_type_ids[0, :active_length] = encoding.type_ids[:active_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "tokens": encoding.tokens[:active_length],
        "offsets": encoding.offsets[:active_length],
        "active_length": active_length,
        "truncated": len(encoding.ids) > sequence_length,
    }


def run_onnx(session: ort.InferenceSession, encoded: dict[str, Any]) -> np.ndarray:
    (logits,) = session.run(
        None,
        {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "token_type_ids": encoded["token_type_ids"],
        },
    )
    return logits


def run_coreml(model: ct.models.MLModel, encoded: dict[str, Any]) -> np.ndarray:
    outputs = model.predict(
        {
            "input_ids": encoded["input_ids"].astype(np.int32),
            "attention_mask": encoded["attention_mask"].astype(np.int32),
            "token_type_ids": encoded["token_type_ids"].astype(np.int32),
        }
    )
    if "logits" not in outputs:
        raise KeyError(f"Core ML output did not include 'logits'; found {sorted(outputs)}")
    return outputs["logits"]


def entity_type(label: str) -> str:
    if label.startswith(("B-", "I-")):
        return label[2:]
    return label


def all_model_entity_types(id2label: dict[int, str]) -> set[str]:
    return {
        entity_type(label)
        for label in id2label.values()
        if label != "O"
    }


def all_model_bio_labels(id2label: dict[int, str]) -> set[str]:
    return {
        label
        for label in id2label.values()
        if label != "O"
    }


def non_o_predictions(
    tokens: list[str],
    offsets: list[tuple[int, int]],
    onnx_labels: list[str],
    coreml_labels: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for index, (token, offset, onnx_label, coreml_label) in enumerate(
        zip(tokens, offsets, onnx_labels, coreml_labels)
    ):
        if token in {"[CLS]", "[SEP]", "[PAD]"} or offset == (0, 0):
            continue
        if onnx_label == "O" and coreml_label == "O":
            continue
        rows.append(
            {
                "index": index,
                "token": token,
                "offset": [int(offset[0]), int(offset[1])],
                "onnx_label": onnx_label,
                "coreml_label": coreml_label,
            }
        )
    return rows


def validate_fixture(
    fixture: Fixture,
    encoded: dict[str, Any],
    onnx_logits: np.ndarray,
    coreml_logits: np.ndarray,
    id2label: dict[int, str],
    expected_shape: tuple[int, int, int],
) -> dict[str, Any]:
    failures = []
    if tuple(onnx_logits.shape) != expected_shape:
        failures.append(f"ONNX shape {tuple(onnx_logits.shape)} != {expected_shape}")
    if tuple(coreml_logits.shape) != expected_shape:
        failures.append(f"Core ML shape {tuple(coreml_logits.shape)} != {expected_shape}")

    active_length = encoded["active_length"]
    active_onnx_logits = onnx_logits[:, :active_length, :]
    active_coreml_logits = coreml_logits[:, :active_length, :]
    diff = np.abs(active_onnx_logits - active_coreml_logits)
    max_abs_diff = float(np.max(diff))
    mean_abs_diff = float(np.mean(diff))
    per_label_max_abs_diff = {
        id2label[label_index]: float(np.max(diff[:, :, label_index]))
        for label_index in sorted(id2label)
    }
    per_label_mean_abs_diff = {
        id2label[label_index]: float(np.mean(diff[:, :, label_index]))
        for label_index in sorted(id2label)
    }

    if max_abs_diff > MAX_ABS_DIFF_THRESHOLD:
        failures.append(
            f"max_abs_diff {max_abs_diff:.6f} > {MAX_ABS_DIFF_THRESHOLD:.6f}"
        )
    if mean_abs_diff > MEAN_ABS_DIFF_THRESHOLD:
        failures.append(
            f"mean_abs_diff {mean_abs_diff:.6f} > {MEAN_ABS_DIFF_THRESHOLD:.6f}"
        )

    onnx_argmax = np.argmax(onnx_logits, axis=-1)[0, :active_length]
    coreml_argmax = np.argmax(coreml_logits, axis=-1)[0, :active_length]
    mismatches = np.flatnonzero(onnx_argmax != coreml_argmax).tolist()
    if mismatches:
        failures.append(f"{len(mismatches)} active-token argmax label mismatch(es)")

    onnx_labels = [id2label[int(index)] for index in onnx_argmax]
    coreml_labels = [id2label[int(index)] for index in coreml_argmax]
    predictions = non_o_predictions(
        encoded["tokens"],
        encoded["offsets"],
        onnx_labels,
        coreml_labels,
    )

    return {
        "name": fixture.name,
        "text": fixture.text,
        "active_length": int(active_length),
        "truncated": bool(encoded["truncated"]),
        "onnx_shape": list(onnx_logits.shape),
        "coreml_shape": list(coreml_logits.shape),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "per_label_max_abs_diff": per_label_max_abs_diff,
        "per_label_mean_abs_diff": per_label_mean_abs_diff,
        "argmax_mismatch_count": len(mismatches),
        "argmax_mismatch_indices": [int(index) for index in mismatches],
        "predictions": predictions,
        "failures": failures,
    }


def aggregate_per_label(
    results: list[dict[str, Any]],
    key: str,
) -> dict[str, float]:
    labels = results[0][key].keys()
    return {
        label: max(result[key][label] for result in results)
        for label in labels
    }


def format_human_report(results: list[dict[str, Any]], failures: list[str]) -> str:
    lines = ["Rampart Core ML parity report", ""]
    for result in results:
        status = "PASS" if not result["failures"] else "FAIL"
        text = result["text"].replace("\n", "\\n")
        lines.append(f"[{status}] {result['name']}")
        lines.append(f"  text: {text}")
        lines.append(
            "  active_tokens={active_length} max_abs_diff={max_abs_diff:.6f} "
            "mean_abs_diff={mean_abs_diff:.6f} argmax_mismatches={argmax_mismatch_count}".format(
                **result
            )
        )
        if result["truncated"]:
            lines.append("  truncated: true")
        if result["predictions"]:
            lines.append("  non-O predictions:")
            for prediction in result["predictions"]:
                start, end = prediction["offset"]
                label = prediction["onnx_label"]
                if prediction["onnx_label"] != prediction["coreml_label"]:
                    label = f"{prediction['onnx_label']} / {prediction['coreml_label']}"
                lines.append(
                    f"    {prediction['index']:>3} {prediction['token']:<16} "
                    f"[{start},{end}) {label}"
                )
        else:
            lines.append("  non-O predictions: none")
        for failure in result["failures"]:
            lines.append(f"  failure: {failure}")
        lines.append("")

    observed_entity_types = sorted({
        entity_type(prediction["onnx_label"])
        for result in results
        for prediction in result["predictions"]
    })
    observed_labels = sorted({
        prediction["onnx_label"]
        for result in results
        for prediction in result["predictions"]
    })
    per_label_max_abs_diff = aggregate_per_label(results, "per_label_max_abs_diff")
    worst_label, worst_label_diff = max(
        per_label_max_abs_diff.items(),
        key=lambda item: item[1],
    )
    lines.append(f"Observed entity families: {', '.join(observed_entity_types)}")
    lines.append(f"Observed BIO labels: {len(observed_labels)}")
    lines.append(f"Compared label channels: {len(per_label_max_abs_diff)}")
    lines.append(f"Worst label-channel max diff: {worst_label}={worst_label_diff:.6f}")
    lines.append("")

    if failures:
        lines.append("Failures:")
        for failure in failures:
            lines.append(f"  - {failure}")
    else:
        lines.append("All parity checks passed.")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir
    onnx_path = args.onnx_path or model_dir / "model_q4.onnx"
    coreml_path = args.coreml_path
    expected_shape = (1, args.sequence_length, 35)

    config = load_config(model_dir)
    id2label = {int(index): label for index, label in config["id2label"].items()}
    tokenizer = load_tokenizer(model_dir)

    if not onnx_path.exists():
        raise FileNotFoundError(f"Missing ONNX model: {onnx_path}")
    if not coreml_path.exists():
        raise FileNotFoundError(f"Missing Core ML model: {coreml_path}")

    onnx_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    coreml_model = ct.models.MLModel(str(coreml_path))

    results = []
    observed_entity_types: set[str] = set()
    observed_labels: set[str] = set()
    for fixture in FIXTURES:
        encoded = encode_fixture(
            tokenizer,
            fixture,
            args.sequence_length,
            pad_token_id=int(config["pad_token_id"]),
        )
        onnx_logits = run_onnx(onnx_session, encoded)
        coreml_logits = run_coreml(coreml_model, encoded)
        result = validate_fixture(
            fixture,
            encoded,
            onnx_logits,
            coreml_logits,
            id2label,
            expected_shape,
        )
        for prediction in result["predictions"]:
            observed_entity_types.add(entity_type(prediction["onnx_label"]))
            observed_labels.add(prediction["onnx_label"])
        results.append(result)

    failures = [
        f"{result['name']}: {failure}"
        for result in results
        for failure in result["failures"]
    ]
    required_entity_types = all_model_entity_types(id2label)
    required_labels = all_model_bio_labels(id2label)
    missing_entities = sorted(required_entity_types - observed_entity_types)
    if missing_entities:
        failures.append(
            "semantic coverage missing expected entity type(s): "
            + ", ".join(missing_entities)
        )
    missing_labels = sorted(required_labels - observed_labels)
    per_label_max_abs_diff = aggregate_per_label(results, "per_label_max_abs_diff")
    per_label_mean_abs_diff = aggregate_per_label(results, "per_label_mean_abs_diff")

    payload = {
        "passed": not failures,
        "thresholds": {
            "max_abs_diff": MAX_ABS_DIFF_THRESHOLD,
            "mean_abs_diff": MEAN_ABS_DIFF_THRESHOLD,
        },
        "required_entity_types": sorted(required_entity_types),
        "observed_entity_types": sorted(observed_entity_types),
        "required_bio_labels": sorted(required_labels),
        "observed_bio_labels": sorted(observed_labels),
        "missing_argmax_bio_labels": missing_labels,
        "missing_bio_labels": missing_labels,
        "compared_label_channels": [id2label[index] for index in sorted(id2label)],
        "per_label_max_abs_diff": per_label_max_abs_diff,
        "per_label_mean_abs_diff": per_label_mean_abs_diff,
        "failures": failures,
        "results": results,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(format_human_report(results, failures))

    return 0 if not failures else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
