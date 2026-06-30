#!/usr/bin/env python3
"""Convert nationaldesignstudio/rampart's quantized ONNX artifact to Core ML.

The released Rampart artifact is an ONNX Runtime Web model with Microsoft
MatMulNBits Q4 weight-only quantization. Core ML Tools cannot import that graph
directly, so this script reconstructs a compatible BERT token-classification
module in PyTorch, unpacks the Q4/INT8 weights, verifies against ONNX Runtime,
and converts the traced PyTorch module to a Core ML ML Program.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import urllib.request
from pathlib import Path

import coremltools as ct
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper
from torch import nn


MODEL_ID = "nationaldesignstudio/rampart"
HF_RESOLVE = f"https://huggingface.co/{MODEL_ID}/resolve/main"
MODEL_FILES = {
    "onnx/model_q4.onnx": "model_q4.onnx",
    "config.json": "config.json",
    "tokenizer.json": "tokenizer.json",
    "tokenizer_config.json": "tokenizer_config.json",
    "special_tokens_map.json": "special_tokens_map.json",
    "vocab.txt": "vocab.txt",
}


class BertEmbeddingsLite(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config["vocab_size"], config["hidden_size"], padding_idx=config["pad_token_id"]
        )
        self.position_embeddings = nn.Embedding(
            config["max_position_embeddings"], config["hidden_size"]
        )
        self.token_type_embeddings = nn.Embedding(
            config["type_vocab_size"], config["hidden_size"]
        )
        self.LayerNorm = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand_as(input_ids)

        embeddings = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        embeddings = self.LayerNorm(embeddings)
        return self.dropout(embeddings)


class BertSelfAttentionLite(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = config["hidden_size"] // config["num_attention_heads"]
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config["hidden_size"], self.all_head_size)
        self.key = nn.Linear(config["hidden_size"], self.all_head_size)
        self.value = nn.Linear(config["hidden_size"], self.all_head_size)
        self.dropout = nn.Dropout(config["attention_probs_dropout_prob"])

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:-2] + (self.all_head_size,)
        return context_layer.view(new_context_shape)


class BertSelfOutputLite(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.dense = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.LayerNorm = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


class BertAttentionLite(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.self = BertSelfAttentionLite(config)
        self.output = BertSelfOutputLite(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        self_output = self.self(hidden_states, attention_mask)
        return self.output(self_output, hidden_states)


class BertIntermediateLite(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.dense = nn.Linear(config["hidden_size"], config["intermediate_size"])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.dense(hidden_states), approximate="none")


class BertOutputLite(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.dense = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.LayerNorm = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


class BertLayerLite(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.attention = BertAttentionLite(config)
        self.intermediate = BertIntermediateLite(config)
        self.output = BertOutputLite(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return self.output(intermediate_output, attention_output)


class BertEncoderLite(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.layer = nn.ModuleList(
            [BertLayerLite(config) for _ in range(config["num_hidden_layers"])]
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


class BertModelLite(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.embeddings = BertEmbeddingsLite(config)
        self.encoder = BertEncoderLite(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(input_ids, token_type_ids)
        extended_attention_mask = attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return self.encoder(hidden_states, extended_attention_mask)


class RampartTokenClassifierLite(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.bert = BertModelLite(config)
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.classifier = nn.Linear(config["hidden_size"], len(config["id2label"]))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.long()
        attention_mask = attention_mask.long()
        token_type_ids = token_type_ids.long()

        sequence_output = self.bert(input_ids, attention_mask, token_type_ids)
        sequence_output = self.dropout(sequence_output)
        return self.classifier(sequence_output)


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, destination)


def ensure_model_files(model_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for remote_path, local_name in MODEL_FILES.items():
        destination = model_dir / local_name
        download_file(f"{HF_RESOLVE}/{remote_path}", destination)
        files[remote_path] = destination
    return files


def unpack_q4_weight(qweight: np.ndarray, scales: np.ndarray, k: int, n: int) -> np.ndarray:
    low = qweight & 0x0F
    high = (qweight >> 4) & 0x0F
    unpacked = np.stack([low, high], axis=-1).reshape(n, qweight.shape[1], -1)
    unpacked = unpacked[:, :, : qweight.shape[2] * 2].astype(np.float32)
    weight = (unpacked - 8.0) * scales[..., None]
    return weight.reshape(n, -1)[:, :k]


def dequant_embedding(initializers: dict[str, np.ndarray], name: str) -> np.ndarray:
    quantized = initializers[f"{name}.weight_quantized"].astype(np.float32)
    scale = initializers[f"{name}.weight_scale"].astype(np.float32)
    zero_point = initializers[f"{name}.weight_zero_point"].astype(np.float32)
    return (quantized - zero_point) * scale


def load_state_dict_from_onnx(onnx_path: Path) -> dict[str, torch.Tensor]:
    model = onnx.load(onnx_path)
    initializers = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    state: dict[str, torch.Tensor] = {}

    for name, value in initializers.items():
        if name.endswith(("_scale", "_zero_point", "_quantized", "_Q4", "_scales")):
            continue
        state[name] = torch.from_numpy(np.array(value))

    for embedding_name in (
        "bert.embeddings.word_embeddings",
        "bert.embeddings.token_type_embeddings",
        "bert.embeddings.position_embeddings",
    ):
        state[f"{embedding_name}.weight"] = torch.from_numpy(
            dequant_embedding(initializers, embedding_name)
        )

    for node in model.graph.node:
        if node.domain != "com.microsoft" or node.op_type != "MatMulNBits":
            continue

        attrs = {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}
        module_name = node.name.strip("/").removesuffix("/MatMul_Q4").replace("/", ".")
        qweight = initializers[node.input[1]]
        scales = initializers[node.input[2]]
        state[f"{module_name}.weight"] = torch.from_numpy(
            unpack_q4_weight(qweight, scales, attrs["K"], attrs["N"])
        )

    return state


def verify_against_onnx(
    torch_model: nn.Module,
    onnx_path: Path,
    sequence_length: int,
    vocab_size: int,
    atol: float,
) -> None:
    rng = np.random.default_rng(1480)
    input_ids = rng.integers(0, vocab_size, size=(1, sequence_length), dtype=np.int64)
    input_ids[:, 0] = 101
    input_ids[:, -1] = 102
    attention_mask = np.ones((1, sequence_length), dtype=np.int64)
    token_type_ids = np.zeros((1, sequence_length), dtype=np.int64)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (onnx_logits,) = session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )

    with torch.no_grad():
        torch_logits = torch_model(
            torch.from_numpy(input_ids),
            torch.from_numpy(attention_mask),
            torch.from_numpy(token_type_ids),
        ).numpy()

    max_diff = float(np.max(np.abs(onnx_logits - torch_logits)))
    mean_diff = float(np.mean(np.abs(onnx_logits - torch_logits)))
    print(f"ONNX/PyTorch verification: max_abs_diff={max_diff:.6g}, mean_abs_diff={mean_diff:.6g}")
    if max_diff > atol:
        raise RuntimeError(
            f"PyTorch reconstruction differs from ONNX by {max_diff:.6g}, "
            f"which exceeds --verify-atol={atol}"
        )


def convert_to_coreml(
    torch_model: nn.Module,
    output_path: Path,
    sequence_length: int,
    minimum_deployment_target: ct.target,
    precision: str,
) -> None:
    example_inputs = (
        torch.zeros((1, sequence_length), dtype=torch.int32),
        torch.ones((1, sequence_length), dtype=torch.int32),
        torch.zeros((1, sequence_length), dtype=torch.int32),
    )
    traced = torch.jit.trace(torch_model, example_inputs)

    mlmodel = ct.convert(
        traced,
        source="pytorch",
        convert_to="mlprogram",
        minimum_deployment_target=minimum_deployment_target,
        compute_precision=ct.precision.FLOAT16 if precision == "float16" else ct.precision.FLOAT32,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, sequence_length), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, sequence_length), dtype=np.int32),
            ct.TensorType(name="token_type_ids", shape=(1, sequence_length), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
    )
    mlmodel.short_description = "Rampart PII token classifier converted from ONNX Q4."
    mlmodel.input_description["input_ids"] = "WordPiece token ids padded to the fixed sequence length."
    mlmodel.input_description["attention_mask"] = "1 for real tokens, 0 for padding."
    mlmodel.input_description["token_type_ids"] = "Segment ids, usually all zeros."
    mlmodel.output_description["logits"] = "Per-token logits for the 35 BIO PII labels."

    if output_path.exists():
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    mlmodel.save(str(output_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/rampart-hf"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/RampartTokenClassifier.mlpackage"))
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--precision", choices=("float16", "float32"), default="float32")
    parser.add_argument("--verify-atol", type=float, default=1e-3)
    parser.add_argument("--skip-coreml", action="store_true", help="Only rebuild and verify the PyTorch model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = ensure_model_files(args.model_dir)

    with files["config.json"].open() as f:
        config = json.load(f)

    torch.manual_seed(0)
    torch_model = RampartTokenClassifierLite(config).eval()
    state = load_state_dict_from_onnx(files["onnx/model_q4.onnx"])
    missing, unexpected = torch_model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch: missing={missing}, unexpected={unexpected}")

    verify_against_onnx(
        torch_model,
        files["onnx/model_q4.onnx"],
        args.sequence_length,
        config["vocab_size"],
        args.verify_atol,
    )

    if args.skip_coreml:
        return

    convert_to_coreml(
        torch_model,
        args.output,
        args.sequence_length,
        minimum_deployment_target=ct.target.iOS16,
        precision=args.precision,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
