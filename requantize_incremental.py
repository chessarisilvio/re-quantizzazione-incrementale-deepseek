#!/usr/bin/env python3
"""
Incremental requantization script for GGUF models.
Loads a model, requantizes only specified layers to Q4_XS, saves new model.
"""

import os
import json
import sys
from typing import List

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFQuantizationType, dequantize, quantize


def load_layers_to_requantize(json_path: str) -> List[str]:
    """Load list of tensor names to requantize from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Expect JSON to be a list of strings
    if isinstance(data, list):
        return [str(item) for item in data]
    elif isinstance(data, dict) and 'layers' in data:
        return [str(item) for item in data['layers']]
    else:
        raise ValueError("Unexpected JSON format. Expected list or dict with 'layers' key.")


def requantize_incremental(
    input_model_path: str,
    output_model_path: str,
    layers_json_path: str
) -> None:
    """Requantize only specified layers to Q4_XS."""
    layers_to_requantize = set(load_layers_to_requantize(layers_json_path))
    print(f"Loaded {len(layers_to_requantize)} layers to requantize from {layers_json_path}")

    # Reader
    reader = GGUFReader(input_model_path)
    print(f"Read model: {input_model_path}")
    print(f"  tensors: {len(reader.tensors)}")

    # Writer with same metadata
    writer = GGUFWriter(
        path=output_model_path,
        arch=reader.arch,
        endian=reader.endian
    )

    # Copy all key-value pairs from reader to writer
    for key, value in reader.fields.items():
        writer.add_field(key, value)

    # Process each tensor
    for tensor in reader.tensors:
        name = tensor.name
        if name in layers_to_requantize:
            print(f"Requantizing tensor: {name}")
            # Dequantize to float32 numpy array
            data = dequantize(tensor.data, tensor.type)
            # Requante to Q4_XS
            new_data = quantize(data, GGUFQuantizationType.Q4_XS)
            writer.add_tensor(name, new_data, raw_dtype=GGUFQuantizationType.Q4_XS)
        else:
            # Keep original tensor data and type
            writer.add_tensor(name, tensor.data, raw_dtype=tensor.type)

    # Write header and tensors
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"Successfully written requantized model to: {output_model_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Incrementally requantize GGUF model layers to Q4_XS"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to input GGUF model"
    )
    parser.add_argument(
        "--layers-json",
        required=True,
        help="Path to JSON file listing layers to requantize"
    )
    parser.add_argument(
        "--output",
        help="Path for output GGUF model (default: input with .inc.q4_xs suffix)"
    )
    args = parser.parse_args()

    input_model = args.model
    layers_json = args.layers_json
    if args.output:
        output_model = args.output
    else:
        base, ext = os.path.splitext(input_model)
        output_model = f"{base}.inc.q4_xs{ext}"

    requantize_incremental(input_model, output_model, layers_json)


if __name__ == "__main__":
    main()