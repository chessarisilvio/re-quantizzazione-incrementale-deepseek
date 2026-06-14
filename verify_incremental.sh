#!/usr/bin/env bash
# Verification script for incremental requantization
# Compares original and requantized GGUF models layer by layer

# Exit on any error
set -e

# Default log file
LOG_FILE="verification.log"

# Function to print usage
usage() {
    echo "Usage: $0 --original <original_model> --requantized <requantized_model> --layers-json <layers_json> [--log <log_file>]"
    echo "  --original        Path to original GGUF model"
    echo "  --requantized     Path to requantized GGUF model"
    echo "  --layers-json     Path to JSON file listing layers that were requantized"
    echo "  --log             Optional: Path to log file (default: verification.log)"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --original)
            ORIGINAL_MODEL="$2"
            shift 2
            ;;
        --requantized)
            REQUIANTIZED_MODEL="$2"
            shift 2
            ;;
        --layers-json)
            LAYERS_JSON="$2"
            shift 2
            ;;
        --log)
            LOG_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check required arguments
if [[ -z "$ORIGINAL_MODEL" || -z "$REQUIANTIZED_MODEL" || -z "$LAYERS_JSON" ]]; then
    echo "Error: Missing required arguments"
    usage
fi

# Check files exist
if [[ ! -f "$ORIGINAL_MODEL" ]]; then
    echo "Error: Original model not found: $ORIGINAL_MODEL"
    exit 1
fi

if [[ ! -f "$REQUIANTIZED_MODEL" ]]; then
    echo "Error: Requantiized model not found: $REQUIANTIZED_MODEL"
    exit 1
fi

if [[ ! -f "$LAYERS_JSON" ]]; then
    echo "Error: Layers JSON not found: $LAYERS_JSON"
    exit 1
fi

# Clear log file
echo "Verification log for incremental requantization" > "$LOG_FILE"
echo "================================================" >> "$LOG_FILE"
echo "Original model: $ORIGINAL_MODEL" >> "$LOG_FILE"
echo "Requantized model: $REQUIANTIZED_MODEL" >> "$LOG_FILE"
echo "Layers JSON: $LAYERS_JSON" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Run Python verification
python3 << EOF >> "$LOG_FILE" 2>&1
import os
import json
import sys
from gguf import GGUFReader, GGUFQuantizationType, dequantize

# Get paths from environment
original_path = os.environ['ORIGINAL_MODEL']
requantized_path = os.environ['REQUIANTIZED_MODEL']
layers_json_path = os.environ['LAYERS_JSON']

# Load layers to requantize
with open(layers_json_path, 'r') as f:
    layers_data = json.load(f)
if isinstance(layers_data, list):
    layers_to_requantize = set(str(item) for item in layers_data)
elif isinstance(layers_data, dict) and 'layers' in layers_data:
    layers_to_requantize = set(str(item) for item in layers_data['layers'])
else:
    raise ValueError("Unexpected JSON format. Expected list or dict with 'layers' key.")

print(f"Loaded {len(layers_to_requantize)} layers to requantize from {layers_json_path}")

# Read models
print(f"Reading original model: {original_path}")
original_reader = GGUFReader(original_path)
print(f"  tensors: {len(original_reader.tensors)}")

print(f"Reading requantized model: {requantized_path}")
requantized_reader = GGUFReader(requantized_path)
print(f"  tensors: {len(requantized_reader.tensors)}")

# Build dict of tensors by name for requantized model for quick lookup
requantized_tensors = {t.name: t for t in requantized_reader.tensors}

# Track results
errors = []
warnings = []
info = []

# Check each tensor in original model
for orig_tensor in original_reader.tensors:
    name = orig_tensor.name
    if name in layers_to_requantize:
        # This tensor should have been requantized
        if name not in requantized_tensors:
            errors.append(f"Tensor '{name}' marked for requantization but missing in requantized model")
            continue
        req_tensor = requantized_tensors[name]
        # Check that the type is now Q4_XS
        if req_tensor.type != GGUFQuantizationType.Q4_XS:
            errors.append(f"Tensor '{name}' should be requantized to Q4_XS but is type {req_tensor.type}")
        else:
            info.append(f"Tensor '{name}' correctly requantized to Q4_XS")
        # Optional: check that data changed (not required but good to note)
        # We could dequantize both and compare, but skip for simplicity
    else:
        # This tensor should be unchanged
        if name not in requantized_tensors:
            errors.append(f"Tensor '{name}' not marked for requantization but missing in requantized model")
            continue
        req_tensor = requantized_tensors[name]
        # Check type matches
        if orig_tensor.type != req_tensor.type:
            errors.append(f"Tensor '{name}' type changed unexpectedly: {orig_tensor.type} -> {req_tensor.type}")
        # Check data matches byte-for-byte
        elif orig_tensor.data != req_tensor.data:
            errors.append(f"Tensor '{name}' data changed but should be unchanged")
        else:
            info.append(f"Tensor '{name}' unchanged as expected")

# Check for extra tensors in requantized model not in original
requantized_names = set(requantized_tensors.keys())
original_names = {t.name for t in original_reader.tensors}
extra_in_requantized = requantized_names - original_names
if extra_in_requantized:
    warnings.append(f"Extra tensors in requantized model not in original: {', '.join(sorted(extra_in_requantized))}")

# Check for missing tensors in requantized model (should have been caught above, but just in case)
missing_in_requantized = original_names - requantized_names
if missing_in_requantized:
    errors.append(f"Tensors missing in requantized model: {', '.join(sorted(missing_in_requantized))}")

# Summary
print("\\n=== VERIFICATION SUMMARY ===")
print(f"Total tensors in original model: {len(original_reader.tensors)}")
print(f"Total tensors in requantized model: {len(requantized_reader.tensors)}")
print(f"Layers to requantize: {len(layers_to_requantize)}")
print()
if info:
    print(f"INFO: {len(info)} checks passed")
    for msg in info[-5:]:  # Show last 5 info messages
        print(f"  {msg}")
    if len(info) > 5:
        print(f"  ... and {len(info) - 5} more")
print()
if warnings:
    print(f"WARNINGS: {len(warnings)}")
    for msg in warnings:
        print(f"  {msg}")
print()
if errors:
    print(f"ERRORS: {len(errors)}")
    for msg in errors:
        print(f"  {msg}")
    print("\\nVERIFICATION FAILED")
    sys.exit(1)
else:
    print("VERIFICATION PASSED: All checks passed")
    sys.exit(0)
EOF

# Check exit status of Python script
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "Verification script failed. See log for details: $LOG_FILE"
    exit 1
else
    echo "Verification completed successfully. See log: $LOG_FILE"
fi