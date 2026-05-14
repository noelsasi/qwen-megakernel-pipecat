"""
Phase D.2 — Compatibility check: megakernel hardcoded constants vs Qwen3-TTS talker config.

Run AFTER phase_a_inspect_model.py has confirmed the model structure.
Run from the project root (where qwen_megakernel/ is cloned).

Usage:
    python scripts/phase_d_compat_check.py 2>&1 | tee compat_output.txt
"""

import json
import re
import os
import sys

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
KERNEL_PATH = "qwen_megakernel/csrc/kernel.cu"

# Confirmed values from phase_a_inspect_model.py (2026-05-14):
# num_hidden_layers: 28        → megakernel NUM_LAYERS=28 matches
# num_attention_heads: 16      → megakernel NUM_HEADS=32 MISMATCH — must change to 16
# num_key_value_heads: 8       → megakernel NUM_KV_HEADS=8 matches
# hidden_size (talker): 1024   → megakernel HIDDEN_SIZE=1024 matches (from lm_head [2048,1024])
# intermediate_size: 3072      → megakernel INTERMEDIATE_SIZE=3072 matches
# vocab_size: 3072             → megakernel VOCAB_SIZE=151936 MISMATCH — must change to 3072
# max_position_embeddings: 32768 → megakernel MAX_SEQ_LEN=2048 MISMATCH — increase
# rope_theta: 1000000          → megakernel must match
# rope_scaling: interleaved MRope [24,20,20] — megakernel may not support MRope (critical risk)
# num_code_groups: 16          → 16 codebooks in code predictor
# position_id_per_seconds: 13  → ~13 codec frames/sec → ~77ms audio per frame
# text_hidden_size: 2048       → text encoder projection into 1024 talker hidden


def extract_kernel_constants(kernel_path: str) -> dict:
    """Extract #define constants from kernel.cu."""
    if not os.path.exists(kernel_path):
        print(f"ERROR: {kernel_path} not found. Run: git clone https://github.com/AlpinDale/qwen_megakernel")
        sys.exit(1)

    with open(kernel_path) as f:
        source = f.read()

    pattern = r"#define\s+([A-Z_]+)\s+(\d+)"
    constants = {}
    for name, val in re.findall(pattern, source):
        constants[name] = int(val)

    return constants


def extract_model_config() -> dict:
    """Download config.json and extract relevant architecture values."""
    from huggingface_hub import hf_hub_download

    config_values = {}

    for config_name in ["config.json", "talker_config.json"]:
        try:
            path = hf_hub_download(MODEL_ID, config_name)
            with open(path) as f:
                data = json.load(f)
            print(f"\n{config_name} content:")
            print(json.dumps(data, indent=2))

            # Extract known fields
            for field in ["num_hidden_layers", "hidden_size", "intermediate_size",
                          "num_attention_heads", "num_key_value_heads", "head_dim",
                          "vocab_size", "max_position_embeddings", "rope_theta",
                          "hidden_act", "rms_norm_eps"]:
                if field in data:
                    config_values[field] = data[field]

            # Look inside nested configs
            for nested_key in ["talker_config", "decoder_config"]:
                if nested_key in data:
                    for field in ["num_hidden_layers", "hidden_size", "intermediate_size",
                                  "num_attention_heads", "num_key_value_heads", "vocab_size"]:
                        if field in data[nested_key]:
                            config_values[f"{nested_key}.{field}"] = data[nested_key][field]

        except Exception as e:
            print(f"  {config_name}: {e}")

    return config_values


def print_compatibility_matrix(kernel: dict, model: dict):
    """Print a comparison table."""
    print("\n" + "=" * 80)
    print("COMPATIBILITY MATRIX")
    print("=" * 80)
    print(f"{'Parameter':<30} {'Kernel (kernel.cu)':<22} {'Model config':<22} {'Match?'}")
    print("-" * 80)

    comparisons = [
        ("NUM_LAYERS",         "num_hidden_layers"),
        ("HIDDEN_SIZE",        "hidden_size"),
        ("INTERMEDIATE_SIZE",  "intermediate_size"),
        ("NUM_HEADS",          "num_attention_heads"),
        ("NUM_KV_HEADS",       "num_key_value_heads"),
        ("HEAD_DIM",           "head_dim"),
        ("VOCAB_SIZE",         "vocab_size"),
        ("MAX_SEQ_LEN",        "max_position_embeddings"),
    ]

    all_match = True
    for kernel_key, model_key in comparisons:
        kval = kernel.get(kernel_key, "???")
        mval = model.get(model_key, "???")
        if kval == "???" or mval == "???":
            match = "UNKNOWN"
        elif kval == mval:
            match = "OK"
        else:
            match = f"MISMATCH — change to {mval}"
            all_match = False
        print(f"  {kernel_key:<28} {str(kval):<22} {str(mval):<22} {match}")

    print()
    if all_match:
        print("All known constants match — safe to proceed with Phase D integration.")
    else:
        print("MISMATCHES FOUND — update kernel.cu #defines before building.")
        print("Edit: qwen_megakernel/csrc/kernel.cu")

    return all_match


def main():
    print("Extracting kernel constants ...")
    kernel = extract_kernel_constants(KERNEL_PATH)
    print("Kernel constants:")
    for k, v in sorted(kernel.items()):
        print(f"  {k} = {v}")

    print("\nFetching model config ...")
    model_cfg = extract_model_config()

    print_compatibility_matrix(kernel, model_cfg)

    print("\n" + "=" * 60)
    print("Next: if there are mismatches, update kernel.cu and rebuild.")
    print("Then proceed to scripts/phase_d_weight_extraction.py (TODO).")


if __name__ == "__main__":
    main()
