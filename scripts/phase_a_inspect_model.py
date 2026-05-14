"""
Phase A.2 / A.3 — Discover Qwen3-TTS package API and inspect model structure.

Run this FIRST on the GPU server before writing any other code.
Captures ground truth: class names, module paths, config values, weight shapes.

Usage:
    python scripts/phase_a_inspect_model.py 2>&1 | tee inspect_output.txt
"""

import json
import sys

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"


def step1_list_repo_files():
    """List all files in the HF repo to find modeling_*.py and README."""
    print("=" * 60)
    print("STEP 1: Repo file listing")
    print("=" * 60)
    from huggingface_hub import list_repo_files
    files = sorted(list_repo_files(MODEL_ID))
    for f in files:
        print(f)
    return files


def step2_inspect_config():
    """Download and print config.json (or any talker config)."""
    print("\n" + "=" * 60)
    print("STEP 2: Config files")
    print("=" * 60)
    from huggingface_hub import hf_hub_download

    for config_name in ["config.json", "talker_config.json", "generation_config.json"]:
        try:
            path = hf_hub_download(MODEL_ID, config_name)
            with open(path) as f:
                data = json.load(f)
            print(f"\n--- {config_name} ---")
            print(json.dumps(data, indent=2))
        except Exception as e:
            print(f"  {config_name}: not found ({e})")


def step3_load_and_inspect():
    """Load model on CPU and print full module hierarchy + parameter shapes."""
    print("\n" + "=" * 60)
    print("STEP 3: Model load + inspection (CPU, bfloat16)")
    print("=" * 60)

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel

    # qwen3_tts requires transformers installed from source (not any released version).
    # If this fails with "does not recognize this architecture", run:
    #   pip install git+https://github.com/huggingface/transformers.git
    print("Attempting: AutoModel.from_pretrained (trust_remote_code=True) ...")
    try:
        model = AutoModel.from_pretrained(
            MODEL_ID,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        print(f"Loaded as: {type(model).__name__}")
    except Exception as e:
        print(f"AutoModel failed: {e}")
        print("Trying Qwen3TtsForConditionalGeneration directly ...")
        try:
            from transformers import Qwen3TtsForConditionalGeneration, Qwen3TtsProcessor
            model = Qwen3TtsForConditionalGeneration.from_pretrained(
                MODEL_ID,
                device_map="cpu",
                torch_dtype=torch.bfloat16,
            )
            print(f"Loaded as: {type(model).__name__}")
        except Exception as e2:
            print(f"Qwen3TtsForConditionalGeneration also failed: {e2}")
            print("Trying AutoModelForCausalLM ...")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID,
                    device_map="cpu",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
                print(f"Loaded as: {type(model).__name__}")
            except Exception as e3:
                print(f"All load attempts failed: {e3}")
                print("STOP: Install transformers from source: pip install git+https://github.com/huggingface/transformers.git")
                sys.exit(1)

    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        print(f"Processor loaded: {type(processor).__name__}")
    except Exception as e:
        print(f"AutoProcessor failed: {e}")
        try:
            from transformers import Qwen3TtsProcessor
            processor = Qwen3TtsProcessor.from_pretrained(MODEL_ID)
            print(f"Processor loaded via Qwen3TtsProcessor: {type(processor).__name__}")
        except Exception as e2:
            print(f"Qwen3TtsProcessor also failed: {e2}")
            processor = None

    print("\n--- NAMED MODULES ---")
    for name, module in model.named_modules():
        print(f"  {name}: {type(module).__name__}")

    print("\n--- PARAMETER SHAPES (talker/decoder layers only) ---")
    for name, param in model.named_parameters():
        print(f"  {name}: {list(param.shape)} {param.dtype}")

    print("\n--- TOP-LEVEL ATTRIBUTES ---")
    for attr in dir(model):
        if not attr.startswith("_"):
            val = getattr(model, attr, None)
            if hasattr(val, "__class__") and "Module" in type(val).__name__:
                print(f"  model.{attr}: {type(val).__name__}")

    print("\n--- generate() SIGNATURE ---")
    import inspect
    try:
        sig = inspect.signature(model.generate)
        print(f"  model.generate{sig}")
    except Exception:
        print("  No generate() method or can't inspect signature")

    print("\n--- MODEL CONFIG ---")
    try:
        print(model.config)
    except Exception:
        print("  (no config attribute)")

    return model


if __name__ == "__main__":
    print(f"Inspecting: {MODEL_ID}")
    print("Output is saved to inspect_output.txt if you ran with: ... | tee inspect_output.txt\n")

    files = step1_list_repo_files()
    step2_inspect_config()
    step3_load_and_inspect()

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)
    print("""
Next steps (fill these in from the output above):
  - Actual class name: ???
  - Talker module path (e.g. model.talker): ???
  - Number of talker layers: ???
  - Hidden size: ???
  - Intermediate size: ???
  - Num heads / KV heads: ???
  - Vocab size at talker output: ???
  - generate() streamer parameter: ???
  - Code predictor module path: ???
  - Vocoder/DAC module path: ???
""")
