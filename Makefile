.PHONY: venv install inspect baseline streaming server benchmark kernel

VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

venv:
	python3 -m venv $(VENV)
	@echo "Activate with: source .venv/bin/activate"

install: venv
	# Install PyTorch with CUDA 12.8 FIRST (RTX 5090 requires cu128 build)
	$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
	$(PIP) install -r requirements.txt

# Phase A: inspect model structure, print all module names + shapes
inspect:
	$(PYTHON) scripts/phase_a_inspect_model.py

# Phase A: baseline TTS to WAV file, measure RTF
baseline:
	$(PYTHON) scripts/phase_a_baseline.py

# Phase B: test streaming feasibility
streaming:
	$(PYTHON) scripts/phase_b_streaming_probe.py

# Phase D: diff megakernel constants vs model config
compat:
	$(PYTHON) scripts/phase_d_compat_check.py

# Build megakernel (run on GPU server, requires nvcc + CUDA 12.8)
kernel:
	cd qwen_megakernel && $(PYTHON) setup.py build_ext --inplace

# Start FastAPI server (run on GPU server)
server:
	$(PYTHON) -m uvicorn server.pipeline.voice_agent:app \
		--host 0.0.0.0 \
		--port 8000 \
		--reload

# Run full benchmark suite
benchmark:
	$(PYTHON) scripts/benchmark.py
