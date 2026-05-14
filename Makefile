.PHONY: setup server benchmark client

PYTHON = .venv/bin/python

# GPU server: full setup (run once after git clone)
setup:
	bash scripts/setup_server.sh

# Start the voice agent server (megakernel backend)
server:
	source .venv/bin/activate && \
	uvicorn server.pipeline.voice_agent:app --host 0.0.0.0 --port 8000

# Run benchmark (requires server already loaded model, or run standalone)
benchmark:
	$(PYTHON) scripts/benchmark.py --backend megakernel

# Frontend dev server (local machine, not GPU server)
client:
	cd client && npm run dev
