.PHONY: clean uv-sync

uv-clean:
	rm -rf uv.lock ~/.cache/uv .venv

uv-sync:
	uv sync --no-cache --no-build-isolation --extra build
