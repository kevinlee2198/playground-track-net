#!/usr/bin/env bash
# Run a command under a memory cap: if it grows past the cap, the kernel kills
# it alone. Without a cap, running out of memory shuts down the whole WSL
# session. The default cap is 8G; set CAP to change it:
#   scripts/capped.sh uv run pytest tests/ -q
#   CAP=12G scripts/capped.sh uv run python main.py infer ...
# MemorySwapMax=0 keeps the job from spilling into swap instead of being killed.
exec systemd-run --user --scope -q -p MemoryMax="${CAP:-8G}" -p MemorySwapMax=0 "$@"
