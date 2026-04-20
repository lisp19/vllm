# Turing Build Helper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated helper script that builds the local Turing `sm7.5` Docker image without modifying the shared Dockerfile.

**Architecture:** Add one standalone shell script under `tools/` that generates a temporary Dockerfile, removes the FlashInfer cubin pre-download step, and runs the existing local Docker build with fixed Turing arguments. Keep all behavior isolated to the script so branch-specific build ergonomics do not conflict with upstream files.

**Tech Stack:** Bash, awk, docker build with BuildKit

---

## Task 1: Add the Turing build helper script

**Files:**

- Create: `tools/build_turing_sm75_image.sh`
- Reference: `docker/Dockerfile`

- [ ] Step 1: Create a standalone helper script that generates a temporary Dockerfile and runs the local Turing build command.
- [ ] Step 2: Ensure the script uses `set -euo pipefail`, cleans up the temp file with `trap`, and logs output with `tee`.
- [ ] Step 3: Use `awk` to remove `flashinfer download-cubin` while preserving `flashinfer show-config`.

## Task 2: Verify the helper script

**Files:**

- Verify: `tools/build_turing_sm75_image.sh`

- [ ] Step 1: Run `bash -n tools/build_turing_sm75_image.sh`.
- [ ] Step 2: Run `pre-commit run --files tools/build_turing_sm75_image.sh`.
- [ ] Step 3: Commit the script with a focused message.
