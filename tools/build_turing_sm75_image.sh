#!/usr/bin/env bash

set -euo pipefail

CUDA_VERSION="${CUDA_VERSION:-12.9.1}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5}"
MAX_JOBS="${MAX_JOBS:-32}"
NVCC_THREADS="${NVCC_THREADS:-1}"
IMAGE_TAG="${IMAGE_TAG:-local/vllm-gemma4:sm7.5}"
BUILD_TARGET="${BUILD_TARGET:-vllm-openai}"
LOG_FILE="${LOG_FILE:-docker-build-sm75-$(date +%F-%H%M%S).log}"

TMP_DOCKERFILE="$(mktemp /tmp/vllm-dockerfile.XXXXXX)"
cleanup() {
    rm -f "$TMP_DOCKERFILE"
}
trap cleanup EXIT

awk '
/&& flashinfer show-config \\$/ {
  sub(/ \\$/, "")
  print
  next
}
/&& flashinfer download-cubin$/ {
  next
}
{ print }
' docker/Dockerfile > "$TMP_DOCKERFILE"

echo "Using temporary Dockerfile: $TMP_DOCKERFILE"
echo "Build log: $LOG_FILE"
echo "Image tag: $IMAGE_TAG"

grep -n -A4 -B2 "FLASHINFER_VERSION" "$TMP_DOCKERFILE"

time DOCKER_BUILDKIT=1 docker build --progress=plain \
  -f "$TMP_DOCKERFILE" \
  --target "$BUILD_TARGET" \
  --build-arg CUDA_VERSION="$CUDA_VERSION" \
  --build-arg torch_cuda_arch_list="$TORCH_CUDA_ARCH_LIST" \
  --build-arg max_jobs="$MAX_JOBS" \
  --build-arg nvcc_threads="$NVCC_THREADS" \
  -t "$IMAGE_TAG" . \
  2>&1 | tee "$LOG_FILE"
