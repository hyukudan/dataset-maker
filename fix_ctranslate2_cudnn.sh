#!/bin/bash
# Fix ctranslate2 cuDNN compatibility for CUDA
# This script ensures ctranslate2 uses the full cuDNN from nvidia-cudnn-cu12
# instead of its incomplete bundled version

echo "Fixing ctranslate2 cuDNN compatibility..."

VENV_DIR=".venv"
CTRANSLATE_LIBS="$VENV_DIR/lib/python3.11/site-packages/ctranslate2.libs"
NVIDIA_CUDNN="$VENV_DIR/lib/python3.11/site-packages/nvidia/cudnn/lib"

if [ ! -d "$CTRANSLATE_LIBS" ]; then
    echo "Error: ctranslate2.libs directory not found"
    exit 1
fi

if [ ! -d "$NVIDIA_CUDNN" ]; then
    echo "Error: nvidia-cudnn-cu12 not installed"
    echo "Run: uv add nvidia-cudnn-cu12>=9.1.0"
    exit 1
fi

cd "$CTRANSLATE_LIBS"

# Backup original if it exists and is not a symlink
if [ -f "libcudnn-74a4c495.so.9.1.0" ] && [ ! -L "libcudnn-74a4c495.so.9.1.0" ]; then
    echo "Backing up original ctranslate2 cuDNN stub..."
    mv libcudnn-74a4c495.so.9.1.0 libcudnn-74a4c495.so.9.1.0.ORIGINAL
fi

# Create symlink to full cuDNN
echo "Creating symlink to nvidia-cudnn-cu12..."
ln -sf ../nvidia/cudnn/lib/libcudnn.so.9 libcudnn-74a4c495.so.9.1.0

# Verify
if [ -L "libcudnn-74a4c495.so.9.1.0" ]; then
    echo "✓ Fix applied successfully!"
    echo "  libcudnn-74a4c495.so.9.1.0 -> $(readlink libcudnn-74a4c495.so.9.1.0)"
else
    echo "✗ Failed to create symlink"
    exit 1
fi

# Also create symlinks for cuDNN ops if needed
cd "$(dirname $0)/$NVIDIA_CUDNN"
if [ ! -L "libcudnn_ops.so.9.1.0" ]; then
    echo "Creating cuDNN ops version symlinks..."
    ln -sf libcudnn_ops.so.9 libcudnn_ops.so.9.1.0
    ln -sf libcudnn_ops.so.9 libcudnn_ops.so.9.1
    ln -sf libcudnn_ops.so.9 libcudnn_ops.so
    echo "✓ Created cuDNN ops symlinks"
fi

echo ""
echo "========================================="
echo "✅ ctranslate2 cuDNN fix complete!"
echo "========================================="
echo ""
echo "WhisperX can now use CUDA for transcription"
echo "Run test_whisperx_cuda_full.py to verify"
