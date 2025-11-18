"""
CUDA Environment Setup - MUST be imported BEFORE torch/pyannote
Sets up optimal memory management for RTX 6000 Blackwell (96GB VRAM)
"""

import os
import sys


def setup_cuda_environment():
    """
    Configure CUDA environment variables BEFORE importing torch.
    This prevents std::bad_alloc errors during initial model loading.
    """
    # Memory allocator configuration for 96GB VRAM
    # - expandable_segments: Reduce fragmentation
    # - max_split_size_mb: Larger chunks for big models (512MB is good for 96GB)
    # - garbage_collection_threshold: More aggressive GC
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.8"

    # Async execution for better performance
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    # Optimize for WSL if detected
    if "microsoft" in os.uname().release.lower() or "wsl" in os.uname().release.lower():
        # Prevent glibc memory fragmentation in WSL
        os.environ["MALLOC_TRIM_THRESHOLD_"] = "100000"

        # Use more efficient malloc arena for multi-threaded apps
        os.environ["MALLOC_ARENA_MAX"] = "4"

    # Hugging Face optimizations
    # Reduce parallel downloads to avoid memory spikes during model loading
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"

    # Disable tokenizers parallelism to avoid memory issues
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # OMP threads for CPU operations (adjust based on your CPU)
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = "8"

    # Print confirmation
    print("[CUDA Setup] Environment configured for RTX 6000 Blackwell (96GB VRAM)")
    print(f"[CUDA Setup] PYTORCH_CUDA_ALLOC_CONF: {os.environ['PYTORCH_CUDA_ALLOC_CONF']}")

    # Check if WSL
    if "MALLOC_TRIM_THRESHOLD_" in os.environ:
        print("[CUDA Setup] WSL optimizations enabled")


# Run setup immediately on import
setup_cuda_environment()
