"""
CUDA Environment Setup - MUST be imported BEFORE torch/pyannote
Sets up optimal memory management for RTX 6000 Blackwell (96GB VRAM)
"""

import os
import sys


def detect_gpus():
    """
    Detect available GPUs before importing torch.
    Returns list of GPU indices if nvidia-smi is available.
    """
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(',')
                    gpu_id = parts[0].strip()
                    name = parts[1].strip()
                    memory = parts[2].strip()
                    gpus.append({"id": gpu_id, "name": name, "memory": memory})
            return gpus
    except Exception:
        pass
    return []


def setup_cuda_environment():
    """
    Configure CUDA environment variables BEFORE importing torch.
    This prevents std::bad_alloc errors during initial model loading.
    """
    # Configure LD_LIBRARY_PATH for cuDNN libraries in virtual environment
    # This allows ctranslate2 to find cuDNN from nvidia-cudnn-cu12 package
    # Find site-packages by checking sys.prefix (works for uv/venv)
    import sysconfig
    site_packages = sysconfig.get_paths()["purelib"]

    cudnn_lib_path = os.path.join(site_packages, "nvidia", "cudnn", "lib")
    ctranslate2_lib_path = os.path.join(site_packages, "ctranslate2.libs")

    libs_to_add = []
    if os.path.exists(cudnn_lib_path):
        libs_to_add.append(cudnn_lib_path)
    if os.path.exists(ctranslate2_lib_path):
        libs_to_add.append(ctranslate2_lib_path)

    if libs_to_add:
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if current_ld_path:
            libs_to_add.append(current_ld_path)

        os.environ["LD_LIBRARY_PATH"] = ":".join(libs_to_add)
        print(f"[CUDA Setup] Added cuDNN libraries to LD_LIBRARY_PATH: {cudnn_lib_path}")

    # Detect GPUs before configuration
    gpus = detect_gpus()

    # Configure CUDA_VISIBLE_DEVICES if not already set
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        if len(gpus) > 1:
            # Multiple GPUs detected - log available options
            print(f"[CUDA Setup] Detected {len(gpus)} GPUs:")
            for gpu in gpus:
                print(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['memory']})")
            print(f"[CUDA Setup] Using GPU 0 by default. Set CUDA_VISIBLE_DEVICES to change.")
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        elif len(gpus) == 1:
            print(f"[CUDA Setup] Detected 1 GPU: {gpus[0]['name']} ({gpus[0]['memory']})")
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        visible = os.environ["CUDA_VISIBLE_DEVICES"]
        print(f"[CUDA Setup] CUDA_VISIBLE_DEVICES already set to: {visible}")

    # Memory allocator configuration
    # Detect VRAM size to optimize chunk size
    # Note: This is a heuristic based on typical GPU memory (detection before torch import)
    total_memory_str = gpus[0]["memory"] if gpus else "0 MiB"

    # Parse memory (format: "98304 MiB" or similar)
    try:
        memory_mb = int(total_memory_str.split()[0])
        vram_gb = memory_mb / 1024
    except:
        vram_gb = 48  # Default assumption

    # Optimize based on VRAM
    # Using more conservative settings for WhisperX compatibility on Linux/WSL
    if vram_gb >= 80:
        # Blackwell 96GB or A100 80GB - using conservative chunks for stability
        max_split = 256  # Reduced from 512 for better WhisperX compatibility
        gc_threshold = 0.7  # More aggressive GC
        print(f"[CUDA Setup] Optimized for {vram_gb:.0f}GB VRAM (Blackwell/large GPU)")
    elif vram_gb >= 40:
        # Ada 48GB or A100 40GB - medium chunks
        max_split = 256  # Reduced from 384
        gc_threshold = 0.75
        print(f"[CUDA Setup] Optimized for {vram_gb:.0f}GB VRAM (Ada/medium GPU)")
    else:
        # Smaller GPUs - conservative
        max_split = 128  # More conservative
        gc_threshold = 0.85
        print(f"[CUDA Setup] Optimized for {vram_gb:.0f}GB VRAM (conservative mode)")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"expandable_segments:True,max_split_size_mb:{max_split},garbage_collection_threshold:{gc_threshold}"

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

    # OpenBLAS/MKL threading configuration to avoid conflicts with CUDA
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # Print confirmation
    print("[CUDA Setup] Environment configured for RTX 6000 Blackwell (96GB VRAM)")
    print(f"[CUDA Setup] PYTORCH_CUDA_ALLOC_CONF: {os.environ['PYTORCH_CUDA_ALLOC_CONF']}")

    # Check if WSL
    if "MALLOC_TRIM_THRESHOLD_" in os.environ:
        print("[CUDA Setup] WSL optimizations enabled")


# Run setup immediately on import
setup_cuda_environment()
