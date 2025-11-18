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


def fix_ctranslate2_cudnn():
    """
    Fix ctranslate2 cuDNN compatibility by creating symlinks.
    ctranslate2 comes with an incomplete cuDNN stub that conflicts with nvidia-cudnn-cu12.
    This function creates symlinks to use the full cuDNN from nvidia-cudnn-cu12.
    """
    import sysconfig
    site_packages = sysconfig.get_paths()["purelib"]

    cudnn_lib_path = os.path.join(site_packages, "nvidia", "cudnn", "lib")
    ctranslate2_libs_path = os.path.join(site_packages, "ctranslate2.libs")

    # Only proceed if both directories exist
    if not os.path.exists(cudnn_lib_path) or not os.path.exists(ctranslate2_libs_path):
        return

    # Fix 1: Create symlink in ctranslate2.libs for the specific cuDNN file it needs
    cudnn_stub = os.path.join(ctranslate2_libs_path, "libcudnn-74a4c495.so.9.1.0")
    cudnn_target = os.path.join(cudnn_lib_path, "libcudnn.so.9")

    # Check if stub needs to be fixed (exists and is not a symlink to the right target)
    if os.path.exists(cudnn_stub):
        if os.path.islink(cudnn_stub):
            # Already a symlink - check if it points to the right place
            current_target = os.readlink(cudnn_stub)
            if "nvidia/cudnn" not in current_target:
                os.remove(cudnn_stub)
                os.symlink(cudnn_target, cudnn_stub)
        else:
            # It's a regular file (the incomplete stub) - backup and replace
            if os.path.getsize(cudnn_stub) < 200000:  # Stub is ~126KB
                os.rename(cudnn_stub, cudnn_stub + ".ORIGINAL")
                os.symlink(cudnn_target, cudnn_stub)

    # Fix 2: Create version symlinks in nvidia/cudnn/lib for compatibility
    cudnn_ops = os.path.join(cudnn_lib_path, "libcudnn_ops.so.9")
    if os.path.exists(cudnn_ops):
        for suffix in ["", ".9.1", ".9.1.0"]:
            link_name = os.path.join(cudnn_lib_path, f"libcudnn_ops.so{suffix}")
            if suffix and not os.path.exists(link_name):
                os.symlink("libcudnn_ops.so.9", link_name)


def setup_cuda_environment():
    """
    Configure CUDA environment variables BEFORE importing torch.
    This prevents std::bad_alloc errors during initial model loading.
    """
    # Fix ctranslate2 cuDNN compatibility first
    try:
        fix_ctranslate2_cudnn()
    except Exception as e:
        # Don't fail if fix doesn't work - just warn
        print(f"[CUDA Setup] Warning: Could not apply ctranslate2 cuDNN fix: {e}")

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
            # Multiple GPUs detected - make all available
            print(f"[CUDA Setup] Detected {len(gpus)} GPUs:")
            for gpu in gpus:
                print(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['memory']})")
            # Make all GPUs available instead of limiting to GPU 0
            all_gpu_ids = ",".join([gpu['id'] for gpu in gpus])
            os.environ["CUDA_VISIBLE_DEVICES"] = all_gpu_ids
            print(f"[CUDA Setup] All GPUs available. CUDA_VISIBLE_DEVICES={all_gpu_ids}")
        elif len(gpus) == 1:
            print(f"[CUDA Setup] Detected 1 GPU: {gpus[0]['name']} ({gpus[0]['memory']})")
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        visible = os.environ["CUDA_VISIBLE_DEVICES"]
        print(f"[CUDA Setup] CUDA_VISIBLE_DEVICES already set to: {visible}")

    # Memory allocator configuration
    # Detect VRAM size to optimize chunk size
    # Note: This is a heuristic based on typical GPU memory (detection before torch import)
    # Use the GPU with maximum VRAM for configuration
    if gpus:
        # Find GPU with most memory
        max_memory_gpu = max(gpus, key=lambda g: int(g["memory"].split()[0]) if g["memory"] else 0)
        total_memory_str = max_memory_gpu["memory"]
    else:
        total_memory_str = "0 MiB"

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
