#!/usr/bin/env python3
"""
Verification script for Dataset Maker installation on RTX 6000 Blackwell.
Checks CUDA, PyTorch, ONNX Runtime, and other critical dependencies.
"""

import sys
import platform
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def check_python_version():
    """Check Python version."""
    print_header("Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    print(f"Platform: {platform.platform()}")

    if version.major == 3 and 10 <= version.minor < 13:
        print("✓ Python version is compatible (3.10-3.12)")
        return True
    else:
        print("✗ Python version should be 3.10-3.12")
        return False


def check_pytorch():
    """Check PyTorch installation and CUDA support."""
    print_header("PyTorch and CUDA")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  Compute Capability: {props.major}.{props.minor}")
                print(f"  Total Memory: {props.total_memory / (1024**3):.2f} GB")
                print(f"  Multi-Processor Count: {props.multi_processor_count}")

                # Check if Blackwell (Compute Capability 9.0+)
                if props.major >= 9:
                    print(f"  ✓ Blackwell architecture detected (CC {props.major}.{props.minor})")
                elif props.major >= 8:
                    print(f"  ✓ Ampere/Ada architecture (CC {props.major}.{props.minor})")
                else:
                    print(f"  ⚠ Older architecture (CC {props.major}.{props.minor})")

            # Test TF32 support
            print(f"\nTF32 for matmul: {torch.backends.cuda.matmul.allow_tf32}")
            print(f"TF32 for cudnn: {torch.backends.cudnn.allow_tf32}")

            # Test a simple CUDA operation
            try:
                x = torch.randn(1000, 1000, device='cuda')
                y = torch.matmul(x, x)
                del x, y
                torch.cuda.empty_cache()
                print("✓ CUDA operations working correctly")
            except Exception as e:
                print(f"✗ CUDA operation failed: {e}")
                return False

            return True
        else:
            print("✗ CUDA is not available. GPU acceleration will not work.")
            return False

    except ImportError:
        print("✗ PyTorch is not installed")
        return False
    except Exception as e:
        print(f"✗ Error checking PyTorch: {e}")
        return False


def check_onnxruntime():
    """Check ONNX Runtime installation and CUDA provider."""
    print_header("ONNX Runtime")
    try:
        import onnxruntime as ort
        print(f"ONNX Runtime version: {ort.__version__}")

        providers = ort.get_available_providers()
        print(f"Available providers: {providers}")

        has_cuda = 'CUDAExecutionProvider' in providers
        has_tensorrt = 'TensorrtExecutionProvider' in providers

        if has_cuda:
            print("✓ CUDAExecutionProvider is available")
        else:
            print("✗ CUDAExecutionProvider is NOT available")
            print("  Run: uv remove onnxruntime && uv add onnxruntime-gpu==1.20.1")

        if has_tensorrt:
            print("✓ TensorrtExecutionProvider is available (bonus!)")

        return has_cuda

    except ImportError:
        print("✗ ONNX Runtime is not installed")
        return False
    except Exception as e:
        print(f"✗ Error checking ONNX Runtime: {e}")
        return False


def check_whisperx():
    """Check WhisperX installation."""
    print_header("WhisperX")
    try:
        import whisperx
        print(f"✓ WhisperX is installed")

        # Try to import key components
        from whisperx.audio import load_audio
        from whisperx.asr import FasterWhisperPipeline
        print("✓ WhisperX components available")
        return True

    except ImportError as e:
        print(f"✗ WhisperX import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error checking WhisperX: {e}")
        return False


def check_pyannote():
    """Check pyannote.audio installation."""
    print_header("Pyannote Audio")
    try:
        import pyannote.audio
        print(f"Pyannote.audio version: {pyannote.audio.__version__}")

        from pyannote.audio import Pipeline
        print("✓ Pyannote Pipeline available")

        # Check if HF token is configured
        import os
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            print("✓ Hugging Face token found in environment")
        else:
            print("⚠ Hugging Face token not found in environment")
            print("  You'll need to provide it in config or set HF_TOKEN env var")

        return True

    except ImportError as e:
        print(f"✗ Pyannote.audio import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error checking pyannote: {e}")
        return False


def check_other_dependencies():
    """Check other critical dependencies."""
    print_header("Other Dependencies")

    deps = {
        "librosa": "Audio processing",
        "soundfile": "Audio I/O",
        "numpy": "Numerical operations",
        "gradio": "Web interface",
        "demucs": "Source separation (alternative)",
        "psutil": "System monitoring",
    }

    all_ok = True
    for module, description in deps.items():
        try:
            __import__(module)
            print(f"✓ {module:20s} - {description}")
        except ImportError:
            print(f"✗ {module:20s} - {description} (NOT INSTALLED)")
            all_ok = False

    return all_ok


def check_wsl_optimizations():
    """Check WSL-specific optimizations."""
    print_header("WSL Optimizations")

    if "microsoft" in platform.release().lower() or "wsl" in platform.release().lower():
        print("✓ Running in WSL environment")

        # Check for common WSL issues
        import os

        # Check CUDA visible devices
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")
        print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")

        # Check PyTorch CUDA allocation config
        cuda_alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "Not set")
        print(f"PYTORCH_CUDA_ALLOC_CONF: {cuda_alloc}")
        if cuda_alloc == "Not set":
            print("  ⚠ Consider setting: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512")

        return True
    else:
        print("ℹ Not running in WSL (or not detected)")
        return True


def estimate_batch_size():
    """Estimate optimal batch size for the GPU."""
    print_header("Batch Size Recommendations")

    try:
        import torch
        if not torch.cuda.is_available():
            print("GPU not available, skipping batch size estimation")
            return

        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Available VRAM: {vram_gb:.2f} GB")

        # Recommendations based on VRAM
        if vram_gb >= 48:  # RTX 6000 Ada/Blackwell
            print("\nRecommended batch sizes for RTX 6000 (48GB):")
            print("  - Emilia pipeline batch_size: 16-24")
            print("  - WhisperX chunk_size: 20-30")
            print("  - Transcriber batch_size: 16-24")
        elif vram_gb >= 24:
            print("\nRecommended batch sizes for 24GB VRAM:")
            print("  - Emilia pipeline batch_size: 8-16")
            print("  - WhisperX chunk_size: 15-20")
            print("  - Transcriber batch_size: 12-16")
        elif vram_gb >= 12:
            print("\nRecommended batch sizes for 12GB VRAM:")
            print("  - Emilia pipeline batch_size: 4-8")
            print("  - WhisperX chunk_size: 10-15")
            print("  - Transcriber batch_size: 8-12")
        else:
            print("\nRecommended batch sizes for <12GB VRAM:")
            print("  - Emilia pipeline batch_size: 2-4")
            print("  - WhisperX chunk_size: 8-10")
            print("  - Transcriber batch_size: 4-8")

        print("\n⚠ Start with lower values and increase if memory allows")

    except Exception as e:
        print(f"Could not estimate batch size: {e}")


def main():
    """Run all checks."""
    print("\n" + "=" * 70)
    print("  Dataset Maker Installation Verification")
    print("  Optimized for RTX 6000 Blackwell")
    print("=" * 70)

    checks = [
        ("Python Version", check_python_version),
        ("PyTorch & CUDA", check_pytorch),
        ("ONNX Runtime", check_onnxruntime),
        ("WhisperX", check_whisperx),
        ("Pyannote Audio", check_pyannote),
        ("Other Dependencies", check_other_dependencies),
        ("WSL Environment", check_wsl_optimizations),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Unexpected error in {name}: {e}")
            results.append((name, False))

    # Estimate batch sizes
    try:
        estimate_batch_size()
    except Exception as e:
        print(f"Could not estimate batch sizes: {e}")

    # Summary
    print_header("Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} - {name}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All checks passed! Your installation is ready for RTX 6000 Blackwell.")
        return 0
    else:
        print("\n⚠ Some checks failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
