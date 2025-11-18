"""
System Health Checker - Comprehensive verification of installation
Detects environment, GPU, ONNX Runtime, and provides recommendations
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def detect_environment() -> Tuple[str, str]:
    """
    Detect if running in WSL, Windows, or Linux
    Returns: (environment_type, details)
    """
    try:
        uname = os.uname()
        release = uname.release.lower()

        if "microsoft" in release or "wsl" in release:
            # WSL environment
            version = "WSL2" if "wsl2" in release else "WSL1"
            return "wsl", f"{version} (Ubuntu/Debian on Windows)"
        elif sys.platform == "win32":
            return "windows", "Native Windows"
        elif sys.platform.startswith("linux"):
            # Check distro
            try:
                with open("/etc/os-release") as f:
                    distro_info = f.read()
                    if "Ubuntu" in distro_info:
                        return "linux", "Native Linux (Ubuntu)"
                    elif "Debian" in distro_info:
                        return "linux", "Native Linux (Debian)"
                    else:
                        return "linux", "Native Linux"
            except:
                return "linux", "Native Linux (Unknown Distro)"
        else:
            return "unknown", sys.platform
    except Exception as e:
        return "unknown", str(e)


def check_onnx_runtime() -> Tuple[bool, List[str], str]:
    """
    Check ONNX Runtime installation and available providers
    Returns: (has_cuda_provider, all_providers, message)
    """
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        has_cuda = 'CUDAExecutionProvider' in providers

        if has_cuda:
            message = "✅ ONNX Runtime with CUDA support detected"
        else:
            message = "⚠️  ONNX Runtime CUDA provider missing - CPU only mode"

        return has_cuda, providers, message
    except ImportError:
        return False, [], "❌ ONNX Runtime not installed"
    except Exception as e:
        return False, [], f"❌ Error checking ONNX Runtime: {str(e)}"


def check_pytorch_cuda() -> Tuple[bool, str, str]:
    """
    Check PyTorch CUDA availability
    Returns: (is_available, cuda_version, message)
    """
    try:
        import torch

        if torch.cuda.is_available():
            cuda_version = torch.version.cuda or "Unknown"
            device_count = torch.cuda.device_count()
            message = f"✅ PyTorch CUDA {cuda_version} available ({device_count} GPU(s))"
            return True, cuda_version, message
        else:
            message = "⚠️  PyTorch installed but CUDA not available"
            return False, "N/A", message
    except ImportError:
        return False, "N/A", "❌ PyTorch not installed"
    except Exception as e:
        return False, "N/A", f"❌ Error checking PyTorch: {str(e)}"


def get_gpu_info() -> List[Dict[str, str]]:
    """
    Get detailed GPU information using nvidia-smi
    Returns: List of GPU dictionaries with id, name, memory, driver
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        gpus.append({
                            "id": parts[0],
                            "name": parts[1],
                            "memory": parts[2],
                            "driver": parts[3]
                        })
            return gpus
    except Exception:
        pass

    return []


def get_compute_capability() -> Dict[str, Tuple[int, str]]:
    """
    Get compute capability for each GPU
    Returns: Dict[gpu_id, (compute_cap, architecture_name)]
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return {}

        result = {}
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            compute_cap = props.major * 10 + props.minor

            # Determine architecture
            if compute_cap >= 90:
                arch = "Blackwell"
            elif compute_cap >= 89:
                arch = "Ada Lovelace"
            elif compute_cap >= 86:
                arch = "Ada Lovelace (Workstation)"
            elif compute_cap >= 80:
                arch = "Ampere"
            elif compute_cap >= 75:
                arch = "Turing"
            elif compute_cap >= 70:
                arch = "Volta"
            else:
                arch = "Legacy"

            result[str(i)] = (compute_cap, arch)

        return result
    except:
        return {}


def get_batch_size_recommendations() -> Dict[str, Dict[str, str]]:
    """
    Get batch size recommendations based on detected GPUs
    Returns: Dict[gpu_id, Dict[task, recommendation]]
    """
    gpus = get_gpu_info()
    compute_caps = get_compute_capability()

    recommendations = {}

    for gpu in gpus:
        gpu_id = gpu["id"]
        memory_str = gpu["memory"]

        # Parse memory
        try:
            memory_mb = int(memory_str.split()[0])
            vram_gb = memory_mb / 1024
        except:
            vram_gb = 48  # Default

        # Get architecture
        compute_cap, arch = compute_caps.get(gpu_id, (80, "Unknown"))

        # Recommendations based on VRAM and architecture
        if vram_gb >= 80:  # Blackwell 96GB or A100 80GB
            if compute_cap >= 90:  # Blackwell
                rec = {
                    "emilia_pipeline": "28-32",
                    "whisperx": "26-30",
                    "transcriber": "28-32",
                    "architecture": arch
                }
            else:  # A100 80GB
                rec = {
                    "emilia_pipeline": "24-28",
                    "whisperx": "22-26",
                    "transcriber": "24-28",
                    "architecture": arch
                }
        elif vram_gb >= 40:  # Ada 48GB or A100 40GB
            rec = {
                "emilia_pipeline": "20-26",
                "whisperx": "22-26",
                "transcriber": "20-26",
                "architecture": arch
            }
        elif vram_gb >= 20:  # Mid-range
            rec = {
                "emilia_pipeline": "12-16",
                "whisperx": "14-18",
                "transcriber": "12-16",
                "architecture": arch
            }
        else:  # Smaller GPUs
            rec = {
                "emilia_pipeline": "4-8",
                "whisperx": "6-10",
                "transcriber": "4-8",
                "architecture": arch
            }

        recommendations[gpu_id] = rec

    return recommendations


def check_critical_packages() -> Dict[str, Tuple[bool, str]]:
    """
    Check critical packages and their versions
    Returns: Dict[package_name, (is_installed, version_or_error)]
    """
    packages = {
        "torch": None,
        "torchaudio": None,
        "onnxruntime-gpu": "onnxruntime",
        "whisperx": None,
        "pyannote.audio": "pyannote.audio",
        "gradio": None,
        "librosa": None,
    }

    results = {}

    for display_name, import_name in packages.items():
        module_name = import_name or display_name
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "Unknown")
            results[display_name] = (True, version)
        except ImportError:
            results[display_name] = (False, "Not installed")
        except Exception as e:
            results[display_name] = (False, f"Error: {str(e)}")

    return results


def generate_health_report() -> str:
    """
    Generate comprehensive health report
    Returns: Formatted health report string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("SYSTEM HEALTH CHECK REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Environment detection
    env_type, env_details = detect_environment()
    lines.append(f"🖥️  Environment: {env_details}")

    if env_type == "wsl":
        lines.append("   💡 WSL detected - ensure you've configured .wslconfig for optimal performance")
        lines.append("   📖 See wsl_setup.md for detailed WSL optimization guide")

    lines.append("")

    # GPU Information
    lines.append("🎮 GPU Information:")
    gpus = get_gpu_info()
    compute_caps = get_compute_capability()

    if gpus:
        for gpu in gpus:
            gpu_id = gpu["id"]
            compute_cap, arch = compute_caps.get(gpu_id, (0, "Unknown"))
            lines.append(f"   GPU {gpu_id}: {gpu['name']}")
            lines.append(f"     Memory: {gpu['memory']}")
            lines.append(f"     Driver: {gpu['driver']}")
            if compute_cap > 0:
                lines.append(f"     Architecture: {arch} (Compute Capability {compute_cap/10:.1f})")
    else:
        lines.append("   ⚠️  No NVIDIA GPUs detected or nvidia-smi not available")

    lines.append("")

    # PyTorch CUDA
    lines.append("🔥 PyTorch CUDA:")
    cuda_available, cuda_version, cuda_msg = check_pytorch_cuda()
    lines.append(f"   {cuda_msg}")

    if cuda_available:
        try:
            import torch
            lines.append(f"   PyTorch version: {torch.__version__}")
        except:
            pass

    lines.append("")

    # ONNX Runtime
    lines.append("⚙️  ONNX Runtime:")
    has_cuda_ort, providers, ort_msg = check_onnx_runtime()
    lines.append(f"   {ort_msg}")

    if providers:
        lines.append(f"   Available providers: {', '.join(providers)}")

    if not has_cuda_ort:
        lines.append("")
        lines.append("   ⚠️  WARNING: ONNX Runtime CUDA provider not found!")
        lines.append("   🔧 Quick fix:")
        lines.append("      uv run python setup_onnx_cuda.py")
        lines.append("   Or manually:")
        lines.append("      uv remove onnxruntime")
        lines.append("      uv add onnxruntime-gpu==1.20.1")

    lines.append("")

    # Critical Packages
    lines.append("📦 Critical Packages:")
    packages = check_critical_packages()

    for pkg_name, (installed, version) in packages.items():
        status = "✅" if installed else "❌"
        lines.append(f"   {status} {pkg_name}: {version}")

    lines.append("")

    # Batch Size Recommendations
    lines.append("🎯 Recommended Batch Sizes:")
    recommendations = get_batch_size_recommendations()

    if recommendations:
        for gpu_id, rec in recommendations.items():
            arch = rec.get("architecture", "Unknown")
            lines.append(f"   GPU {gpu_id} ({arch}):")
            lines.append(f"     Emilia Pipeline: {rec.get('emilia_pipeline', 'N/A')}")
            lines.append(f"     WhisperX: {rec.get('whisperx', 'N/A')}")
            lines.append(f"     Transcriber: {rec.get('transcriber', 'N/A')}")
    else:
        lines.append("   No GPU detected - recommendations unavailable")

    lines.append("")

    # Environment Variables
    lines.append("🔧 CUDA Environment Variables:")
    cuda_vars = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "Not set (all GPUs visible)"),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "Not set"),
        "CUDA_LAUNCH_BLOCKING": os.environ.get("CUDA_LAUNCH_BLOCKING", "Not set"),
    }

    for var, value in cuda_vars.items():
        lines.append(f"   {var}: {value}")

    lines.append("")
    lines.append("=" * 80)

    # Overall Status
    all_good = cuda_available and has_cuda_ort and all(
        installed for installed, _ in [packages.get("torch", (False, "")),
                                       packages.get("whisperx", (False, "")),
                                       packages.get("pyannote.audio", (False, ""))]
    )

    if all_good:
        lines.append("✅ SYSTEM READY - All critical components detected and working")
    else:
        lines.append("⚠️  SYSTEM ISSUES DETECTED - Please review warnings above")

    lines.append("=" * 80)

    return "\n".join(lines)


def auto_fix_onnx_runtime() -> Tuple[bool, str]:
    """
    Attempt to automatically fix ONNX Runtime CUDA provider issue
    Returns: (success, message)
    """
    has_cuda, providers, _ = check_onnx_runtime()

    if has_cuda:
        return True, "✅ ONNX Runtime CUDA provider already available - no fix needed"

    try:
        # Try to reinstall onnxruntime-gpu
        result = subprocess.run(
            ["uv", "remove", "onnxruntime"],
            capture_output=True,
            text=True,
            timeout=30
        )

        result = subprocess.run(
            ["uv", "add", "onnxruntime-gpu==1.20.1"],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            # Verify fix
            has_cuda_after, _, _ = check_onnx_runtime()
            if has_cuda_after:
                return True, "✅ ONNX Runtime CUDA provider fixed successfully!"
            else:
                return False, "⚠️  Reinstallation completed but CUDA provider still not available"
        else:
            return False, f"❌ Failed to reinstall: {result.stderr}"

    except subprocess.TimeoutExpired:
        return False, "❌ Installation timeout - please try manually"
    except Exception as e:
        return False, f"❌ Error during auto-fix: {str(e)}"


if __name__ == "__main__":
    # Run health check if executed directly
    print(generate_health_report())
