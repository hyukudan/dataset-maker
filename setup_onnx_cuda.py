#!/usr/bin/env python3
"""
ONNX Runtime CUDA Provider Setup Script
Ensures ONNX Runtime is correctly configured for CUDA execution on RTX 6000 Blackwell.
"""

import subprocess
import sys


def check_onnxruntime():
    """Check current ONNX Runtime installation."""
    try:
        import onnxruntime as ort
        print(f"Current ONNX Runtime version: {ort.__version__}")
        providers = ort.get_available_providers()
        print(f"Available providers: {providers}")

        if 'CUDAExecutionProvider' in providers:
            print("✓ CUDAExecutionProvider is already available!")
            return True
        else:
            print("✗ CUDAExecutionProvider is NOT available")
            return False
    except ImportError:
        print("✗ ONNX Runtime is not installed")
        return False


def reinstall_onnxruntime_gpu():
    """Reinstall onnxruntime-gpu with correct version."""
    print("\n" + "=" * 70)
    print("Reinstalling ONNX Runtime with GPU support...")
    print("=" * 70)

    commands = [
        # Remove existing onnxruntime packages
        ["uv", "remove", "onnxruntime"],
        ["uv", "remove", "onnxruntime-gpu"],
        ["uv", "remove", "optimum"],

        # Install onnxruntime-gpu first
        ["uv", "add", "onnxruntime-gpu==1.20.1"],

        # Then install optimum without onnxruntime extras
        ["uv", "add", "optimum>=2.0.0"],
    ]

    for cmd in commands:
        print(f"\nRunning: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if result.returncode != 0 and "not found" not in result.stderr.lower():
                print(f"Warning: {result.stderr}")
        except Exception as e:
            print(f"Command failed (this might be okay): {e}")

    print("\n" + "=" * 70)
    print("Reinstallation complete!")
    print("=" * 70)


def verify_installation():
    """Verify the ONNX Runtime installation."""
    print("\n" + "=" * 70)
    print("Verifying ONNX Runtime installation...")
    print("=" * 70)

    # Need to restart Python to reload modules
    print("\nPlease run the following command to verify:")
    print("  uv run python -c \"import onnxruntime as ort; print('Providers:', ort.get_available_providers())\"")
    print("\nYou should see 'CUDAExecutionProvider' in the list.")


def main():
    print("\n" + "=" * 70)
    print("  ONNX Runtime CUDA Setup for RTX 6000 Blackwell")
    print("=" * 70)

    has_cuda = check_onnxruntime()

    if has_cuda:
        print("\n✓ ONNX Runtime is already correctly configured!")
        print("No action needed.")
        return 0

    print("\n⚠ ONNX Runtime needs to be reconfigured for CUDA support.")
    response = input("Do you want to reinstall ONNX Runtime with GPU support? [y/N]: ")

    if response.lower() in ['y', 'yes']:
        reinstall_onnxruntime_gpu()
        verify_installation()
        print("\n✓ Setup complete! Please restart your Python environment and verify.")
        return 0
    else:
        print("\nSetup cancelled. You can run this script again later.")
        print("Manual fix: uv remove onnxruntime && uv add onnxruntime-gpu==1.20.1")
        return 1


if __name__ == "__main__":
    sys.exit(main())
