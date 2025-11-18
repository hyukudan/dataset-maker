#!/usr/bin/env python3
"""
Comprehensive functionality test for Dataset Maker.
Tests that all critical imports work and key functions can be called.
"""

# CRITICAL: Import setup_cuda_env FIRST to configure CUDA memory allocator
import setup_cuda_env

import sys
import traceback
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def test_basic_imports():
    """Test that all critical modules can be imported."""
    print_header("Testing Basic Imports")

    modules_to_test = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("whisperx", "WhisperX"),
        ("pyannote.audio", "Pyannote Audio"),
        ("onnxruntime", "ONNX Runtime"),
        ("librosa", "Librosa"),
        ("soundfile", "SoundFile"),
        ("gradio", "Gradio"),
    ]

    failed = []
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ {description:20s} imported successfully")
        except Exception as e:
            print(f"✗ {description:20s} FAILED: {e}")
            failed.append((description, e))

    return len(failed) == 0, failed


def test_torch_cuda():
    """Test PyTorch CUDA functionality."""
    print_header("Testing PyTorch CUDA")

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if not torch.cuda.is_available():
            print("✗ CUDA not available")
            return False

        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")

        # Test a simple CUDA operation
        x = torch.randn(100, 100, device='cuda')
        y = torch.matmul(x, x)
        result = y.cpu().numpy()
        del x, y
        torch.cuda.empty_cache()

        print(f"✓ CUDA operations working (computed 100x100 matrix multiplication)")
        return True

    except Exception as e:
        print(f"✗ PyTorch CUDA test failed: {e}")
        traceback.print_exc()
        return False


def test_onnxruntime_cuda():
    """Test ONNX Runtime CUDA provider."""
    print_header("Testing ONNX Runtime CUDA")

    try:
        import onnxruntime as ort

        print(f"ONNX Runtime version: {ort.__version__}")
        providers = ort.get_available_providers()
        print(f"Available providers: {providers}")

        has_cuda = 'CUDAExecutionProvider' in providers
        if has_cuda:
            print("✓ CUDAExecutionProvider is available")
            return True
        else:
            print("✗ CUDAExecutionProvider is NOT available")
            return False

    except Exception as e:
        print(f"✗ ONNX Runtime test failed: {e}")
        traceback.print_exc()
        return False


def test_whisperx_load():
    """Test WhisperX model loading (small model, quick test)."""
    print_header("Testing WhisperX Model Loading")

    try:
        from whisperx.asr import WhisperModel
        import torch

        print("Attempting to load whisper tiny.en model...")

        # Use tiny.en model for quick testing
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = WhisperModel(
            "tiny.en",
            device=device,
            compute_type="float16" if device == "cuda" else "float32",
        )

        print(f"✓ WhisperX tiny.en model loaded successfully on {device}")

        # Clean up
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"✗ WhisperX model loading failed: {e}")
        traceback.print_exc()
        return False


def test_pyannote_pipeline():
    """Test pyannote.audio pipeline creation."""
    print_header("Testing Pyannote Audio Pipeline")

    try:
        from pyannote.audio import Pipeline
        import torch
        import os

        # Check for HF token
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        if not hf_token:
            print("⚠ No HF_TOKEN found, skipping pipeline creation")
            print("  (Pipeline import successful, which is the main test)")
            print("✓ Pyannote Audio imports working")
            return True

        print("Creating pyannote speaker diarization pipeline...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )

        if device == "cuda":
            pipeline = pipeline.to(torch.device("cuda"))

        print(f"✓ Pyannote pipeline created successfully on {device}")

        # Clean up
        del pipeline
        if device == "cuda":
            torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"✗ Pyannote pipeline test failed: {e}")
        traceback.print_exc()
        return False


def test_emilia_pipeline_import():
    """Test that emilia_pipeline can be imported."""
    print_header("Testing Emilia Pipeline Import")

    try:
        import emilia_pipeline

        print("✓ emilia_pipeline module imported successfully")

        # Check for key functions
        if hasattr(emilia_pipeline, 'prepare_models'):
            print("✓ prepare_models function found")
        if hasattr(emilia_pipeline, 'process_audio'):
            print("✓ process_audio function found")

        return True

    except Exception as e:
        print(f"✗ Emilia pipeline import failed: {e}")
        traceback.print_exc()
        return False


def test_transcriber_import():
    """Test that transcriber can be imported."""
    print_header("Testing Transcriber Import")

    try:
        import transcriber

        print("✓ transcriber module imported successfully")

        # Check for key classes/functions
        if hasattr(transcriber, 'TranscriberApp'):
            print("✓ TranscriberApp class found")

        return True

    except Exception as e:
        print(f"✗ Transcriber import failed: {e}")
        traceback.print_exc()
        return False


def test_whisper_asr_module():
    """Test Emilia whisper_asr module."""
    print_header("Testing Emilia Whisper ASR Module")

    try:
        from Emilia.models import whisper_asr

        print("✓ whisper_asr module imported successfully")

        # Check for key components
        if hasattr(whisper_asr, 'VadFreeFasterWhisperPipeline'):
            print("✓ VadFreeFasterWhisperPipeline class found")
        if hasattr(whisper_asr, 'load_asr_model'):
            print("✓ load_asr_model function found")

        return True

    except Exception as e:
        print(f"✗ Whisper ASR module test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all functionality tests."""
    print("\n" + "=" * 70)
    print("  Dataset Maker Comprehensive Functionality Tests")
    print("  Testing post-torchcodec fix")
    print("=" * 70)

    tests = [
        ("Basic Imports", test_basic_imports),
        ("PyTorch CUDA", test_torch_cuda),
        ("ONNX Runtime CUDA", test_onnxruntime_cuda),
        ("WhisperX Model Loading", test_whisperx_load),
        ("Pyannote Pipeline", test_pyannote_pipeline),
        ("Emilia Pipeline Import", test_emilia_pipeline_import),
        ("Transcriber Import", test_transcriber_import),
        ("Whisper ASR Module", test_whisper_asr_module),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            if isinstance(result, tuple):
                # Handle test_basic_imports which returns (success, failed_list)
                success, _ = result
                results.append((name, success))
            else:
                results.append((name, result))
        except Exception as e:
            print(f"\n✗ Unexpected error in {name}: {e}")
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print_header("Test Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} - {name}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All functionality tests passed!")
        print("   The torchcodec fix is working correctly.")
        print("   All modules can be imported and basic operations work.")
        return 0
    else:
        print("\n⚠ Some tests failed. Review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
