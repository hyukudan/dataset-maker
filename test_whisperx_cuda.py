#!/usr/bin/env python3
"""
Test WhisperX specifically with CUDA to identify the cuDNN issue.
"""

import setup_cuda_env
import sys


def test_whisperx_cuda():
    """Test WhisperX with CUDA."""
    print("=" * 70)
    print("Testing WhisperX with CUDA")
    print("=" * 70)

    try:
        import torch
        print(f"\n1. PyTorch Info:")
        print(f"   Version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   cuDNN version: {torch.backends.cudnn.version()}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

        print(f"\n2. Testing basic CUDA operation...")
        x = torch.randn(100, 100, device='cuda')
        y = torch.matmul(x, x)
        del x, y
        torch.cuda.empty_cache()
        print("   ✓ Basic CUDA works")

        print(f"\n3. Loading WhisperX tiny.en model on CUDA...")
        from Emilia.models.whisper_asr import load_asr_model

        try:
            model = load_asr_model(
                whisper_arch="tiny.en",
                device="cuda",
                compute_type="float16",
                language="en"
            )
            print("   ✓ WhisperX model loaded successfully on CUDA!")

            # Clean up
            del model
            torch.cuda.empty_cache()

            print("\n" + "=" * 70)
            print("✓ WhisperX works with CUDA!")
            print("=" * 70)
            return True

        except Exception as e:
            print(f"   ✗ WhisperX CUDA loading failed:")
            print(f"   Error: {e}")
            print(f"\n   This is likely a cuDNN version mismatch:")
            print(f"   - PyTorch has cuDNN 9.1 (version {torch.backends.cudnn.version()})")
            print(f"   - FasterWhisper/CTranslate2 expects cuDNN 8.x")
            print(f"\n   However, CPU mode should still work fine.")

            print(f"\n4. Testing WhisperX on CPU as fallback...")
            model = load_asr_model(
                whisper_arch="tiny.en",
                device="cpu",
                compute_type="float32",
                language="en"
            )
            print("   ✓ WhisperX works on CPU!")
            del model

            print("\n" + "=" * 70)
            print("⚠ WhisperX works on CPU but not CUDA (cuDNN incompatibility)")
            print("=" * 70)
            return False

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_whisperx_cuda()
    sys.exit(0 if success else 1)
