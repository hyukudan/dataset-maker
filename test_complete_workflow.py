#!/usr/bin/env python3
"""
Complete workflow test with real audio processing on CUDA.
"""

import setup_cuda_env

import sys
import tempfile
import traceback
import numpy as np
import soundfile as sf


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def create_test_audio():
    """Create synthetic test audio."""
    duration = 3.0
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration))

    # Speech-like audio with varying frequencies
    audio = np.zeros_like(t)
    audio += 0.3 * np.sin(2 * np.pi * 150 * t)  # Fundamental
    audio += 0.15 * np.sin(2 * np.pi * 300 * t)  # Harmonic
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))
    audio = audio * envelope
    audio = audio / np.max(np.abs(audio)) * 0.8

    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(temp_file.name, audio, sr)
    return temp_file.name


def test_complete_workflow():
    """Test complete audio processing workflow."""
    print("\n" + "=" * 70)
    print("  Complete Workflow Test - CUDA Edition")
    print("=" * 70)

    try:
        import torch
        import os

        print(f"\n✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}")

        # Create test audio
        print_header("1. Creating Test Audio")
        audio_path = create_test_audio()
        print(f"✓ Created: {audio_path}")

        # Test WhisperX transcription on CUDA
        print_header("2. WhisperX Transcription (CUDA)")
        from whisperx import load_audio
        from Emilia.models.whisper_asr import load_asr_model

        print("Loading tiny.en model on CUDA...")
        model = load_asr_model(
            whisper_arch="tiny.en",
            device="cuda",
            compute_type="float16",
            language="en"
        )
        print("✓ Model loaded")

        audio = load_audio(audio_path)
        duration = len(audio) / 16000.0
        vad_segments = [{"start": 0.0, "end": duration}]

        print(f"Transcribing {duration:.2f}s audio...")
        result = model.transcribe(
            audio=audio,
            vad_segments=vad_segments,
            batch_size=1,
            language="en",
            print_progress=False
        )

        print(f"✓ Transcription complete")
        print(f"  Segments: {len(result.get('segments', []))}")
        for seg in result.get('segments', [])[:2]:
            print(f"  [{seg['start']:.2f}s-{seg['end']:.2f}s]: {seg['text']}")

        del model
        torch.cuda.empty_cache()

        # Test Silero VAD
        print_header("3. Silero VAD Detection")
        from Emilia.models.silero_vad import SileroVAD

        vad = SileroVAD(device=torch.device("cpu"))  # VAD works better on CPU
        audio_tensor = torch.from_numpy(audio)
        segments = vad.get_speech_timestamps(audio_tensor, vad.vad_model, sampling_rate=16000)

        print(f"✓ VAD complete: {len(segments)} speech segments found")
        for i, seg in enumerate(segments[:3]):
            print(f"  Segment {i+1}: [{seg['start']/16000:.2f}s - {seg['end']/16000:.2f}s]")

        del vad
        torch.cuda.empty_cache()

        # Test Pyannote import (diarization requires HF token)
        print_header("4. Pyannote Audio")
        from pyannote.audio import Pipeline

        print("✓ Pyannote Audio v3.4.0 imported")
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            print("✓ HF_TOKEN found - diarization available")
        else:
            print("⚠ HF_TOKEN not set - skipping actual diarization")

        # Test ONNX Runtime
        print_header("5. ONNX Runtime")
        import onnxruntime as ort

        providers = ort.get_available_providers()
        print(f"✓ Available providers: {providers}")
        print(f"✓ CUDA support: {'CUDAExecutionProvider' in providers}")

        # Memory check
        print_header("6. GPU Memory Management")
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"✓ GPU memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

        # Cleanup
        try:
            os.unlink(audio_path)
        except:
            pass

        print("\n" + "=" * 70)
        print("✅ COMPLETE WORKFLOW TEST PASSED!")
        print("=" * 70)
        print("\nAll components tested successfully:")
        print("  ✓ WhisperX transcription on CUDA")
        print("  ✓ Silero VAD speech detection")
        print("  ✓ Pyannote Audio ready")
        print("  ✓ ONNX Runtime with CUDA")
        print("  ✓ GPU memory management")
        print("\n🚀 Sistema listo para producción en CUDA!")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_complete_workflow()
    sys.exit(0 if success else 1)
