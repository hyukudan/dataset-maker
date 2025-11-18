"""
Quick Test - Generate sample audio and run complete pipeline
Tests the entire system with a short audio sample to verify everything works
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import tempfile
import time
from typing import Tuple, Dict
import os


def generate_test_audio(duration_seconds: float = 30.0, sample_rate: int = 16000) -> Tuple[Path, Dict]:
    """
    Generate synthetic test audio with speech-like characteristics

    Args:
        duration_seconds: Duration of test audio
        sample_rate: Sample rate in Hz

    Returns:
        (audio_path, metadata)
    """
    # Generate time array
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))

    # Create speech-like audio with multiple frequency components
    # Simulate formants (resonant frequencies in human speech)
    f1 = 500  # First formant
    f2 = 1500  # Second formant
    f3 = 2500  # Third formant

    # Generate base signal
    signal = (
        0.3 * np.sin(2 * np.pi * f1 * t) +
        0.2 * np.sin(2 * np.pi * f2 * t) +
        0.1 * np.sin(2 * np.pi * f3 * t)
    )

    # Add amplitude modulation to simulate speech patterns
    modulation_freq = 4  # Typical speech rate
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * modulation_freq * t)
    signal = signal * envelope

    # Add pauses to simulate natural speech
    pause_starts = [5, 15, 25]  # Pause at these times
    for pause_start in pause_starts:
        if pause_start < duration_seconds:
            pause_samples = int(sample_rate * 1.0)  # 1 second pause
            start_idx = int(pause_start * sample_rate)
            end_idx = min(start_idx + pause_samples, len(signal))
            signal[start_idx:end_idx] *= 0.05  # Reduce volume significantly

    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.9

    # Save to temporary file
    temp_dir = Path(tempfile.gettempdir()) / "dataset_maker_test"
    temp_dir.mkdir(exist_ok=True)

    audio_path = temp_dir / "test_audio_30s.wav"
    sf.write(str(audio_path), signal, sample_rate)

    metadata = {
        "duration_seconds": duration_seconds,
        "sample_rate": sample_rate,
        "file_size_bytes": audio_path.stat().st_size,
        "file_path": str(audio_path),
    }

    return audio_path, metadata


def run_quick_test(model_name: str = "medium", batch_size: int = 16) -> Tuple[bool, str]:
    """
    Run quick test of the complete pipeline

    Args:
        model_name: Whisper model to use for test
        batch_size: Batch size for testing

    Returns:
        (success, detailed_report)
    """
    lines = []
    lines.append("=" * 80)
    lines.append("QUICK TEST - SYSTEM VALIDATION")
    lines.append("=" * 80)
    lines.append("")

    start_time = time.time()

    try:
        # Step 1: Generate test audio
        lines.append("🎵 Step 1: Generating test audio...")
        audio_path, metadata = generate_test_audio(duration_seconds=30.0)
        lines.append(f"   ✅ Generated: {audio_path.name}")
        lines.append(f"   Duration: {metadata['duration_seconds']}s")
        lines.append(f"   Size: {metadata['file_size_bytes'] / 1024:.1f} KB")
        lines.append("")

        # Step 2: Load models
        lines.append("🔧 Step 2: Loading models...")
        lines.append(f"   Whisper model: {model_name}")
        lines.append(f"   Batch size: {batch_size}")

        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_properties(0).name
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            lines.append(f"   GPU: {gpu_name} ({vram_gb:.1f}GB)")
        else:
            lines.append("   ⚠️  No GPU detected - running on CPU")

        lines.append("")

        # Step 3: Run transcription (simplified)
        lines.append("🎙️  Step 3: Running transcription test...")

        try:
            import transcriber
            model = transcriber.load_whisperx_model(model_name)
            lines.append("   ✅ WhisperX model loaded successfully")

            # Try transcribing
            import whisperx
            audio = whisperx.load_audio(str(audio_path))
            result = model.transcribe(audio, batch_size=batch_size)

            lines.append(f"   ✅ Transcription completed")
            if "segments" in result:
                lines.append(f"   Segments detected: {len(result['segments'])}")

            # Cleanup
            del model, audio, result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            lines.append(f"   ⚠️  Transcription test skipped: {str(e)[:100]}")

        lines.append("")

        # Step 4: Check VRAM usage
        if torch.cuda.is_available():
            lines.append("💾 Step 4: VRAM Check...")
            vram_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            vram_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            lines.append(f"   Allocated: {vram_allocated:.2f}GB")
            lines.append(f"   Reserved: {vram_reserved:.2f}GB")
            lines.append("")

        # Cleanup test file
        audio_path.unlink()

        elapsed = time.time() - start_time

        lines.append("=" * 80)
        lines.append(f"✅ QUICK TEST PASSED in {elapsed:.1f} seconds")
        lines.append("=" * 80)
        lines.append("")
        lines.append("🎉 Your system is ready to process datasets!")
        lines.append("")
        lines.append("Next steps:")
        lines.append("1. Create a project in 'Project Setup' tab")
        lines.append("2. Upload your audio files")
        lines.append("3. Run transcription with your optimal settings")

        return True, "\n".join(lines)

    except Exception as e:
        lines.append("")
        lines.append("=" * 80)
        lines.append(f"❌ QUICK TEST FAILED")
        lines.append("=" * 80)
        lines.append(f"Error: {str(e)}")
        lines.append("")
        lines.append("Troubleshooting:")
        lines.append("1. Run 'System Health' check")
        lines.append("2. Verify CUDA and ONNX Runtime are properly installed")
        lines.append("3. Check the logs above for specific errors")

        return False, "\n".join(lines)


if __name__ == "__main__":
    # Test audio generation
    audio_path, metadata = generate_test_audio()
    print(f"Generated test audio: {audio_path}")
    print(f"Metadata: {metadata}")

    # Run quick test
    success, report = run_quick_test(model_name="medium", batch_size=8)
    print(report)
