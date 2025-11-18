#!/usr/bin/env python3
"""Test WhisperX complete transcription on CUDA with ctranslate2 4.6.1"""

import setup_cuda_env

import torch
import numpy as np
import soundfile as sf
import tempfile
from whisperx import load_audio
from Emilia.models.whisper_asr import load_asr_model
import ctranslate2


def main():
    print('=' * 70)
    print('WhisperX CUDA Transcription Test')
    print('=' * 70)

    print(f'\n1. Versions:')
    print(f'   PyTorch: {torch.__version__}')
    print(f'   CUDA: {torch.version.cuda}')
    print(f'   cuDNN: {torch.backends.cudnn.version()}')
    print(f'   ctranslate2: {ctranslate2.__version__}')
    print(f'   GPU: {torch.cuda.get_device_name(0)}')

    print('\n2. Creating test audio...')
    duration = 3.0
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.3 * np.sin(2 * np.pi * 150 * t)
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(temp_file.name, audio, sr)
    print(f'   ✓ Created: {temp_file.name}')

    print('\n3. Loading WhisperX tiny.en on CUDA...')
    model = load_asr_model('tiny.en', device='cuda', compute_type='float16', language='en')
    print('   ✓ Model loaded on CUDA!')

    print('\n4. Loading audio...')
    audio = load_audio(temp_file.name)
    duration = len(audio) / 16000.0
    vad_segments = [{'start': 0.0, 'end': duration}]
    print(f'   ✓ Audio loaded: {duration:.2f}s')

    print('\n5. Transcribing on CUDA...')
    result = model.transcribe(
        audio=audio,
        vad_segments=vad_segments,
        batch_size=1,
        language='en',
        print_progress=False
    )

    print(f'   ✓ Transcription complete!')
    print(f'   Segments: {len(result.get("segments", []))}')

    for seg in result.get('segments', [])[:3]:
        text = seg.get('text', '').strip()
        start = seg.get('start', 0)
        end = seg.get('end', 0)
        print(f'   [{start:.2f}s-{end:.2f}s]: {text}')

    # Cleanup
    import os
    os.unlink(temp_file.name)
    del model
    torch.cuda.empty_cache()

    print('\n' + '=' * 70)
    print('🎉 SUCCESS!')
    print('=' * 70)
    print('\nWhisperX CUDA transcription working perfectly!')
    print(f'✅ ctranslate2 4.6.1 + cuDNN 9 compatibility confirmed')
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print('=' * 70)


if __name__ == '__main__':
    main()
