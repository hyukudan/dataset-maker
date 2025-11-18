# Dataset Maker
Multi-purpose dataset maker for various TTS models.
- Tortoise TTS/XTTS
- StyleTTS 2 ~ [Webui](https://github.com/JarodMica/StyleTTS-WebUI)
- Higgs Audio ~ [Base](https://github.com/JimmyMa99/train-higgs-audio) - [My fork](https://github.com/JarodMica/higgs-audio/tree/training)
- VibeVoice ~ [Base](https://github.com/voicepowered-ai/VibeVoice-finetuning) - [My fork](https://github.com/JarodMica/VibeVoice-finetuning)
- IndexTTS 2 ~ [My Trainer](https://github.com/JarodMica/index-tts/tree/training_v2)

## What does it output?
**Tortoise, StyleTTS2, XTTS** - Models like these take in a simple text file where audio:text pairs are sorted something like:
 - `path/to/audio/file | transcription`
 
 **Folder Sturcutre**
 ```bash
Dataset_name
- train.txt
-- seg1.wav
-- seg2.wav
 ```

**Higgs Audio** has a main metadata.json that includes all of the information and instructions for how to train on audio files, broken down by .txt files and .wav.

**Folder Structure**
```bash
Dataset_name
- metadata.json
- some_audio_1.txt
- some_audio_1.wav
- some_audio_2.txt
- some_audio_2.wav
```

**Vibe Voice** has a main `.jsonl` file that contains individual json entries with text and audio keys. It always prepends "Speaker 0: " before each transcription in accordance with what the trainer is expecting.
 - `{"text": "Speaker 0: some transcription", "audio": "path/to/audio"}`

**Folder Structure**
```bash
Dataset_name
- <project_name>_train.jsonl
- vibevoice_000000.wav
- vibevoice_000001.wav
```

## Installation

### Windows / WSL2 (Recommended for RTX 6000 Blackwell)

1. Make sure you have astral uv installed on your PC
2. Run the following:
    ```bash
    git clone https://github.com/JarodMica/dataset-maker.git
    cd dataset-maker
    uv sync
    ```
3. **Verify installation** (especially important for WSL and high-end GPUs):
    ```bash
    uv run python verify_installation.py
    ```
    This will check:
    - PyTorch with CUDA 12.8 support
    - ONNX Runtime with CUDAExecutionProvider
    - All dependencies are correctly installed
    - Optimal batch sizes for your GPU

4. **If using WSL2**, see [WSL Setup Guide](wsl_setup.md) for additional optimizations

5. Launch the gradio interface:
    ```bash
    uv run python gradio_interface.py
    ```

### RTX 6000 Blackwell Optimizations

This branch includes specific optimizations for RTX 6000 Blackwell GPUs:

- ✅ **Memory Management:** Aggressive garbage collection to prevent `std::bad_alloc` errors
- ✅ **TF32 Acceleration:** Automatic TF32 enablement for Blackwell architecture (~20-30% faster)
- ✅ **CUDA 12.8 Support:** Optimized PyTorch and ONNX Runtime versions
- ✅ **WSL2 Compatible:** Special configurations for Windows Subsystem for Linux
- ✅ **Batch Size Auto-tuning:** Recommendations based on available VRAM

**Recommended Settings:**
- **RTX 6000 Blackwell (96GB):**
  - Emilia pipeline `batch_size`: 24-32
  - WhisperX `chunk_size`: 25-30
  - Transcriber `batch_size`: 24-32

- **RTX 6000 Ada (48GB):**
  - Emilia pipeline `batch_size`: 16-24
  - WhisperX `chunk_size`: 20-30
  - Transcriber `batch_size`: 16-24

## Troubleshooting

### ONNX Runtime CUDA Provider Missing

**Problem:** CUDAExecutionProvider not available even after installation

**Quick Fix:**
```bash
uv run python setup_onnx_cuda.py
```

Or manually:
```bash
uv remove onnxruntime
uv add onnxruntime-gpu==1.20.1
```

**Verify:**
```bash
uv run python -c "import onnxruntime as ort; print('Providers:', ort.get_available_providers())"
```

You should see `['CUDAExecutionProvider', 'CPUExecutionProvider']` or similar.

### std::bad_alloc Errors with Pyannote/Whisper

**Problem:** Out of memory errors despite having plenty of VRAM

**Solutions:**
1. The code now includes aggressive garbage collection - this should fix most issues
2. If still occurring, reduce batch sizes:
   ```bash
   # In Emilia pipeline
   --batch-size 8

   # In transcriber
   batch_size=8
   ```
3. Use a smaller Whisper model:
   ```bash
   --whisper-arch medium  # instead of large-v3
   ```

### Performance Issues in WSL

See the detailed [WSL Setup Guide](wsl_setup.md) for:
- Environment variable configurations
- Driver requirements
- Performance optimizations
- Multi-GPU setup (if applicable)

### Multi-GPU Configuration

**Detect and Select GPU:**
```bash
# Interactive GPU manager
uv run python gpu_manager.py

# Or manually set GPU
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0
export CUDA_VISIBLE_DEVICES=1  # Use GPU 1
```

**Parallel Processing with Multiple GPUs:**

See [Multi-GPU Guide](MULTI_GPU_GUIDE.md) for detailed instructions on:
- Running multiple instances in parallel
- GPU selection and monitoring
- Performance optimization strategies

### Verification Script

Run the verification script to diagnose issues:
```bash
uv run python verify_installation.py
```

This provides detailed information about your setup and recommendations.

