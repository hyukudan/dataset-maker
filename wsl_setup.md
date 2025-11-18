# WSL Setup Guide for RTX 6000 Blackwell

Este documento proporciona instrucciones específicas para configurar el Dataset Maker en WSL con RTX 6000 Blackwell.

## Requisitos Previos

1. **WSL2** instalado (no WSL1)
2. **NVIDIA Driver** instalado en Windows (versión 545.84 o superior para Blackwell)
3. **CUDA Toolkit 12.8** en WSL2

### Verificar Driver NVIDIA en Windows

```powershell
# En PowerShell (Windows)
nvidia-smi
```

Deberías ver tu RTX 6000 Blackwell listada.

### Verificar CUDA en WSL

```bash
# En WSL2
nvidia-smi
nvcc --version  # Opcional, solo si instalaste CUDA Toolkit
```

## Instalación Paso a Paso

### 1. Configurar Variables de Entorno

Añade esto a tu `~/.bashrc` o `~/.zshrc`:

```bash
# CUDA Configuration for RTX 6000 Blackwell
export CUDA_VISIBLE_DEVICES=0  # Ajusta si tienes múltiples GPUs
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

# Optimizations for WSL
export CUDA_LAUNCH_BLOCKING=0  # Async execution

# Prevent memory fragmentation
export MALLOC_TRIM_THRESHOLD_=100000
```

Recarga tu shell:
```bash
source ~/.bashrc  # o source ~/.zshrc
```

### 2. Instalar Dataset Maker

```bash
# Clonar repositorio
git clone https://github.com/JarodMica/dataset-maker.git
cd dataset-maker

# Crear checkout de la rama optimizada
git checkout claude/optimize-rtx6000-blackwell-01QJPCmY29AKERpafq3RKPGz

# Instalar con uv (asegúrate de tener uv instalado)
uv sync
```

### 3. Verificar Instalación

```bash
# Verificar que todo esté correctamente instalado
uv run python verify_installation.py
```

Este script verificará:
- ✓ Python 3.10-3.12
- ✓ PyTorch con CUDA 12.8
- ✓ ONNX Runtime con CUDAExecutionProvider
- ✓ WhisperX
- ✓ Pyannote Audio
- ✓ Todas las dependencias

### 4. Configurar ONNX Runtime (si es necesario)

Si `verify_installation.py` muestra que ONNX Runtime no tiene CUDAExecutionProvider:

```bash
uv run python setup_onnx_cuda.py
```

Este script reinstalará ONNX Runtime con soporte CUDA correcto.

## Optimizaciones Específicas para RTX 6000 Blackwell

### Batch Sizes Recomendados

**Para RTX 6000 Blackwell (96GB VRAM):**

```python
# En emilia_pipeline.py o transcriber.py
batch_size = 24-32          # Para WhisperX y Emilia
chunk_size = 25-30          # Para WhisperX
```

**Para RTX 6000 Ada (48GB VRAM):**

```python
# En emilia_pipeline.py o transcriber.py
batch_size = 16-24          # Para WhisperX y Emilia
chunk_size = 20-30          # Para WhisperX
```

### TF32 (Habilitado Automáticamente)

El código ahora habilita automáticamente TF32 para arquitectura Blackwell (Compute Capability 9.0+):

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

Esto mejora el rendimiento en un ~20-30% sin perder precisión significativa.

### Multi-GPU (Opcional)

Si tienes múltiples GPUs RTX 6000:

```bash
# Ver GPUs disponibles
nvidia-smi

# Usar GPU específica
export CUDA_VISIBLE_DEVICES=0  # Primera GPU
export CUDA_VISIBLE_DEVICES=1  # Segunda GPU
export CUDA_VISIBLE_DEVICES=0,1  # Ambas GPUs (experimental)
```

**Nota:** El pipeline actual no tiene soporte nativo multi-GPU, pero puedes ejecutar instancias paralelas en GPUs diferentes.

## Problemas Comunes y Soluciones

### std::bad_alloc (Agotamiento de Memoria)

**Síntoma:** Error `std::bad_alloc` aunque tengas 48GB VRAM

**Solución:**
1. El código ahora incluye garbage collection agresivo
2. Reduce batch_size si persiste:
   ```bash
   # En gradio_interface.py o al ejecutar scripts
   batch_size=8  # En lugar de 16
   ```

3. Configura memory allocation:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
   ```

### ONNX Runtime sin CUDA Provider

**Síntoma:** `Available providers: ['CPUExecutionProvider']`

**Solución:**
```bash
uv run python setup_onnx_cuda.py
# O manualmente:
uv remove onnxruntime
uv add onnxruntime-gpu==1.20.1
```

### WhisperX Lento o con Errores

**Síntoma:** Transcripción muy lenta o errores de memoria

**Solución:**
1. Reduce `chunk_size` y `batch_size`:
   ```python
   chunk_size=15
   batch_size=8
   ```

2. Usa modelo Whisper más pequeño:
   ```bash
   --whisper-arch medium  # En lugar de large-v3
   ```

### Pyannote Diarization Falla

**Síntoma:** Error al cargar modelo de diarización

**Solución:**
1. Verifica token de Hugging Face:
   ```bash
   # En config.json
   "huggingface_token": "hf_..."

   # O como variable de entorno
   export HF_TOKEN="hf_..."
   ```

2. Descarga manual si es necesario:
   ```bash
   uv run python -c "from pyannote.audio import Pipeline; Pipeline.from_pretrained('pyannote/speaker-diarization-community-1', token='hf_...')"
   ```

## Monitoreo de Recursos

### Monitorear GPU en Tiempo Real

```bash
# Terminal separada
watch -n 1 nvidia-smi

# O con más detalles
nvidia-smi dmon -s pucvmet
```

### Monitorear Memoria Python

```python
# Añadir al código para debugging
import psutil
import torch

print(f"RAM: {psutil.virtual_memory().percent}%")
print(f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"VRAM Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

## Performance Benchmarks

En RTX 6000 Blackwell (48GB), deberías ver aproximadamente:

- **Pyannote Diarization:** ~5-10 min por hora de audio
- **WhisperX Transcription:** ~1-3 min por hora de audio (large-v3)
- **UVR Separation:** ~2-4 min por hora de audio
- **Pipeline Completo:** ~10-20 min por hora de audio

Si ves tiempos significativamente peores, revisa:
1. Batch sizes muy pequeños
2. GPU no siendo utilizada (verificar con `nvidia-smi`)
3. Throttling térmico (verificar temperaturas)

## Actualizar Instalación

```bash
cd dataset-maker
git pull origin claude/optimize-rtx6000-blackwell-01QJPCmY29AKERpafq3RKPGz
uv sync
uv run python verify_installation.py
```

## Soporte

Si encuentras problemas:

1. Ejecuta `uv run python verify_installation.py` y guarda el output
2. Revisa esta guía para soluciones comunes
3. Verifica logs en la carpeta `logs/` de tu proyecto
4. Crea un issue en el repositorio con:
   - Output de `verify_installation.py`
   - Output de `nvidia-smi`
   - Mensaje de error completo
   - Configuración utilizada (batch_size, modelo, etc.)
