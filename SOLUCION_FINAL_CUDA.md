# 🎉 Solución Final - WhisperX en CUDA Funcionando!

## Fecha: 2025-11-18

---

## ✅ ESTADO FINAL: TODO FUNCIONANDO EN GPU

```
✓ PyTorch 2.8.0+cu128 en CUDA
✓ ONNX Runtime 1.20.1 con CUDAExecutionProvider + TensorRT
✓ WhisperX 3.7.4 transcripción en CUDA
✓ ctranslate2 4.6.1 con cuDNN 9 support
✓ Pyannote Audio 3.4.0 sin torchcodec
✓ nvidia-cudnn-cu12 9.10.2.21 en entorno virtual
```

**Test de verificación**: 7/7 PASSED ✅

---

## 📊 Versiones Finales Instaladas

```toml
[dependencies]
torch[cu128] = "2.8.0"
whisperx = "3.7.4" (git main branch)
pyannote-audio = "3.4.0" (3.x sin torchcodec)
onnxruntime-gpu = "1.20.1"
ctranslate2 = "4.6.1"
nvidia-cudnn-cu12 = "9.10.2.21"
triton = "3.5.1"
```

---

## 🔧 Problemas Resueltos

### 1. ✅ ONNX Runtime CUDAExecutionProvider
**Problema**: Solo CPUExecutionProvider disponible
**Solución**: Eliminado conflicto onnxruntime vs onnxruntime-gpu
**Resultado**: CUDAExecutionProvider + TensorrtExecutionProvider funcionando

### 2. ✅ WhisperX/Pyannote std::bad_alloc
**Problema**: Crash al importar con std::bad_alloc
**Causa**: torchcodec wheel incompatible con WSL2
**Solución**: Downgrade pyannote-audio de 4.0.1 a 3.4.0
**Resultado**: Pyannote importa sin errores

### 3. ✅ WhisperX CUDA Inferencia
**Problema inicial**: ctranslate2 4.4.0 incompatible con cuDNN 9
**Problema encontrado**: ctranslate2 4.6.1 con cuDNN stub conflictivo
**Solución final**: Symlink cuDNN de nvidia-cudnn-cu12
**Resultado**: WhisperX transcripción en CUDA funcionando perfectamente

---

## 🎯 La Solución Clave: Fix cuDNN para ctranslate2

### El Problema Descubierto

ctranslate2 4.6.1 viene con un **stub de cuDNN incompleto** (126KB) en:
```
.venv/lib/python3.11/site-packages/ctranslate2.libs/libcudnn-74a4c495.so.9.1.0
```

Este stub conflictúa con el cuDNN completo de `nvidia-cudnn-cu12`, causando:
```
Unable to load any of {libcudnn_ops.so.9.1.0, ...}
Invalid handle. Cannot load symbol cudnnCreateTensorDescriptor
```

### La Solución Aplicada

1. **Instalar nvidia-cudnn-cu12** (ya hecho):
```bash
uv add "nvidia-cudnn-cu12>=9.1.0"
```

2. **Crear symlink en ctranslate2.libs**:
```bash
cd .venv/lib/python3.11/site-packages/ctranslate2.libs
ln -sf ../nvidia/cudnn/lib/libcudnn.so.9 libcudnn-74a4c495.so.9.1.0
```

3. **Crear symlinks de versiones en nvidia/cudnn/lib**:
```bash
cd .venv/lib/python3.11/site-packages/nvidia/cudnn/lib
ln -sf libcudnn_ops.so.9 libcudnn_ops.so.9.1.0
ln -sf libcudnn_ops.so.9 libcudnn_ops.so.9.1
ln -sf libcudnn_ops.so.9 libcudnn_ops.so
```

4. **Configurar LD_LIBRARY_PATH** (ya en setup_cuda_env.py):
```python
cudnn_lib_path = os.path.join(site_packages, "nvidia", "cudnn", "lib")
os.environ["LD_LIBRARY_PATH"] = f"{cudnn_lib_path}:..."
```

### Script Automatizado

Ejecuta el script para aplicar el fix automáticamente:
```bash
chmod +x fix_ctranslate2_cudnn.sh
./fix_ctranslate2_cudnn.sh
```

**IMPORTANTE**: Ejecutar este script después de cada `uv sync` que actualice ctranslate2.

---

## 📝 Archivos Modificados

### pyproject.toml
```toml
dependencies = [
    ...
    "whisperx>=3.7.0",
    "pyannote-audio>=3.1.1,<4.0",  # Sin torchcodec
    "onnxruntime-gpu==1.20.1",
    "nvidia-cudnn-cu12>=9.1.0",  # ← NUEVO
]

override-dependencies = [
    ...
    "ctranslate2>=4.6.0",  # ← NUEVO
    "torch==2.8.0",
    "torchaudio==2.8.0",
    "triton>=3.3.0",
]

[tool.uv.sources]
whisperx = { git = "https://github.com/m-bain/whisperX.git", branch = "main" }
```

### setup_cuda_env.py
- Añadido: Configuración de LD_LIBRARY_PATH para cuDNN
- Añadido: Detección automática de site-packages con sysconfig

### Emilia/models/whisper_asr.py
- Actualizado: Imports para WhisperX 3.7.4
- Cambiado: `from whisperx.types` → `from whisperx.asr`

---

## 🚀 Rendimiento Esperado

### GPU: NVIDIA GeForce RTX 4090 (24GB)
- **WhisperX tiny**: ~15-20x realtime en CUDA
- **WhisperX base**: ~10-15x realtime en CUDA
- **WhisperX small**: ~5-8x realtime en CUDA
- **WhisperX medium**: ~2-4x realtime en CUDA
- **Pyannote diarization**: GPU acelerada
- **ONNX models**: TensorRT acelerado

### GPU: NVIDIA RTX 6000 Blackwell (96GB)
- Puede procesar batch sizes mucho mayores
- Multiple streams simultáneos
- Modelos más grandes (large-v2, large-v3)

---

## 🧪 Tests de Verificación

### Test 1: Verificación General
```bash
uv run python verify_installation.py
```
**Esperado**: 7/7 PASSED ✅

### Test 2: WhisperX CUDA Específico
```bash
uv run python test_whisperx_cuda_full.py
```
**Esperado**: Transcripción completa en CUDA ✅

### Test 3: End-to-End Simplificado
```bash
uv run python test_simple_end_to_end.py
```
**Esperado**: 8/8 PASSED ✅

---

## 📦 Estructura del Entorno Virtual

```
.venv/lib/python3.11/site-packages/
├── nvidia/
│   └── cudnn/
│       └── lib/
│           ├── libcudnn.so.9 (122MB - COMPLETO)
│           ├── libcudnn_ops.so.9 (122MB - COMPLETO)
│           ├── libcudnn_ops.so.9.1.0 → libcudnn_ops.so.9 (symlink)
│           └── ... (otras bibliotecas cuDNN completas)
│
└── ctranslate2.libs/
    ├── libcudnn-74a4c495.so.9.1.0 → ../nvidia/cudnn/lib/libcudnn.so.9 (symlink)
    └── libcudnn-74a4c495.so.9.1.0.ORIGINAL (126KB - stub deshabilitado)
```

---

## 💡 Uso en Producción

### Configuración Recomendada para WhisperX:

```python
from Emilia.models.whisper_asr import load_asr_model

# Para máximo rendimiento en GPU
model = load_asr_model(
    whisper_arch="base.en",  # o "small.en", "medium.en"
    device="cuda",           # ← CUDA ahora funciona!
    compute_type="float16",  # FP16 para velocidad
    language="en"
)

# Transcribir
result = model.transcribe(
    audio=audio,
    vad_segments=vad_segments,
    batch_size=16,  # Ajustar según VRAM disponible
    language="en"
)
```

### Batch Sizes Recomendados:

**RTX 4090 (24GB)**:
- tiny/base: batch_size=24-32
- small: batch_size=16-24
- medium: batch_size=8-12

**RTX 6000 Blackwell (96GB)**:
- tiny/base: batch_size=64-96
- small: batch_size=48-64
- medium: batch_size=24-32
- large-v2: batch_size=12-16

---

## ⚠️ Mantenimiento

### Después de actualizar paquetes con `uv sync`:

1. **Si se actualiza ctranslate2**, ejecutar:
```bash
./fix_ctranslate2_cudnn.sh
```

2. **Verificar que todo funciona**:
```bash
uv run python test_whisperx_cuda_full.py
```

### Archivos a NO modificar manualmente:
- `.venv/lib/python3.11/site-packages/ctranslate2.libs/` (contiene symlinks)
- `.venv/lib/python3.11/site-packages/nvidia/cudnn/lib/` (contiene symlinks)

---

## 🎯 Resumen Ejecutivo

### Antes:
- ❌ ONNX Runtime: Solo CPU
- ❌ WhisperX: std::bad_alloc crash
- ❌ Pyannote: std::bad_alloc crash
- ❌ WhisperX CUDA: No funciona (cuDNN mismatch)

### Después:
- ✅ ONNX Runtime: CUDA + TensorRT
- ✅ WhisperX: Importa correctamente
- ✅ Pyannote: v3.4.0 sin torchcodec
- ✅ WhisperX CUDA: Transcripción funcionando en GPU!

### Problemas Resueltos:
1. ✅ Conflicto onnxruntime packages
2. ✅ torchcodec incompatibility en WSL2
3. ✅ ctranslate2 cuDNN version mismatch
4. ✅ ctranslate2 cuDNN stub incompleto conflictivo

### Resultado:
🚀 **Sistema completamente funcional en GPU con RTX 6000 Blackwell + RTX 4090**

---

## 📚 Referencias

- **Issue clave**: [OpenNMT/CTranslate2#1826](https://github.com/OpenNMT/CTranslate2/issues/1826)
- **WhisperX cuDNN**: [m-bain/whisperX#1225](https://github.com/m-bain/whisperX/issues/1225)
- **Upgrade ctranslate2**: [m-bain/whisperX#1158](https://github.com/m-bain/whisperX/issues/1158)
- **cuDNN 9 support**: [OpenNMT/CTranslate2#1780](https://github.com/OpenNMT/CTranslate2/issues/1780)

---

**Creado**: 2025-11-18
**Estado**: ✅ COMPLETAMENTE FUNCIONAL
**GPUs**: RTX 6000 Blackwell (96GB) + RTX 4090 (24GB)
**Sistema**: WSL2 (Ubuntu) con CUDA 12.8
