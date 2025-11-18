# Fix: std::bad_alloc Error at Startup (Before Gradio Loads)

## El Problema

Con RTX 6000 Blackwell (96GB VRAM), el error `std::bad_alloc` ocurría **al arrancar** la aplicación, incluso antes de que Gradio se iniciara. Esto es particularmente problemático porque:

1. **No es falta de VRAM** - Tienes 96GB disponibles
2. **Ocurre durante imports** - Al importar torch, pyannote, whisperx
3. **Fragmentación de memoria** - CUDA allocator no está configurado antes de las primeras allocaciones

## Por Qué Pasaba

### Orden de Importación Incorrecto

```python
# INCORRECTO (código anterior)
if __name__ == "__main__":
    import transcriber                      # <- Importa whisperx
    from emilia_pipeline import ...         # <- Importa torch, pyannote
    # CUDA env NO está configurado todavía!
```

Cuando Python importa estos módulos:
1. `torch` se importa y inicializa CUDA allocator con configuración por defecto
2. `pyannote` carga modelos y aloca memoria
3. `whisperx` también aloca memoria
4. Todo esto ANTES de que `PYTORCH_CUDA_ALLOC_CONF` se configure
5. Resultado: **Fragmentación de memoria desde el inicio** → `std::bad_alloc`

## La Solución

### Nuevo Archivo: `setup_cuda_env.py`

Este archivo **DEBE** ser importado PRIMERO, antes de cualquier import de torch/pyannote:

```python
# setup_cuda_env.py
import os

def setup_cuda_environment():
    # Configuración para 96GB VRAM
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.8"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    # Optimizaciones WSL
    if "microsoft" in os.uname().release.lower():
        os.environ["MALLOC_TRIM_THRESHOLD_"] = "100000"
        os.environ["MALLOC_ARENA_MAX"] = "4"

# Se ejecuta al importar
setup_cuda_environment()
```

### Orden Correcto en `gradio_interface.py`

```python
# CORRECTO (código nuevo)
if __name__ == "__main__":
    import setup_cuda_env  # ← PRIMERO! Configura CUDA env

    # AHORA sí, importar torch/pyannote
    import transcriber
    from emilia_pipeline import ...
```

### Orden Correcto en `transcriber.py`

```python
def main():
    import setup_cuda_env  # ← PRIMERO!

    # Resto del código...
    model = load_whisperx_model("large-v3")
```

## Configuraciones Clave para 96GB VRAM

### `PYTORCH_CUDA_ALLOC_CONF`

```bash
expandable_segments:True           # Permite que segmentos crezcan dinámicamente
max_split_size_mb:512             # Chunks de 512MB (adecuado para 96GB)
garbage_collection_threshold:0.8   # GC agresivo al 80% de uso
```

**Por qué 512MB?**
- Con 96GB VRAM, chunks más grandes reducen overhead
- 512MB es un buen balance entre memoria y fragmentación
- Para 48GB VRAM, usar 256-384MB

### Variables de Entorno WSL

```bash
# En ~/.bashrc
export MALLOC_TRIM_THRESHOLD_=100000  # Previene fragmentación glibc
export MALLOC_ARENA_MAX=4             # Limita arenas malloc (multi-thread)
export OMP_NUM_THREADS=8              # Threads para operaciones CPU
export TOKENIZERS_PARALLELISM=false   # Evita problemas con HF tokenizers
```

## Mejoras Adicionales en Carga de Modelos

### En `transcriber.py`

```python
def load_whisperx_model(model_name="large-v2"):
    import torch

    # Limpiar ANTES de cargar
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Cargar modelo
    model = whisperx.load_model(...)

    # Limpiar DESPUÉS de cargar
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return model
```

### En `emilia_pipeline.py`

```python
def prepare_models(...):
    # Cargar cada modelo secuencialmente con limpieza entre cada uno

    diarisation = Pipeline.from_pretrained(...)
    diarisation.to(device)
    torch.cuda.empty_cache()  # ← Limpiar
    gc.collect()              # ← Limpiar

    asr_model = whisper_asr.load_asr_model(...)
    torch.cuda.empty_cache()  # ← Limpiar
    gc.collect()              # ← Limpiar

    vad_model = silero_vad.SileroVAD(...)
    torch.cuda.empty_cache()  # ← Limpiar
    gc.collect()              # ← Limpiar
```

## Verificación

Después de los cambios, ejecuta:

```bash
uv run python verify_installation.py
```

Deberías ver:

```
[CUDA Setup] Environment configured for RTX 6000 Blackwell (96GB VRAM)
[CUDA Setup] PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.8
✓ CUDA environment setup loaded

GPU detected: NVIDIA RTX 6000 Ada Generation with 96.00 GB VRAM
✓ Blackwell architecture detected (CC 9.0)
```

## Monitoreo Durante Inicio

Para verificar que no hay problemas de memoria durante inicio:

```bash
# Terminal 1: Monitorear GPU
watch -n 0.5 nvidia-smi

# Terminal 2: Iniciar aplicación
uv run python gradio_interface.py
```

Deberías ver que la memoria crece gradualmente sin spikes grandes.

## Troubleshooting

### Sigue dando std::bad_alloc al inicio

1. **Verifica que `setup_cuda_env` se importa PRIMERO:**
   ```python
   # Añadir print para debugging
   import setup_cuda_env  # Debería imprimir mensaje
   ```

2. **Verifica variables de entorno:**
   ```bash
   uv run python -c "import os; print(os.environ.get('PYTORCH_CUDA_ALLOC_CONF'))"
   ```

3. **Reduce max_split_size_mb:**
   ```python
   # En setup_cuda_env.py
   os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.8"
   ```

### Error persiste solo con Emilia Pipeline

El pipeline Emilia carga 3 modelos grandes simultáneamente. Solución:

```python
# En emilia_pipeline.py:prepare_models()
# Cargar modelos uno a uno con delays opcionales

import time

diarisation = Pipeline.from_pretrained(...)
diarisation.to(device)
torch.cuda.empty_cache()
gc.collect()
time.sleep(0.5)  # Opcional: dar tiempo al allocator

asr_model = ...
torch.cuda.empty_cache()
gc.collect()
time.sleep(0.5)  # Opcional
```

### WSL-Specific: "CUDA out of memory" pero nvidia-smi muestra memoria libre

Esto puede ser problema de WSL2 memory management. Solución:

```bash
# En Windows PowerShell (como Admin)
wsl --shutdown

# Crear/editar C:\Users\<tu-usuario>\.wslconfig
[wsl2]
memory=64GB           # Ajusta según tu RAM
swap=0                # Deshabilitar swap
localhostForwarding=true

# Reiniciar WSL
wsl
```

## Performance Esperado Post-Fix

Con los fixes aplicados:

- **Inicio de Gradio:** 30-60 segundos (carga de modelos)
- **Memoria peak durante inicio:** ~20-30GB VRAM
- **Sin errores std::bad_alloc**
- **Fragmentación mínima** (verificar con `nvidia-smi`)

## Referencias

- PyTorch CUDA allocator: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
- WSL2 + CUDA: https://docs.nvidia.com/cuda/wsl-user-guide/
- Configuración WSL: https://learn.microsoft.com/en-us/windows/wsl/wsl-config

## Resumen

1. ✅ **Crear `setup_cuda_env.py`** con configuración CUDA
2. ✅ **Importar `setup_cuda_env` PRIMERO** en todos los entry points
3. ✅ **Garbage collection agresivo** después de cada carga de modelo
4. ✅ **Configurar WSL** (si aplica) con .wslconfig
5. ✅ **Verificar con `verify_installation.py`**

Con estos cambios, el error `std::bad_alloc` al inicio debería desaparecer completamente, incluso con modelos grandes como pyannote y WhisperX large-v3.
