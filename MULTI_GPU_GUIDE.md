# Guía de Uso Multi-GPU para Dataset Maker

## Detección Automática de GPUs

El código ahora detecta automáticamente todas las GPUs disponibles y muestra información detallada al inicio.

### Al Iniciar

```bash
uv run python gradio_interface.py
```

Verás algo como:

```
[CUDA Setup] Detected 2 GPUs:
  GPU 0: NVIDIA RTX 6000 Ada Generation (98304 MiB)
  GPU 1: NVIDIA RTX 6000 Ada Generation (98304 MiB)
[CUDA Setup] Using GPU 0 by default. Set CUDA_VISIBLE_DEVICES to change.

...

Using GPU 0: NVIDIA RTX 6000 Ada Generation
  VRAM: 96.00 GB
  Compute Capability: 9.0
  CUDA_VISIBLE_DEVICES: 0
Other available GPUs:
  GPU 1: NVIDIA RTX 6000 Ada Generation (96.00 GB)
To use a different GPU, set CUDA_VISIBLE_DEVICES before running
```

---

## Seleccionar GPU Específica

### Método 1: GPU Manager Interactivo (Recomendado)

```bash
# Ejecutar el gestor de GPUs
uv run python gpu_manager.py
```

Esto muestra:
- Estado de cada GPU (memoria usada/libre)
- Temperatura y utilización
- Recomendación de cuál usar
- Selección interactiva

**Ejemplo de output:**
```
====================================================================================================
Available GPUs:
====================================================================================================
ID   Name                                Memory               Usage      Temp
----------------------------------------------------------------------------------------------------
0    NVIDIA RTX 6000 Ada Generation     2048MB / 98304MB     5%         45°C
1    NVIDIA RTX 6000 Ada Generation     45000MB / 98304MB    78%        72°C
====================================================================================================

💡 Recommended: GPU 0 (NVIDIA RTX 6000 Ada Generation) with 96256MB free

Select GPU to use:
  - Single GPU: Enter GPU ID (e.g., '0' or '1')
  - Multiple GPUs: Enter comma-separated IDs (e.g., '0,1')
  - All GPUs: Enter 'all'
  - Cancel: Enter 'q'

Your choice: 0

✓ CUDA_VISIBLE_DEVICES set to: 0

To use this GPU selection in your current shell, run:
  export CUDA_VISIBLE_DEVICES=0
```

Luego copiar el comando `export` y ejecutar tu aplicación.

### Método 2: Variable de Entorno Manual

```bash
# Usar GPU 0
export CUDA_VISIBLE_DEVICES=0
uv run python gradio_interface.py

# Usar GPU 1
export CUDA_VISIBLE_DEVICES=1
uv run python gradio_interface.py

# Usar múltiples GPUs (ver sección avanzada)
export CUDA_VISIBLE_DEVICES=0,1
```

### Método 3: En Línea de Comando

```bash
# Una sola línea
CUDA_VISIBLE_DEVICES=0 uv run python gradio_interface.py

# Emilia pipeline con GPU específica
CUDA_VISIBLE_DEVICES=1 uv run python emilia_pipeline.py --config Emilia/config.json
```

---

## Procesamiento Paralelo con Múltiples GPUs

La implementación actual carga **todos los modelos en una sola GPU**. Sin embargo, con 96GB VRAM por GPU, puedes ejecutar **múltiples instancias en paralelo**.

### Caso de Uso 1: Procesar Diferentes Datasets en Paralelo

```bash
# Terminal 1 - GPU 0
export CUDA_VISIBLE_DEVICES=0
uv run python emilia_pipeline.py --config Emilia/config.json --input-folder /path/to/dataset1

# Terminal 2 - GPU 1
export CUDA_VISIBLE_DEVICES=1
uv run python emilia_pipeline.py --config Emilia/config.json --input-folder /path/to/dataset2
```

**Resultado:** Procesas 2 datasets simultáneamente, cada uno usando una GPU diferente.

### Caso de Uso 2: Múltiples Instancias de Gradio

```bash
# Terminal 1 - GPU 0, Puerto 7860
export CUDA_VISIBLE_DEVICES=0
uv run python gradio_interface.py

# Terminal 2 - GPU 1, Puerto 7861
export CUDA_VISIBLE_DEVICES=1
uv run python gradio_interface.py
```

**Nota:** Gradio automáticamente asignará puertos diferentes si 7860 está ocupado.

### Caso de Uso 3: Batch Processing con Script

Crear `process_parallel.sh`:

```bash
#!/bin/bash

# Lista de datasets a procesar
DATASETS=(
    "/path/to/dataset1"
    "/path/to/dataset2"
    "/path/to/dataset3"
    "/path/to/dataset4"
)

# Número de GPUs disponibles
NUM_GPUS=2

# Procesar en paralelo
for i in "${!DATASETS[@]}"; do
    GPU_ID=$((i % NUM_GPUS))
    DATASET="${DATASETS[$i]}"

    echo "Processing $DATASET on GPU $GPU_ID"

    CUDA_VISIBLE_DEVICES=$GPU_ID uv run python emilia_pipeline.py \
        --config Emilia/config.json \
        --input-folder "$DATASET" \
        --batch-size 24 &

    # Esperar un poco antes de lanzar el siguiente
    sleep 5
done

# Esperar a que todos terminen
wait
echo "All datasets processed!"
```

Ejecutar:
```bash
chmod +x process_parallel.sh
./process_parallel.sh
```

---

## Monitoreo de GPUs

### Ver Estado en Tiempo Real

```bash
# Monitoreo cada 1 segundo
watch -n 1 nvidia-smi

# O con más detalles
nvidia-smi dmon -s pucvmet
```

### Ver Qué Proceso Usa Qué GPU

```bash
nvidia-smi pmon
```

Output:
```
# gpu   pid  type    sm   mem   enc   dec   command
# Idx     #   C/G     %     %     %     %   name
    0  1234     C    85    45     0     0   python
    1  5678     C    92    67     0     0   python
```

---

## Configuración WSL Multi-GPU

### .wslconfig para 2+ GPUs

```ini
# C:\Users\<tu-usuario>\.wslconfig
[wsl2]
memory=128GB          # Para 2x GPUs con 96GB cada una
swap=0
processors=16
localhostForwarding=true

[experimental]
autoMemoryReclaim=gradual
sparseVhd=true
```

### Verificar GPUs en WSL

```bash
# Debería mostrar todas las GPUs
nvidia-smi

# Ver CUDA devices disponibles
uv run python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

---

## Performance con Multi-GPU

### Throughput Esperado

**Single GPU (96GB):**
- Emilia Pipeline: ~10-20 min/hora de audio
- Batch size: 24-32

**Dual GPU (2x 96GB) en Paralelo:**
- Emilia Pipeline: ~5-10 min/hora de audio (por GPU)
- Throughput total: **2x** el de single GPU
- Batch size por GPU: 24-32

**Nota:** El speedup es casi lineal porque cada GPU procesa datasets independientes.

### Cuándo Usar Multi-GPU

✅ **SÍ usar multi-GPU cuando:**
- Tienes múltiples datasets para procesar
- Necesitas procesar gran volumen de audio
- Quieres reducir tiempo total de procesamiento
- Cada GPU tiene suficiente VRAM (96GB es más que suficiente)

❌ **NO necesitas multi-GPU si:**
- Solo procesas un dataset pequeño
- Tu bottleneck es CPU (ej. I/O, preprocessing)
- Una sola GPU ya maneja tu workload cómodamente

---

## Troubleshooting Multi-GPU

### "RuntimeError: CUDA out of memory" con múltiples GPUs visibles

**Problema:** Configuraste `CUDA_VISIBLE_DEVICES=0,1` pero el código intenta usar ambas y se queda sin memoria.

**Solución:** El código actual usa solo GPU 0. Para usar GPU 1, configura:
```bash
export CUDA_VISIBLE_DEVICES=1  # Solo GPU 1
```

### No veo todas mis GPUs en nvidia-smi (WSL)

**Solución:**
```bash
# Verificar driver
nvidia-smi

# Si falla, reinstalar driver NVIDIA en Windows
# Luego en WSL:
sudo apt update
sudo apt install nvidia-cuda-toolkit
```

### Proceso usa GPU incorrecta

**Verificar:**
```bash
# Ver CUDA_VISIBLE_DEVICES actual
echo $CUDA_VISIBLE_DEVICES

# Ver procesos por GPU
nvidia-smi pmon
```

**Fix:**
```bash
# Asegurarse de exportar la variable
export CUDA_VISIBLE_DEVICES=1

# Verificar antes de ejecutar
echo $CUDA_VISIBLE_DEVICES
uv run python gradio_interface.py
```

---

## Implementación Futura: Data Parallelism Nativo

Actualmente, cada instancia usa una GPU completa. Para implementar data parallelism nativo:

```python
# Futuro: Usar torch.nn.DataParallel
model = torch.nn.DataParallel(model, device_ids=[0, 1])

# O mejor: DistributedDataParallel
torch.distributed.init_process_group(backend='nccl')
model = torch.nn.parallel.DistributedDataParallel(model)
```

Esto requeriría modificaciones significativas al pipeline actual. Si necesitas esta funcionalidad, abre un issue en el repositorio.

---

## Resumen de Comandos Útiles

```bash
# Ver GPUs disponibles con manager
uv run python gpu_manager.py

# Usar GPU específica (método rápido)
CUDA_VISIBLE_DEVICES=0 uv run python gradio_interface.py

# Procesar en paralelo (2 GPUs)
CUDA_VISIBLE_DEVICES=0 uv run python emilia_pipeline.py --input-folder batch1 &
CUDA_VISIBLE_DEVICES=1 uv run python emilia_pipeline.py --input-folder batch2 &

# Monitorear GPUs
watch -n 1 nvidia-smi

# Ver procesos por GPU
nvidia-smi pmon
```

---

## Referencias

- PyTorch Multi-GPU: https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html
- CUDA Best Practices: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- nvidia-smi Guide: https://developer.nvidia.com/nvidia-system-management-interface
