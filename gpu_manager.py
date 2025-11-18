#!/usr/bin/env python3
"""
GPU Manager - Utility for multi-GPU management and selection
Run this before starting the main application to select which GPU to use.
"""

import os
import subprocess
import sys
from typing import List, Dict, Optional


def get_gpu_info() -> List[Dict[str, str]]:
    """Get information about all available GPUs."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            print("Error: nvidia-smi failed")
            return []

        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 7:
                    gpus.append({
                        "id": parts[0],
                        "name": parts[1],
                        "memory_total": parts[2],
                        "memory_used": parts[3],
                        "memory_free": parts[4],
                        "utilization": parts[5],
                        "temperature": parts[6]
                    })
        return gpus
    except FileNotFoundError:
        print("Error: nvidia-smi not found. Is NVIDIA driver installed?")
        return []
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return []


def display_gpus(gpus: List[Dict[str, str]]) -> None:
    """Display GPU information in a formatted table."""
    if not gpus:
        print("No GPUs detected.")
        return

    print("\n" + "=" * 100)
    print("Available GPUs:")
    print("=" * 100)
    print(f"{'ID':<4} {'Name':<35} {'Memory':<20} {'Usage':<10} {'Temp':<8}")
    print("-" * 100)

    for gpu in gpus:
        memory_info = f"{gpu['memory_used']}MB / {gpu['memory_total']}MB"
        usage = f"{gpu['utilization']}%"
        temp = f"{gpu['temperature']}°C"
        print(f"{gpu['id']:<4} {gpu['name']:<35} {memory_info:<20} {usage:<10} {temp:<8}")

    print("=" * 100)


def select_gpu_interactive(gpus: List[Dict[str, str]]) -> Optional[str]:
    """Interactive GPU selection."""
    if not gpus:
        return None

    if len(gpus) == 1:
        print(f"\nOnly one GPU available: {gpus[0]['name']}")
        return "0"

    while True:
        print("\nSelect GPU to use:")
        print("  - Single GPU: Enter GPU ID (e.g., '0' or '1')")
        print("  - Multiple GPUs: Enter comma-separated IDs (e.g., '0,1')")
        print("  - All GPUs: Enter 'all'")
        print("  - Cancel: Enter 'q'")

        choice = input("\nYour choice: ").strip().lower()

        if choice == 'q':
            return None

        if choice == 'all':
            return ','.join([gpu['id'] for gpu in gpus])

        # Validate choice
        try:
            selected_ids = [id.strip() for id in choice.split(',')]
            available_ids = [gpu['id'] for gpu in gpus]

            if all(id in available_ids for id in selected_ids):
                return ','.join(selected_ids)
            else:
                print(f"Invalid GPU ID(s). Available: {', '.join(available_ids)}")
        except Exception:
            print("Invalid input format.")


def set_cuda_visible_devices(gpu_ids: str) -> None:
    """Set CUDA_VISIBLE_DEVICES environment variable."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    print(f"\n✓ CUDA_VISIBLE_DEVICES set to: {gpu_ids}")

    # Write to shell export format for easy copy-paste
    print("\nTo use this GPU selection in your current shell, run:")
    print(f"  export CUDA_VISIBLE_DEVICES={gpu_ids}")


def get_recommended_gpu(gpus: List[Dict[str, str]]) -> Optional[str]:
    """Recommend GPU with most free memory."""
    if not gpus:
        return None

    # Sort by free memory (descending)
    sorted_gpus = sorted(
        gpus,
        key=lambda g: int(g['memory_free']),
        reverse=True
    )

    best_gpu = sorted_gpus[0]
    print(f"\n💡 Recommended: GPU {best_gpu['id']} ({best_gpu['name']}) with {best_gpu['memory_free']}MB free")

    return best_gpu['id']


def show_multi_gpu_usage_guide():
    """Display guide for using multiple GPUs."""
    print("\n" + "=" * 100)
    print("Multi-GPU Usage Guide")
    print("=" * 100)
    print("""
The current implementation loads all models on a single GPU. However, you can run
multiple instances of the application in parallel, each using a different GPU.

Examples:

1. Run two instances on different GPUs:

   Terminal 1:
   $ export CUDA_VISIBLE_DEVICES=0
   $ uv run python gradio_interface.py --server-name 0.0.0.0 --server-port 7860

   Terminal 2:
   $ export CUDA_VISIBLE_DEVICES=1
   $ uv run python gradio_interface.py --server-name 0.0.0.0 --server-port 7861

2. Process different audio files in parallel:

   Terminal 1 (GPU 0):
   $ export CUDA_VISIBLE_DEVICES=0
   $ uv run python emilia_pipeline.py --config config.json --input-folder /path/to/batch1

   Terminal 2 (GPU 1):
   $ export CUDA_VISIBLE_DEVICES=1
   $ uv run python emilia_pipeline.py --config config.json --input-folder /path/to/batch2

3. For true multi-GPU data parallelism (advanced):
   - Requires code modifications to use torch.nn.DataParallel or DistributedDataParallel
   - Current implementation focuses on single-GPU optimization
   - Contact maintainer for multi-GPU training support

Note: With 96GB VRAM per GPU, a single GPU should be sufficient for most workloads.
Multi-GPU is mainly useful for processing multiple datasets in parallel.
""")
    print("=" * 100)


def main():
    """Main entry point for GPU manager."""
    print("\n" + "=" * 100)
    print("GPU Manager - Dataset Maker")
    print("=" * 100)

    # Get GPU info
    gpus = get_gpu_info()

    if not gpus:
        print("\n❌ No GPUs detected or nvidia-smi not available.")
        print("Please check your NVIDIA driver installation.")
        return 1

    # Display available GPUs
    display_gpus(gpus)

    # Get recommendation
    recommended = get_recommended_gpu(gpus)

    # Interactive selection
    selected = select_gpu_interactive(gpus)

    if selected is None:
        print("\nGPU selection cancelled.")
        return 0

    # Set environment variable
    set_cuda_visible_devices(selected)

    # Show multi-GPU guide if multiple GPUs detected
    if len(gpus) > 1:
        show_guide = input("\nShow multi-GPU usage guide? [y/N]: ").strip().lower()
        if show_guide == 'y':
            show_multi_gpu_usage_guide()

    print("\n✓ GPU configuration complete!")
    print("\nNext steps:")
    print("  1. Export the CUDA_VISIBLE_DEVICES variable in your shell (see above)")
    print("  2. Run your application:")
    print("     $ uv run python gradio_interface.py")
    print("     or")
    print("     $ uv run python emilia_pipeline.py --config Emilia/config.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
