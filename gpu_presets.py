"""
GPU Preset Profiles - Auto-configuration for different GPU architectures
Provides one-click optimization for Blackwell, Ada, Ampere, and conservative modes
"""

import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class PresetConfig:
    """Configuration preset for a specific GPU profile"""
    name: str
    description: str
    emilia_batch_size: int
    whisper_chunk_size: int
    transcriber_batch_size: int
    whisper_model: str
    compute_type: str
    uvr_separation: bool
    emilia_threads: int
    min_segment_duration: float
    max_split_size_mb: int
    gc_threshold: float
    recommended_for: List[str]


# Predefined presets
PRESETS = {
    "blackwell_96gb": PresetConfig(
        name="Blackwell 96GB Ultra",
        description="Maximum performance for RTX 6000 Blackwell with 96GB VRAM",
        emilia_batch_size=30,
        whisper_chunk_size=28,
        transcriber_batch_size=30,
        whisper_model="large-v3",
        compute_type="float16",
        uvr_separation=True,
        emilia_threads=6,
        min_segment_duration=0.25,
        max_split_size_mb=512,
        gc_threshold=0.8,
        recommended_for=["RTX 6000 Blackwell", "H100", "A100 80GB"]
    ),

    "ada_48gb": PresetConfig(
        name="Ada 48GB High Performance",
        description="Optimized for RTX 6000 Ada Generation with 48GB VRAM",
        emilia_batch_size=24,
        whisper_chunk_size=24,
        transcriber_batch_size=24,
        whisper_model="large-v3",
        compute_type="float16",
        uvr_separation=True,
        emilia_threads=4,
        min_segment_duration=0.25,
        max_split_size_mb=384,
        gc_threshold=0.85,
        recommended_for=["RTX 6000 Ada", "RTX 5000 Ada", "A100 40GB"]
    ),

    "ampere_24gb": PresetConfig(
        name="Ampere 24GB Balanced",
        description="Balanced settings for A10, A40, RTX A5000 (24GB VRAM)",
        emilia_batch_size=16,
        whisper_chunk_size=20,
        transcriber_batch_size=16,
        whisper_model="large-v3",
        compute_type="float16",
        uvr_separation=True,
        emilia_threads=4,
        min_segment_duration=0.3,
        max_split_size_mb=256,
        gc_threshold=0.9,
        recommended_for=["A10", "A40", "RTX A5000", "RTX 4090"]
    ),

    "conservative": PresetConfig(
        name="Conservative (Any GPU)",
        description="Safe settings for GPUs with <24GB VRAM or unknown architecture",
        emilia_batch_size=8,
        whisper_chunk_size=16,
        transcriber_batch_size=8,
        whisper_model="medium",
        compute_type="float16",
        uvr_separation=False,
        emilia_threads=2,
        min_segment_duration=0.5,
        max_split_size_mb=128,
        gc_threshold=0.95,
        recommended_for=["RTX 3090", "RTX 3080", "V100", "T4", "Any GPU <24GB"]
    ),

    "quality_focused": PresetConfig(
        name="Quality Focused (Slow)",
        description="Maximum quality, slower processing - for final production datasets",
        emilia_batch_size=4,
        whisper_chunk_size=8,
        transcriber_batch_size=4,
        whisper_model="large-v3",
        compute_type="float32",
        uvr_separation=True,
        emilia_threads=8,
        min_segment_duration=0.1,
        max_split_size_mb=256,
        gc_threshold=0.9,
        recommended_for=["Any GPU with 16GB+ VRAM"]
    ),
}


def detect_optimal_preset() -> Tuple[str, PresetConfig, Dict[str, any]]:
    """
    Automatically detect the best preset based on available GPU
    Returns: (preset_key, preset_config, gpu_info)
    """
    gpu_info = {}

    try:
        import torch

        if not torch.cuda.is_available():
            return "conservative", PRESETS["conservative"], gpu_info

        # Get GPU properties
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024**3)
        compute_cap = props.major * 10 + props.minor
        gpu_name = props.name

        gpu_info = {
            "name": gpu_name,
            "vram_gb": vram_gb,
            "compute_cap": compute_cap,
        }

        # Determine architecture
        if compute_cap >= 90:
            arch = "Blackwell"
        elif compute_cap >= 89:
            arch = "Ada"
        elif compute_cap >= 80:
            arch = "Ampere"
        else:
            arch = "Legacy"

        gpu_info["architecture"] = arch

        # Select optimal preset
        if arch == "Blackwell" and vram_gb >= 80:
            return "blackwell_96gb", PRESETS["blackwell_96gb"], gpu_info
        elif arch == "Ada" and vram_gb >= 40:
            return "ada_48gb", PRESETS["ada_48gb"], gpu_info
        elif vram_gb >= 20:
            return "ampere_24gb", PRESETS["ampere_24gb"], gpu_info
        else:
            return "conservative", PRESETS["conservative"], gpu_info

    except ImportError:
        # PyTorch not available
        return "conservative", PRESETS["conservative"], gpu_info
    except Exception as e:
        gpu_info["error"] = str(e)
        return "conservative", PRESETS["conservative"], gpu_info


def apply_preset_to_environment(preset: PresetConfig) -> Dict[str, str]:
    """
    Apply preset configuration to environment variables
    Returns: Dictionary of applied settings
    """
    applied = {}

    # CUDA memory allocator settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        f"expandable_segments:True,"
        f"max_split_size_mb:{preset.max_split_size_mb},"
        f"garbage_collection_threshold:{preset.gc_threshold}"
    )
    applied["PYTORCH_CUDA_ALLOC_CONF"] = os.environ["PYTORCH_CUDA_ALLOC_CONF"]

    # Other optimizations
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    applied["CUDA_LAUNCH_BLOCKING"] = "0"

    return applied


def get_preset_as_dict(preset: PresetConfig) -> Dict[str, any]:
    """Convert preset to dictionary for UI or export"""
    return {
        "name": preset.name,
        "description": preset.description,
        "emilia_batch_size": preset.emilia_batch_size,
        "whisper_chunk_size": preset.whisper_chunk_size,
        "transcriber_batch_size": preset.transcriber_batch_size,
        "whisper_model": preset.whisper_model,
        "compute_type": preset.compute_type,
        "uvr_separation": preset.uvr_separation,
        "emilia_threads": preset.emilia_threads,
        "min_segment_duration": preset.min_segment_duration,
        "max_split_size_mb": preset.max_split_size_mb,
        "gc_threshold": preset.gc_threshold,
        "recommended_for": preset.recommended_for,
    }


def format_preset_summary(preset_key: str, gpu_info: Dict[str, any] = None) -> str:
    """
    Generate formatted summary of preset configuration
    Returns: Formatted string for display
    """
    if preset_key not in PRESETS:
        return f"Unknown preset: {preset_key}"

    preset = PRESETS[preset_key]
    lines = []

    lines.append("=" * 80)
    lines.append(f"PRESET: {preset.name}")
    lines.append("=" * 80)
    lines.append(f"\n{preset.description}\n")

    if gpu_info:
        lines.append("📊 Detected GPU:")
        lines.append(f"   Name: {gpu_info.get('name', 'Unknown')}")
        lines.append(f"   VRAM: {gpu_info.get('vram_gb', 0):.1f} GB")
        lines.append(f"   Architecture: {gpu_info.get('architecture', 'Unknown')}")
        lines.append("")

    lines.append("⚙️  Configuration:")
    lines.append(f"   Emilia Batch Size: {preset.emilia_batch_size}")
    lines.append(f"   Whisper Chunk Size: {preset.whisper_chunk_size}")
    lines.append(f"   Transcriber Batch Size: {preset.transcriber_batch_size}")
    lines.append(f"   Whisper Model: {preset.whisper_model}")
    lines.append(f"   Compute Type: {preset.compute_type}")
    lines.append(f"   UVR Separation: {'Enabled' if preset.uvr_separation else 'Disabled'}")
    lines.append(f"   Threads: {preset.emilia_threads}")
    lines.append(f"   Min Segment Duration: {preset.min_segment_duration}s")
    lines.append("")

    lines.append("🧠 Memory Settings:")
    lines.append(f"   Max Split Size: {preset.max_split_size_mb} MB")
    lines.append(f"   GC Threshold: {preset.gc_threshold}")
    lines.append("")

    lines.append("✅ Recommended For:")
    for gpu_name in preset.recommended_for:
        lines.append(f"   • {gpu_name}")

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


def save_custom_preset(name: str, config_dict: Dict[str, any], path: Optional[Path] = None) -> str:
    """
    Save a custom preset configuration
    Returns: Success message or error
    """
    if path is None:
        path = Path.cwd() / "custom_presets"
        path.mkdir(exist_ok=True)

    filename = path / f"{name.lower().replace(' ', '_')}.json"

    try:
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=2)
        return f"✅ Custom preset saved to: {filename}"
    except Exception as e:
        return f"❌ Error saving preset: {str(e)}"


def load_custom_preset(name: str, path: Optional[Path] = None) -> Tuple[bool, Dict[str, any], str]:
    """
    Load a custom preset configuration
    Returns: (success, config_dict, message)
    """
    if path is None:
        path = Path.cwd() / "custom_presets"

    filename = path / f"{name.lower().replace(' ', '_')}.json"

    if not filename.exists():
        return False, {}, f"❌ Preset file not found: {filename}"

    try:
        with open(filename, 'r') as f:
            config = json.load(f)
        return True, config, f"✅ Loaded custom preset: {name}"
    except Exception as e:
        return False, {}, f"❌ Error loading preset: {str(e)}"


def list_all_presets() -> Dict[str, str]:
    """
    List all available presets (built-in and custom)
    Returns: Dictionary of {preset_key: description}
    """
    all_presets = {
        key: preset.description
        for key, preset in PRESETS.items()
    }

    # Add custom presets
    custom_path = Path.cwd() / "custom_presets"
    if custom_path.exists():
        for preset_file in custom_path.glob("*.json"):
            preset_name = preset_file.stem
            all_presets[f"custom_{preset_name}"] = f"Custom: {preset_name}"

    return all_presets


if __name__ == "__main__":
    # Test detection and display optimal preset
    preset_key, preset, gpu_info = detect_optimal_preset()
    print(format_preset_summary(preset_key, gpu_info))
