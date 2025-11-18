"""
Configuration Manager - Export/Import system settings
Allows users to save optimal configurations and transfer between machines
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import platform


def get_current_system_info() -> Dict[str, any]:
    """
    Collect current system information
    Returns: Dictionary with system details
    """
    info = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
    }

    # Try to get GPU info
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = {
                "name": props.name,
                "vram_gb": props.total_memory / (1024**3),
                "compute_capability": f"{props.major}.{props.minor}",
                "cuda_version": torch.version.cuda,
            }
    except:
        info["gpu"] = None

    # Environment variables
    info["env_vars"] = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "CUDA_LAUNCH_BLOCKING": os.environ.get("CUDA_LAUNCH_BLOCKING"),
    }

    return info


def export_configuration(
    config: Dict[str, any],
    export_path: Optional[Path] = None,
    include_system_info: bool = True
) -> Tuple[bool, str]:
    """
    Export configuration to JSON file

    Args:
        config: Configuration dictionary to export
        export_path: Optional custom path (defaults to configs/ folder)
        include_system_info: Include system information in export

    Returns:
        (success, message/path)
    """
    if export_path is None:
        export_dir = Path.cwd() / "saved_configs"
        export_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = export_dir / f"config_{timestamp}.json"

    export_data = {
        "version": "1.0",
        "config": config,
    }

    if include_system_info:
        export_data["system_info"] = get_current_system_info()

    try:
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        return True, str(export_path)
    except Exception as e:
        return False, f"Error exporting configuration: {str(e)}"


def import_configuration(import_path: Path) -> Tuple[bool, Dict[str, any], str]:
    """
    Import configuration from JSON file

    Args:
        import_path: Path to configuration file

    Returns:
        (success, config_dict, message)
    """
    if not import_path.exists():
        return False, {}, f"Configuration file not found: {import_path}"

    try:
        with open(import_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if "config" not in data:
            return False, {}, "Invalid configuration file format"

        config = data["config"]
        system_info = data.get("system_info", {})

        # Generate info message
        message_lines = [f"✅ Configuration loaded from: {import_path.name}"]

        if system_info:
            message_lines.append(f"\n📋 Original System:")
            message_lines.append(f"   Created: {system_info.get('timestamp', 'Unknown')}")
            message_lines.append(f"   Platform: {system_info.get('platform', 'Unknown')}")

            if system_info.get("gpu"):
                gpu = system_info["gpu"]
                message_lines.append(f"   GPU: {gpu.get('name', 'Unknown')} ({gpu.get('vram_gb', 0):.0f}GB)")

        message = "\n".join(message_lines)

        return True, config, message

    except json.JSONDecodeError as e:
        return False, {}, f"Invalid JSON format: {str(e)}"
    except Exception as e:
        return False, {}, f"Error importing configuration: {str(e)}"


def list_saved_configurations() -> List[Dict[str, any]]:
    """
    List all saved configurations with their metadata

    Returns:
        List of configuration metadata dictionaries
    """
    configs_dir = Path.cwd() / "saved_configs"
    if not configs_dir.exists():
        return []

    configs = []

    for config_file in configs_dir.glob("*.json"):
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)

            metadata = {
                "filename": config_file.name,
                "path": str(config_file),
                "size_kb": config_file.stat().st_size / 1024,
                "modified": datetime.fromtimestamp(config_file.stat().st_mtime).isoformat(),
            }

            # Extract system info if available
            if "system_info" in data:
                sys_info = data["system_info"]
                metadata["platform"] = sys_info.get("platform", "Unknown")
                if sys_info.get("gpu"):
                    metadata["gpu_name"] = sys_info["gpu"].get("name", "Unknown")
                    metadata["vram_gb"] = sys_info["gpu"].get("vram_gb", 0)

            configs.append(metadata)

        except Exception:
            continue

    # Sort by modification time (newest first)
    configs.sort(key=lambda x: x["modified"], reverse=True)

    return configs


def validate_configuration(config: Dict[str, any]) -> Tuple[bool, List[str]]:
    """
    Validate configuration dictionary

    Args:
        config: Configuration dictionary to validate

    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    required_fields = [
        "emilia_batch_size",
        "whisper_chunk_size",
        "transcriber_batch_size",
        "whisper_model",
    ]

    # Check required fields
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")

    # Validate ranges
    if "emilia_batch_size" in config:
        if not isinstance(config["emilia_batch_size"], int) or config["emilia_batch_size"] < 1:
            errors.append("emilia_batch_size must be a positive integer")

    if "whisper_chunk_size" in config:
        if not isinstance(config["whisper_chunk_size"], int) or config["whisper_chunk_size"] < 1:
            errors.append("whisper_chunk_size must be a positive integer")

    if "transcriber_batch_size" in config:
        if not isinstance(config["transcriber_batch_size"], int) or config["transcriber_batch_size"] < 1:
            errors.append("transcriber_batch_size must be a positive integer")

    if "whisper_model" in config:
        valid_models = ["small", "medium", "large-v2", "large-v3"]
        if config["whisper_model"] not in valid_models:
            errors.append(f"whisper_model must be one of: {', '.join(valid_models)}")

    return len(errors) == 0, errors


def compare_configurations(config1: Dict[str, any], config2: Dict[str, any]) -> str:
    """
    Compare two configurations and return formatted diff

    Args:
        config1: First configuration (e.g., current)
        config2: Second configuration (e.g., imported)

    Returns:
        Formatted comparison string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("CONFIGURATION COMPARISON")
    lines.append("=" * 80)

    all_keys = set(config1.keys()) | set(config2.keys())

    for key in sorted(all_keys):
        val1 = config1.get(key, "NOT SET")
        val2 = config2.get(key, "NOT SET")

        if val1 == val2:
            lines.append(f"  {key}: {val1}")
        else:
            lines.append(f"  {key}:")
            lines.append(f"    Current:  {val1}")
            lines.append(f"    New:      {val2}")

    lines.append("=" * 80)

    return "\n".join(lines)


def create_config_from_ui_values(
    emilia_batch_size: int,
    whisper_chunk_size: int,
    transcriber_batch_size: int,
    whisper_model: str,
    compute_type: str,
    uvr_separation: bool,
    emilia_threads: int,
    min_segment_duration: float,
    language: str = "en",
    slice_method: str = "WhisperX Timestamps",
    purge_long_segments: bool = False,
    max_segment_length: int = 12,
) -> Dict[str, any]:
    """
    Create configuration dictionary from UI values

    Returns:
        Configuration dictionary ready for export
    """
    return {
        "emilia_batch_size": emilia_batch_size,
        "whisper_chunk_size": whisper_chunk_size,
        "transcriber_batch_size": transcriber_batch_size,
        "whisper_model": whisper_model,
        "compute_type": compute_type,
        "uvr_separation": uvr_separation,
        "emilia_threads": emilia_threads,
        "min_segment_duration": min_segment_duration,
        "language": language,
        "slice_method": slice_method,
        "purge_long_segments": purge_long_segments,
        "max_segment_length": max_segment_length,
    }


def generate_config_summary(config: Dict[str, any]) -> str:
    """
    Generate human-readable summary of configuration

    Args:
        config: Configuration dictionary

    Returns:
        Formatted summary string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("CONFIGURATION SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    lines.append("🎯 Batch Sizes:")
    lines.append(f"   Emilia Pipeline: {config.get('emilia_batch_size', 'Not set')}")
    lines.append(f"   Whisper Chunk Size: {config.get('whisper_chunk_size', 'Not set')}")
    lines.append(f"   Transcriber: {config.get('transcriber_batch_size', 'Not set')}")
    lines.append("")

    lines.append("⚙️  Model Settings:")
    lines.append(f"   Whisper Model: {config.get('whisper_model', 'Not set')}")
    lines.append(f"   Compute Type: {config.get('compute_type', 'Not set')}")
    lines.append(f"   UVR Separation: {config.get('uvr_separation', 'Not set')}")
    lines.append(f"   Threads: {config.get('emilia_threads', 'Not set')}")
    lines.append("")

    lines.append("🎙️  Audio Processing:")
    lines.append(f"   Language: {config.get('language', 'Not set')}")
    lines.append(f"   Slice Method: {config.get('slice_method', 'Not set')}")
    lines.append(f"   Min Segment Duration: {config.get('min_segment_duration', 'Not set')}s")
    lines.append(f"   Max Segment Length: {config.get('max_segment_length', 'Not set')}s")
    lines.append(f"   Purge Long Segments: {config.get('purge_long_segments', 'Not set')}")
    lines.append("")

    lines.append("=" * 80)

    return "\n".join(lines)


if __name__ == "__main__":
    # Test configuration export
    test_config = create_config_from_ui_values(
        emilia_batch_size=24,
        whisper_chunk_size=24,
        transcriber_batch_size=24,
        whisper_model="large-v3",
        compute_type="float16",
        uvr_separation=True,
        emilia_threads=4,
        min_segment_duration=0.25,
    )

    print(generate_config_summary(test_config))
    print("\nValidating configuration...")
    is_valid, errors = validate_configuration(test_config)
    print(f"Valid: {is_valid}")
    if errors:
        print("Errors:", errors)
