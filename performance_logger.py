"""
Performance Telemetry Logger - Optional detailed logging for troubleshooting
Tracks VRAM usage, processing times, throughput, and system metrics
"""

import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import os


@dataclass
class ProcessingMetrics:
    """Metrics for a single processing operation"""
    operation: str  # "diarization", "transcription", "separation", etc.
    file_name: str
    start_time: float
    end_time: float
    duration_seconds: float
    success: bool
    error_message: Optional[str] = None
    vram_before_mb: Optional[float] = None
    vram_after_mb: Optional[float] = None
    vram_peak_mb: Optional[float] = None
    audio_duration_seconds: Optional[float] = None
    throughput_ratio: Optional[float] = None  # processing_time / audio_duration
    batch_size: Optional[int] = None
    model_name: Optional[str] = None


class PerformanceLogger:
    """
    Performance telemetry logger with optional detailed tracking
    """

    def __init__(self, project_name: str, log_dir: Optional[Path] = None, enabled: bool = True):
        """
        Initialize performance logger

        Args:
            project_name: Name of the project being processed
            log_dir: Directory to store logs (defaults to project_folder/logs/performance)
            enabled: Enable/disable logging
        """
        self.project_name = project_name
        self.enabled = enabled
        self.metrics: List[ProcessingMetrics] = []
        self.session_start = time.time()

        if log_dir is None:
            log_dir = Path.cwd() / "datasets_folder" / project_name / "logs" / "performance"

        self.log_dir = log_dir

        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = self.log_dir / f"performance_{self.session_id}.jsonl"

    def get_vram_usage(self) -> Optional[float]:
        """
        Get current VRAM usage in MB

        Returns:
            VRAM usage in MB or None if not available
        """
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated(0) / (1024 ** 2)
        except:
            pass
        return None

    def get_vram_peak(self) -> Optional[float]:
        """
        Get peak VRAM usage in MB

        Returns:
            Peak VRAM usage in MB or None if not available
        """
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.max_memory_allocated(0) / (1024 ** 2)
        except:
            pass
        return None

    def reset_peak_stats(self):
        """Reset peak memory statistics"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(0)
        except:
            pass

    def start_operation(
        self,
        operation: str,
        file_name: str,
        audio_duration: Optional[float] = None,
        batch_size: Optional[int] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Start tracking an operation

        Args:
            operation: Name of the operation
            file_name: File being processed
            audio_duration: Duration of audio in seconds
            batch_size: Batch size used
            model_name: Model name used

        Returns:
            Context dictionary to pass to end_operation
        """
        if not self.enabled:
            return {}

        self.reset_peak_stats()

        context = {
            "operation": operation,
            "file_name": file_name,
            "start_time": time.time(),
            "vram_before_mb": self.get_vram_usage(),
            "audio_duration": audio_duration,
            "batch_size": batch_size,
            "model_name": model_name,
        }

        return context

    def end_operation(self, context: Dict[str, Any], success: bool = True, error: Optional[str] = None):
        """
        End tracking an operation and log metrics

        Args:
            context: Context returned by start_operation
            success: Whether operation succeeded
            error: Error message if failed
        """
        if not self.enabled or not context:
            return

        end_time = time.time()
        duration = end_time - context["start_time"]

        vram_after = self.get_vram_usage()
        vram_peak = self.get_vram_peak()

        # Calculate throughput ratio if audio duration available
        throughput = None
        if context.get("audio_duration"):
            throughput = duration / context["audio_duration"]

        metrics = ProcessingMetrics(
            operation=context["operation"],
            file_name=context["file_name"],
            start_time=context["start_time"],
            end_time=end_time,
            duration_seconds=duration,
            success=success,
            error_message=error,
            vram_before_mb=context.get("vram_before_mb"),
            vram_after_mb=vram_after,
            vram_peak_mb=vram_peak,
            audio_duration_seconds=context.get("audio_duration"),
            throughput_ratio=throughput,
            batch_size=context.get("batch_size"),
            model_name=context.get("model_name"),
        )

        self.metrics.append(metrics)
        self._write_metric(metrics)

    def _write_metric(self, metric: ProcessingMetrics):
        """Write metric to log file"""
        if not self.enabled:
            return

        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(asdict(metric)) + '\n')
        except Exception as e:
            print(f"Warning: Failed to write performance metric: {e}")

    def generate_summary(self) -> str:
        """
        Generate summary report of all metrics

        Returns:
            Formatted summary string
        """
        if not self.metrics:
            return "No metrics recorded"

        lines = []
        lines.append("=" * 80)
        lines.append("PERFORMANCE SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Project: {self.project_name}")
        lines.append(f"Session: {self.session_id}")
        lines.append(f"Total Operations: {len(self.metrics)}")
        lines.append("")

        # Group by operation type
        by_operation: Dict[str, List[ProcessingMetrics]] = {}
        for metric in self.metrics:
            if metric.operation not in by_operation:
                by_operation[metric.operation] = []
            by_operation[metric.operation].append(metric)

        # Statistics per operation
        for operation, ops in by_operation.items():
            lines.append(f"📊 {operation.upper()}:")
            lines.append(f"   Operations: {len(ops)}")

            successful = [m for m in ops if m.success]
            failed = [m for m in ops if not m.success]

            lines.append(f"   Success: {len(successful)} | Failed: {len(failed)}")

            if successful:
                durations = [m.duration_seconds for m in successful]
                avg_duration = sum(durations) / len(durations)
                min_duration = min(durations)
                max_duration = max(durations)

                lines.append(f"   Duration: Avg={avg_duration:.2f}s, Min={min_duration:.2f}s, Max={max_duration:.2f}s")

                # Throughput if available
                throughputs = [m.throughput_ratio for m in successful if m.throughput_ratio]
                if throughputs:
                    avg_throughput = sum(throughputs) / len(throughputs)
                    lines.append(f"   Throughput Ratio: {avg_throughput:.2f}x (lower is better)")

                # VRAM usage
                vram_peaks = [m.vram_peak_mb for m in successful if m.vram_peak_mb]
                if vram_peaks:
                    avg_vram = sum(vram_peaks) / len(vram_peaks)
                    max_vram = max(vram_peaks)
                    lines.append(f"   VRAM: Avg={avg_vram:.0f}MB, Peak={max_vram:.0f}MB")

            if failed:
                lines.append(f"   ⚠️  Failed operations:")
                for fail in failed[:5]:  # Show first 5 failures
                    lines.append(f"      {fail.file_name}: {fail.error_message}")

            lines.append("")

        # Overall statistics
        session_duration = time.time() - self.session_start
        lines.append(f"⏱️  Total Session Time: {session_duration / 60:.1f} minutes")

        # Audio processing stats
        total_audio = sum(m.audio_duration_seconds for m in self.metrics if m.audio_duration_seconds)
        if total_audio > 0:
            lines.append(f"🎵 Total Audio Processed: {total_audio / 60:.1f} minutes")
            overall_throughput = session_duration / total_audio
            lines.append(f"📈 Overall Throughput: {overall_throughput:.2f}x realtime")

        lines.append("\n" + "=" * 80)
        lines.append(f"Full log saved to: {self.log_file}")
        lines.append("=" * 80)

        return "\n".join(lines)

    def save_summary(self, output_path: Optional[Path] = None):
        """
        Save summary report to file

        Args:
            output_path: Path to save summary (defaults to logs directory)
        """
        if not self.enabled:
            return

        if output_path is None:
            output_path = self.log_dir / f"summary_{self.session_id}.txt"

        summary = self.generate_summary()

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary)
        except Exception as e:
            print(f"Warning: Failed to save summary: {e}")

    def export_csv(self, output_path: Optional[Path] = None):
        """
        Export metrics to CSV for analysis

        Args:
            output_path: Path to save CSV (defaults to logs directory)
        """
        if not self.enabled or not self.metrics:
            return

        if output_path is None:
            output_path = self.log_dir / f"metrics_{self.session_id}.csv"

        try:
            import csv

            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=asdict(self.metrics[0]).keys())
                writer.writeheader()
                for metric in self.metrics:
                    writer.writerow(asdict(metric))

        except Exception as e:
            print(f"Warning: Failed to export CSV: {e}")


# Singleton instance for easy access
_global_logger: Optional[PerformanceLogger] = None


def init_global_logger(project_name: str, enabled: bool = True) -> PerformanceLogger:
    """Initialize global performance logger"""
    global _global_logger
    _global_logger = PerformanceLogger(project_name, enabled=enabled)
    return _global_logger


def get_global_logger() -> Optional[PerformanceLogger]:
    """Get global performance logger"""
    return _global_logger


def close_global_logger():
    """Close and finalize global logger"""
    global _global_logger
    if _global_logger and _global_logger.enabled:
        _global_logger.save_summary()
        _global_logger.export_csv()
    _global_logger = None


if __name__ == "__main__":
    # Test logger
    logger = PerformanceLogger("test_project", enabled=True)

    # Simulate operations
    ctx = logger.start_operation("transcription", "test_audio.wav", audio_duration=120.0, batch_size=16)
    time.sleep(0.5)  # Simulate processing
    logger.end_operation(ctx, success=True)

    ctx = logger.start_operation("diarization", "test_audio.wav", audio_duration=120.0)
    time.sleep(0.3)
    logger.end_operation(ctx, success=True)

    print(logger.generate_summary())
