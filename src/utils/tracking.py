"""Simple experiment tracking utilities using JSON logs."""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

class ExperimentTracker:
    """Simple JSON-based experiment tracking."""
    
    def __init__(self, experiment_name: str, log_dir: str = "data/logs"):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{self.run_id}.jsonl"
        self.config: Dict[str, Any] = {}
        self.metrics: Dict[str, list] = {}
        
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log experiment configuration."""
        self.config = config
        self._write_entry({"type": "config", "data": config})
        
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        entry = {"name": name, "value": value}
        if step is not None:
            entry["step"] = step
        self.metrics[name].append(value)
        self._write_entry({"type": "metric", "data": entry})
        
    def log_episode(self, episode: int, metrics: Dict[str, float]) -> None:
        """Log metrics for an episode."""
        self._write_entry({
            "type": "episode",
            "episode": episode,
            "data": metrics
        })
        
    def _write_entry(self, entry: Dict[str, Any]) -> None:
        """Write an entry to the JSONL log file."""
        entry["timestamp"] = datetime.now().isoformat()
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
            
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of logged metrics."""
        summary = {"experiment": self.experiment_name, "run_id": self.run_id}
        for name, values in self.metrics.items():
            if values:
                summary[f"{name}_mean"] = sum(values) / len(values)
                summary[f"{name}_min"] = min(values)
                summary[f"{name}_max"] = max(values)
        return summary


def load_experiment_logs(log_file: str) -> list:
    """Load all entries from a JSONL log file."""
    entries = []
    with open(log_file) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries
