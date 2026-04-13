import json
import csv
import os
from datetime import datetime

class BardicLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.trace_file = os.path.join(self.log_dir, "full_traces.jsonl")
        self.metrics_file = os.path.join(self.log_dir, "summary_metrics.csv")
        self._init_metrics()

    def _init_metrics(self):
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "topic", "score", "status", "tokens_in"])

    def record(self, trace):
        # The Deep Dive (JSONL)
        with open(self.trace_file, 'a') as f:
            f.write(json.dumps(trace) + "\n")
            
        # The Metrics Dashboard (CSV)
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                trace["timestamp"],
                trace["topic"],
                trace["evaluation"]["final_score"],
                trace["evaluation"]["status"],
                trace["poet_metadata"]["tokens_used"]
            ])