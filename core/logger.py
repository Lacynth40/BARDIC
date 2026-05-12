import json
import os
from datetime import datetime

class AgentLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"full_traces_{self.timestamp}.jsonl")

    # FIX 1: Changed 'topic' to 'query' to match the main.py keyword arguments
    def log_mission(self, query: str, draft: str, score: float, status: str, awen: dict | None = None):
        
        # Build the core dictionary
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "draft": draft,
            "score": score,
            "status": status
        }
        
        # FIX 2: Merge the Awen payload directly into the JSON object
        # This keeps the output as exactly one valid JSON object per line
        if awen:
            entry["awen_payload"] = awen
            
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")