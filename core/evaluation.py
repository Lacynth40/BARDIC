class ArtEvaluator:
    def evaluate(self, draft, context, raw_score):
        """
        Refines the ArtCritic's raw output into a diagnostic report for the Logger.
        """
        # 1. Internal Diagnostics
        # Note: Using 'draft' to match the argument name
        results = {
            "has_title": "<TITLE>" in draft,
            "line_count": draft.count("<LINE>"),
            "is_boolean": draft.strip() in ["True", "False", "<TITLE> True", "<TITLE> False"],
            "context_hallucination": False # Placeholder for future RAG checks
        }
        
        # 2. Status Determination
        # We use the raw_score from the Critic, but we can also use our internal results
        status = "NOMINAL" if raw_score >= 0.8 else "UNSTABLE"
        
        # 3. The Single Return
        # This packages everything for the Logger in main.py
        return {
            "final_score": round(raw_score, 2),
            "status": status,
            "diagnostics": results,
            "length_chars": len(draft)
        }