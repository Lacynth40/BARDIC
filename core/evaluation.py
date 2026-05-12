class ArtEvaluator:
    def __init__(self):
        pass

    def evaluate(self, draft: str, awen: dict, score: float):
        """
        Performs the final audit before Ceridwen archives the results.
        """
        # Detection for "Boolean Panic" (returning True/False instead of text)
        is_hallucinated = False
        if draft.strip().lower() in ["true", "false", "none", "no context found"]:
            is_hallucinated = True
            score = 0.0
            
        # Check if the Poet actually used the sacred essence (awen)
        # We look for the 4 line titles in the poem
        found_titles = [t for t in awen['line_titles'] if t.lower() in draft.lower()]
        
        report = {
            "score": score,
            "status": "NOMINAL" if score >= 0.75 else "UNSTABLE",
            "internal_consistency": {
                "titles_found": found_titles,
                "title_count": len(found_titles),
                "context_summary_used": any(word in draft.lower() for word in awen['summary'].split()[:5])
            },
            "boolean_panic_detected": is_hallucinated
        }
        
        return report