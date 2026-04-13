# main.py (Root Level)
from datetime import datetime
from core.rag_encoder import RAGEncoder
from core.t5_poet import T5Poet 
from core.art_critic import ArtCritic
from core.evaluation import ArtEvaluator
from core.logger import BardicLogger

def start_mission():
    # Setup
    config = "config/settings.yaml"
    print("--- [BARDIC] Initializing Systems ---")
    
    librarian = RAGEncoder(config)
    poet = T5Poet(config)
    critic = ArtCritic(config)
    logger = BardicLogger()
    evaluator = ArtEvaluator()

    # 1. Sensing (Wikipedia)
    query = input("\nEnter Topic: ")
    context = librarian.retrieve_context(query)
    print(f"\n[Librarian] Context Retrieved ({len(context)} chars).")

    # 2. Acting (T5)
    print("[Poet] Drafting quatrain...")
    draft = poet.draft_poem(context)
    print(f"\n--- DRAFT ---\n{draft}\n-------------")

    # 3. Critiquing (NLTK/CMU)
    print("[Critic] Evaluating structural integrity...")
    score = critic.score_draft(draft)
    
    print(f"\nFinal Architecture Score: {score}")
    if score < 0.8:
        print("Status: UNSTABLE. Structural constraints not met.")
    else:
        print("Status: NOMINAL. Agentic consistency achieved.")

    # 4. Auditing (Evaluator)
    report = evaluator.evaluate(draft, context, score)

    # 4. Recording (Logger)
    trace = {
        "timestamp": datetime.now().isoformat(),
        "topic": query,
        "inputs": {"context": context},
        "outputs": {"draft": draft},
        "evaluation": report,
        "poet_metadata": {"tokens_used": len(draft.split())} # Rough estimate
    }
    logger.record(trace)

    return report


if __name__ == "__main__":
    start_mission()