import json
import sys
import yaml
import subprocess
from pathlib import Path

from core.t5_poet import T5Poet
from core.rag_encoder import RAGEncoder
from core.art_critic import ArtCritic
from core.spin_doctor import SpinDoctor

ROOT_DIR = Path(__file__).resolve().parent

CONFIG_PATH = ROOT_DIR / "config" / "settings.yaml"
LOG_PATH = ROOT_DIR / "logs" / "critic_runs.jsonl"
RL_TRAIN_PATH = ROOT_DIR / "core" / "rl_train.py"


def load_config():
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    except FileNotFoundError:
        print("[Main] Missing config/settings.yaml")
        return {}


def log_critic_run(entry, log_path=LOG_PATH):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def count_logged_runs(log_path=LOG_PATH):
    if not log_path.exists():
        return 0

    with log_path.open("r", encoding="utf-8") as f:
        return sum(
            1 for line in f
            if line.strip()
        )


def maybe_run_rl_training(log_path=LOG_PATH, threshold=5):
    logged_count = count_logged_runs(log_path)

    print(f"\n[RL] Logged critic runs: {logged_count}/{threshold}")

    if logged_count < threshold:
        return

    if not RL_TRAIN_PATH.exists():
        print("[RL] Threshold reached, but core/rl_train.py does not exist yet.")
        return

    print("[RL] Threshold reached. Starting rl_train.py...")

    result = subprocess.run(
        [sys.executable, str(RL_TRAIN_PATH)],
        cwd=str(ROOT_DIR),
        text=True,
    )

    if result.returncode == 0:
        print("[RL] rl_train.py completed successfully.")
    else:
        print(f"[RL] rl_train.py failed with return code {result.returncode}.")


def main():
    print("\n=== BARDIC STRUCTURED POET ===\n")

    config = load_config()

    rag = RAGEncoder()
    poet = T5Poet(config)
    spin_doctor = SpinDoctor()
    critic = ArtCritic()

    query = input("Enter topic/query: ").strip()

    if not query:
        print("[Main] Empty query. Exiting.")
        return

    print("\n[RAG] Fetching context...\n")

    payload = rag.fetch_awen(query)

    summary = payload["summary"]
    keywords = payload["line_titles"]

    query_embedding = payload.get("query_embedding")
    context_embedding = payload.get("context_embedding")

    print("[Summary]")
    print(summary.strip())

    print("\n[Keywords]")
    print(keywords)

    print("\n[Embedding Check]")
    print(
        "Query embedding length:   "
        f"{len(query_embedding) if query_embedding else 'missing'}"
    )
    print(
        "Context embedding length: "
        f"{len(context_embedding) if context_embedding else 'missing'}"
    )

    print("\n[Generating]\n")

    draft_poem = poet.generate_poem(
        summary=summary,
        keywords=keywords,
    )

    print("[Draft]")
    print(draft_poem)

    print("\n[Spin Doctor]\n")

    poem = spin_doctor.repair_poem(draft_poem)

    print("[Final]")
    print(poem)

    print("\n[Critic]\n")

    critic_result = critic.weigh_heart(
        draft=poem,
        query_embedding=query_embedding,
        context_embedding=context_embedding,
    )

    print(f"Scalar score: {critic_result['score']}")
    print(f"Score / 10:    {critic_result['score_10']}")
    print(f"Status:        {critic_result['status']}")

    print("\nSubscores / 10:")
    for key, value in critic_result["subscores_10"].items():
        print(f"  {key}: {value}")

    print("\nFeedback:")
    for item in critic_result["feedback"]:
        print(f"  - {item}")

    log_entry = {
        "query": query,

        # Human-readable RAG output.
        # Logged for debugging and future analysis.
        "summary": summary,
        "keywords": keywords,

        # Generated output.
        "draft_poem": draft_poem,
        "poem": poem,

        # Critic output.
        "critic_score": critic_result["score"],
        "critic_score_10": critic_result["score_10"],
        "critic_status": critic_result["status"],
        "critic_subscores_10": critic_result["subscores_10"],
        "critic_feedback": critic_result["feedback"],

        # RAG vector output.
        # Used by the vector-grounded critic and future RL.
        "query_embedding": (
            query_embedding.tolist()
            if hasattr(query_embedding, "tolist")
            else query_embedding
        ),
        "context_embedding": (
            context_embedding.tolist()
            if hasattr(context_embedding, "tolist")
            else context_embedding
        ),
    }

    log_critic_run(log_entry)

    print(f"\n[Log] Saved critic run to {LOG_PATH}")

    maybe_run_rl_training()

    print("\n=== DONE ===\n")


if __name__ == "__main__":
    main()