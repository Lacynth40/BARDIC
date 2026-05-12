import os
import re
import time
import torch
import logging
import wikipedia
import nltk

from dotenv import load_dotenv
from collections import Counter
from sentence_transformers import SentenceTransformer, util
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize


# ---------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# LOGGER
# ---------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# NLTK SETUP
# ---------------------------------------------------------

def ensure_nltk():
    resources = [
        "punkt",
        "punkt_tab",
        "wordnet",
        "stopwords",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
    ]

    for resource in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource, quiet=True)


ensure_nltk()


class RAGEncoder:
    def __init__(self, config_path: str = "config/settings.yaml"):
        logger.info("Waking the Sentence Transformer...")

        self.memory_cache = {}

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        hf_token = os.getenv("HF_TOKEN")

        self.encoder = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=self.device,
            token=hf_token,
        )

        wikipedia.set_lang("en")
        wikipedia.set_user_agent(
            "EmergitBot/1.0 (Academic Research Project)"
        )

        self.last_wiki_request_time = 0.0
        self.min_wiki_interval_seconds = 2.0

        self.lemmatizer = WordNetLemmatizer()

        self.stop_words = set(stopwords.words("english"))
        self.stop_words.update(
            {
                "wikipedia",
                "article",
                "references",
                "external",
                "links",
                "history",
                "first",
                "second",
                "years",
                "known",
                "early",
                "often",
                "also",
                "however",
            }
        )

    # ---------------------------------------------------------
    # WIKIPEDIA RATE LIMITING
    # ---------------------------------------------------------

    def _wiki_wait(self):
        elapsed = time.time() - self.last_wiki_request_time

        if elapsed < self.min_wiki_interval_seconds:
            time.sleep(self.min_wiki_interval_seconds - elapsed)

        self.last_wiki_request_time = time.time()

    # ---------------------------------------------------------
    # CHUNK FILTERING
    # ---------------------------------------------------------

    def is_bad_chunk(self, chunk: str) -> bool:
        bad_terms = [
            "list of",
            "see also",
            "references",
            "external links",
            "bibliography",
            "further reading",
            "navbox",
        ]

        chunk_lower = chunk.lower()

        if any(term in chunk_lower for term in bad_terms):
            return True

        lines = [
            line.strip()
            for line in chunk.splitlines()
            if line.strip()
        ]

        if len(lines) >= 5:
            short_lines = sum(
                1 for line in lines
                if len(line.split()) <= 8
            )

            if short_lines / len(lines) > 0.6:
                return True

        return False

    # ---------------------------------------------------------
    # MAIN FETCH
    # ---------------------------------------------------------

    def fetch_awen(self, query: str) -> dict:
        query = query.strip()

        if not query:
            raise ValueError("[AWEN] Empty query received.")

        # ---------------------------------------------------------
        # MEMORY CACHE
        # ---------------------------------------------------------

        if query in self.memory_cache:
            return self.memory_cache[query]

        # ---------------------------------------------------------
        # WIKIPEDIA FETCH
        # ---------------------------------------------------------

        try:
            self._wiki_wait()
            search_results = wikipedia.search(query)

            if not search_results:
                raise ValueError("No Wikipedia results found.")

            top_page = search_results[0]

            try:
                self._wiki_wait()
                page = wikipedia.page(
                    top_page,
                    auto_suggest=False,
                )

            except wikipedia.exceptions.DisambiguationError as e:
                self._wiki_wait()
                page = wikipedia.page(
                    e.options[0],
                    auto_suggest=False,
                )

            content_raw = page.content

            print("SEARCH RESULTS:", search_results)
            print("SELECTED PAGE:", page.title)
            print("PAGE URL:", page.url)
            print("CONTENT PREVIEW:")
            print(content_raw[:1000])

            content_raw = re.sub(
                r"==+.*?==+",
                "",
                content_raw,
            )

            content_raw = re.sub(
                r"\n{3,}",
                "\n\n",
                content_raw,
            )

        except Exception as e:
            logger.warning(
                f"Wikipedia extraction failed for {query}: {e}"
            )

            fallback_summary = (
                f"{query.capitalize()} is a subject with limited "
                f"available context in the archives."
            )

            query_emb = self.encoder.encode(
                query,
                convert_to_tensor=True,
                device=self.device,
            )

            context_text = (
                fallback_summary +
                "\nKeywords: " +
                query.capitalize()
            )

            context_emb = self.encoder.encode(
                context_text,
                convert_to_tensor=True,
                device=self.device,
            )

            return {
                "summary": fallback_summary,
                "line_titles": [
                    query.capitalize(),
                    "History",
                    "Nature",
                    "Form",
                ],
                "query_embedding": query_emb.cpu().tolist(),
                "context_embedding": context_emb.cpu().tolist(),
            }

        # ---------------------------------------------------------
        # GLOBAL ARTICLE SAMPLING
        # ---------------------------------------------------------

        words = content_raw.split()
        total_words = len(words)

        if total_words > 2400:
            part_size = 800
            mid_start = (total_words // 2) - (part_size // 2)

            global_text = " ".join(
                words[:part_size]
                + words[mid_start: mid_start + part_size]
                + words[-part_size:]
            )
        else:
            global_text = content_raw

        # ---------------------------------------------------------
        # ARTICLE-WIDE NOUN CANDIDATE EXTRACTION
        # ---------------------------------------------------------

        tokens = nltk.word_tokenize(global_text)
        tagged = nltk.pos_tag(tokens)

        lemma_counts = Counter()
        lemma_to_surface = {}

        for word, pos in tagged:
            word_lower = word.lower()

            if not pos.startswith("NN"):
                continue

            if word_lower in self.stop_words:
                continue

            if not word.isalpha():
                continue

            if len(word_lower) <= 2:
                continue

            lemma = self.lemmatizer.lemmatize(word_lower)

            lemma_counts[lemma] += 1

            if lemma not in lemma_to_surface:
                lemma_to_surface[lemma] = word_lower

        # ---------------------------------------------------------
        # BUILD CANDIDATE POOL ONLY
        # Do not choose final keywords yet.
        # ---------------------------------------------------------

        candidate_pool = []

        query_emb = self.encoder.encode(
            query,
            convert_to_tensor=True,
            device=self.device,
        )

        if lemma_counts:
            top_lemmas = [
                lemma
                for lemma, count in lemma_counts.most_common(80)
            ]

            top_candidates = [
                lemma_to_surface[lemma]
                for lemma in top_lemmas
            ]

            max_freq = lemma_counts[top_lemmas[0]]

            cand_embs = self.encoder.encode(
                top_candidates,
                convert_to_tensor=True,
                device=self.device,
            )

            query_terms = {
                self.lemmatizer.lemmatize(w.lower())
                for w in re.findall(r"[a-zA-Z]+", query)
                if len(w) > 2
            }

            seen_lemmas = set()

            for i, surface_word in enumerate(top_candidates):
                word_lower = surface_word.lower()
                lemma = self.lemmatizer.lemmatize(word_lower)

                # Do not let the query itself become a keyword.
                if lemma in query_terms:
                    continue

                # Prevent singular/plural duplicate concepts.
                if lemma in seen_lemmas:
                    continue

                seen_lemmas.add(lemma)

                freq_score = lemma_counts[lemma] / max_freq

                candidate_pool.append(
                    {
                        "word": surface_word,
                        "lemma": lemma,
                        "freq_score": freq_score,
                        "embedding": cand_embs[i],
                    }
                )

        # ---------------------------------------------------------
        # SUMMARY SELECTION
        # Prefer the Wikipedia lead paragraph.
        # Fall back to semantic chunk search only if needed.
        # ---------------------------------------------------------

        paragraphs = [
            para.strip()
            for para in content_raw.split("\n\n")
            if para.strip()
        ]

        lead_summary = ""

        for para in paragraphs:
            if len(para.split()) >= 25 and not self.is_bad_chunk(para):
                sentences = sent_tokenize(para)

                temp_summary = ""
                word_count = 0

                for sentence in sentences:
                    temp_summary += sentence + " "
                    word_count += len(sentence.split())

                    if word_count >= 120:
                        break

                lead_summary = temp_summary.strip()
                break

        if lead_summary:
            best_summary = lead_summary
            best_score = 1.0

        else:
            chunks = [
                chunk.strip()
                for chunk in content_raw.split("\n\n")
                if len(chunk.split()) > 20
                and not self.is_bad_chunk(chunk)
            ]

            if not chunks:
                chunks = [content_raw[:500]]

            chunk_embeddings = self.encoder.encode(
                chunks,
                convert_to_tensor=True,
                device=self.device,
            )

            summary_query = f"{page.title} overview"

            summary_emb = self.encoder.encode(
                summary_query,
                convert_to_tensor=True,
                device=self.device,
            )

            semantic_results = util.semantic_search(
                summary_emb,
                chunk_embeddings,
                top_k=3,
            )[0]

            best_summary = ""
            best_score = float("-inf")

            for result in semantic_results:
                chunk = chunks[result["corpus_id"]]
                sentences = sent_tokenize(chunk)

                temp_summary = ""
                word_count = 0

                for sentence in sentences:
                    temp_summary += sentence + " "
                    word_count += len(sentence.split())

                    if word_count >= 120:
                        break

                temp_summary = temp_summary.strip()

                score = result["score"]

                if score > best_score:
                    best_score = score
                    best_summary = temp_summary

            if not best_summary:
                best_summary = content_raw[:500].strip()

        # ---------------------------------------------------------
        # ARTICLE-CENTERED KEYWORD SELECTION
        # Keywords are ranked against article content,
        # not query and not summary.
        # ---------------------------------------------------------

        article_keyword_emb = self.encoder.encode(
            global_text,
            convert_to_tensor=True,
            device=self.device,
        )

        scored_candidates = []

        for item in candidate_pool:
            sim_score = util.cos_sim(
                item["embedding"],
                article_keyword_emb,
            ).item()

            combined_score = (
                sim_score * 0.65
                + item["freq_score"] * 0.35
            )

            scored_candidates.append(
                (
                    combined_score,
                    item["word"],
                    item["lemma"],
                    item["embedding"],
                )
            )

        scored_candidates.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        candidates = []
        thresholds = [0.75, 0.85, 0.95]

        for threshold in thresholds:
            candidates = []
            selected_embs = []
            seen_lemmas = set()

            for score, word, lemma, emb in scored_candidates:
                if lemma in seen_lemmas:
                    continue

                is_diverse = True

                for previous_emb in selected_embs:
                    if util.cos_sim(emb, previous_emb).item() > threshold:
                        is_diverse = False
                        break

                if not is_diverse:
                    continue

                candidates.append(word.capitalize())
                selected_embs.append(emb)
                seen_lemmas.add(lemma)

                if len(candidates) == 4:
                    break

            if len(candidates) == 4:
                break

        # ---------------------------------------------------------
        # KEYWORD FALLBACK
        # If article-centered selection finds fewer than 4.
        # ---------------------------------------------------------

        if len(candidates) < 4:
            logger.warning(
                f"[AWEN] Only {len(candidates)} keyword candidates found. "
                f"Expanding fallback lexicon."
            )

            seen = {
                self.lemmatizer.lemmatize(word.lower())
                for word in candidates
            }

            for lemma, count in lemma_counts.most_common(80):
                if lemma in seen:
                    continue

                if lemma not in lemma_to_surface:
                    continue

                word = lemma_to_surface[lemma].capitalize()

                candidates.append(word)
                seen.add(lemma)

                if len(candidates) == 4:
                    break

        global_keywords = tuple(candidates[:4])

        logger.info(f"AWEN GENERATED | Query: {query.upper()}")
        logger.info(f"THEMES: {global_keywords}")
        logger.info(f"SUMMARY SCORE: {best_score}")

        # ---------------------------------------------------------
        # SUMMARY ↔ KEYWORD CONTRACT CHECK
        # Do not alter the summary.
        # The poet/critic will handle keyword usage downstream.
        # ---------------------------------------------------------

        valid_keywords = [
            keyword.capitalize()
            for keyword in global_keywords
        ]

        # ---------------------------------------------------------
        # FINAL KEYWORD NORMALIZATION GATE
        # ---------------------------------------------------------

        deduped_keywords = []
        seen_keyword_lemmas = set()

        for keyword in valid_keywords:
            keyword_clean = keyword.strip()

            if not keyword_clean:
                continue

            keyword_lower = keyword_clean.lower()
            keyword_lemma = self.lemmatizer.lemmatize(keyword_lower)

            if keyword_lemma in seen_keyword_lemmas:
                continue

            seen_keyword_lemmas.add(keyword_lemma)
            deduped_keywords.append(keyword_clean.capitalize())

        valid_keywords = deduped_keywords[:4]

        if len(valid_keywords) != 4:
            raise ValueError(
                f"[AWEN CONTRACT FAIL] Only "
                f"{len(valid_keywords)} unique keywords could be "
                f"enforced into summary."
            )

        # ---------------------------------------------------------
        # EMBED SUMMARY + KEYWORDS AS CONTEXT
        # ---------------------------------------------------------

        context_text = (
            best_summary
            + "\nKeywords: "
            + ", ".join(valid_keywords)
        )

        context_emb = self.encoder.encode(
            context_text,
            convert_to_tensor=True,
            device=self.device,
        )

        payload = {
            "summary": best_summary,
            "line_titles": valid_keywords,
            "query_embedding": query_emb.cpu().tolist(),
            "context_embedding": context_emb.cpu().tolist(),
        }

        self.memory_cache[query] = payload

        logger.info(f"FINAL KEYWORDS: {valid_keywords}")
        logger.info(
            f"KEYWORD DISTRIBUTION IN SUMMARY: "
            f"{[kw for kw in valid_keywords if kw.lower() in best_summary.lower()]}"
        )

        return payload