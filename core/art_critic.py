import re
import torch
import nltk

from nltk.corpus import cmudict
from sentence_transformers import SentenceTransformer, util


class ArtCritic:
    def __init__(self):
        print("[Anubis] Awakening transformer-backed poetic critic...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.evaluator = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=self.device,
        )

        try:
            self.cmu = cmudict.dict()
        except LookupError:
            nltk.download("cmudict", quiet=True)
            self.cmu = cmudict.dict()

        self.bad_output_phrases = [
            "output only",
            "include the word",
            "requirements",
            "no explanations",
            "context summary",
            "semantic anchors",
            "keywords:",
            "themes:",
            "write four poetic lines",
            "separate lines with",
            "must follow",
        ]

        self.poetic_lexicon = {
            "shadow", "river", "stone", "ash", "wind", "memory", "moon",
            "silence", "fire", "mist", "crown", "oath", "throne", "dream",
            "blood", "rain", "forest", "sea", "night", "song", "law",
            "king", "queen", "land", "name", "sorrow", "bone", "star",
            "hill", "storm", "gate", "flame", "dust", "grave", "echo",
            "wolf", "sword", "sky", "breeze", "muse",
        }

    # ---------------------------------------------------------
    # BASIC HELPERS
    # ---------------------------------------------------------

    def _clip_10(self, value: float) -> float:
        return round(max(0.0, min(10.0, value)), 3)

    def _extract_words(self, text: str) -> list:
        return re.findall(r"\b[a-zA-Z']+\b", text.lower())

    def _embed(self, text: str):
        return self.evaluator.encode(
            text,
            convert_to_tensor=True,
            device=self.device,
        )

    def _tensor_from_embedding(self, embedding):
        if embedding is None:
            return None

        if isinstance(embedding, torch.Tensor):
            tensor = embedding
        else:
            tensor = torch.tensor(
                embedding,
                dtype=torch.float32,
            )

        return tensor.to(self.device)

    def _cosine_embedding_to_text(self, embedding, text: str) -> float:
        emb_tensor = self._tensor_from_embedding(embedding)

        if emb_tensor is None:
            raise ValueError("Missing embedding.")

        text_emb = self._embed(text)

        return util.cos_sim(text_emb, emb_tensor).item()

    def _similarity_to_10(
        self,
        similarity: float,
        excellent: float,
        good: float,
        fair: float,
        weak: float,
        poor: float,
    ) -> float:
        if similarity >= excellent:
            return 10.0
        if similarity >= good:
            return 8.0
        if similarity >= fair:
            return 6.0
        if similarity >= weak:
            return 4.0
        if similarity >= poor:
            return 2.0
        return 0.0

    # ---------------------------------------------------------
    # SYLLABLE / RHYME
    # ---------------------------------------------------------

    def _count_syllables(self, word: str) -> int:
        w = word.lower().strip('.,!?;:"()[]{}')
        if not w:
            return 0

        if w in self.cmu:
            return len([
                p for p in self.cmu[w][0]
                if p[-1].isdigit()
            ])

        count = 0
        vowels = "aeiouy"

        if w and w[0] in vowels:
            count += 1

        for index in range(1, len(w)):
            if w[index] in vowels and w[index - 1] not in vowels:
                count += 1

        if w.endswith("e") and count > 1:
            count -= 1

        return count if count > 0 else 1

    def _line_syllable_count(self, line: str) -> int:
        words = re.findall(r"\b[a-zA-Z']+\b", line)
        return sum(self._count_syllables(word) for word in words)

    def _check_rhyme(self, word1: str, word2: str) -> bool:
        w1 = word1.lower().strip('.,!?;:"()[]{}')
        w2 = word2.lower().strip('.,!?;:"()[]{}')

        if not w1 or not w2:
            return False

        if w1 in self.cmu and w2 in self.cmu:
            pron1 = self.cmu[w1][0]
            pron2 = self.cmu[w2][0]

            def get_rhyme_part(pron):
                for i in range(len(pron) - 1, -1, -1):
                    if pron[i].endswith("1"):
                        return pron[i:]

                for i in range(len(pron) - 1, -1, -1):
                    if pron[i][-1].isdigit():
                        return pron[i:]

                return pron

            return get_rhyme_part(pron1) == get_rhyme_part(pron2)

        if len(w1) >= 3 and len(w2) >= 3:
            return w1[-3:] == w2[-3:]

        return False

    def _last_word(self, line: str) -> str:
        clean_line = re.sub(
            r"\[FALLBACK\]",
            "",
            line,
            flags=re.IGNORECASE,
        )

        words = re.findall(r"\b[a-zA-Z']+\b", clean_line)

        return words[-1] if words else ""

    # ---------------------------------------------------------
    # SUBSCORES
    # ---------------------------------------------------------

    def _score_structure_10(
        self,
        lines: list,
        feedback: list,
    ) -> float:
        if len(lines) == 4:
            return 10.0

        feedback.append(
            f"Structure failed: expected 4 lines, got {len(lines)}."
        )
        return 0.0

    def _score_instruction_cleanliness_10(
        self,
        draft: str,
        feedback: list,
    ) -> float:
        lower = draft.lower()

        if "[fallback]" in lower:
            feedback.append(
                "Fallback marker detected: generated poem used fallback templates."
            )
            return 0.0

        for phrase in self.bad_output_phrases:
            if phrase in lower:
                feedback.append(
                    f"Instruction echo detected: {phrase}."
                )
                return 0.0

        if "<pad>" in lower or "</s>" in lower:
            feedback.append(
                "Instruction cleanliness failed: special tokens detected."
            )
            return 0.0

        return 10.0

    def _score_cadence_10(
        self,
        lines: list,
        feedback: list,
    ) -> float:
        if not lines:
            feedback.append("Cadence failed: no lines available.")
            return 0.0

        total = 0.0

        for i, line in enumerate(lines):
            syllables = self._line_syllable_count(line)

            if 10 <= syllables <= 12:
                total += 2.5
            else:
                feedback.append(
                    f"Cadence issue: line {i + 1} has {syllables} syllables; expected 10-12."
                )

        return self._clip_10(total)

    def _score_context_vector_alignment_10(
        self,
        draft: str,
        context_embedding,
        feedback: list,
    ) -> float:
        if context_embedding is None:
            feedback.append(
                "Context vector alignment failed: no context_embedding provided."
            )
            return 0.0

        try:
            similarity = self._cosine_embedding_to_text(
                context_embedding,
                draft,
            )
        except Exception as error:
            feedback.append(
                f"Context vector alignment error: {error}"
            )
            return 0.0

        score = self._similarity_to_10(
            similarity=similarity,
            excellent=0.55,
            good=0.45,
            fair=0.35,
            weak=0.25,
            poor=0.15,
        )

        if score == 0.0:
            feedback.append(
                f"Context vector alignment failed: similarity={similarity:.3f}."
            )
        elif score <= 4.0:
            feedback.append(
                f"Context vector alignment weak: similarity={similarity:.3f}."
            )

        return score

    def _score_query_vector_alignment_10(
        self,
        draft: str,
        query_embedding,
        feedback: list,
    ) -> float:
        if query_embedding is None:
            feedback.append(
                "Query vector alignment failed: no query_embedding provided."
            )
            return 0.0

        try:
            similarity = self._cosine_embedding_to_text(
                query_embedding,
                draft,
            )
        except Exception as error:
            feedback.append(
                f"Query vector alignment error: {error}"
            )
            return 0.0

        score = self._similarity_to_10(
            similarity=similarity,
            excellent=0.50,
            good=0.40,
            fair=0.30,
            weak=0.20,
            poor=0.10,
        )

        if score == 0.0:
            feedback.append(
                f"Query vector alignment failed: similarity={similarity:.3f}."
            )
        elif score <= 4.0:
            feedback.append(
                f"Query vector alignment weak: similarity={similarity:.3f}."
            )

        return score

    def _score_list_fragment_control_10(
        self,
        lines: list,
        feedback: list,
    ) -> float:
        if not lines:
            feedback.append(
                "List fragment control failed: no lines available."
            )
            return 0.0

        score = 10.0

        for index, line in enumerate(lines):
            words = self._extract_words(line)

            if len(words) <= 3:
                score -= 3.0
                feedback.append(
                    f"List fragment issue: line {index + 1} is too short or fragmentary."
                )

            if line.count(",") >= 2 and len(words) <= 8:
                score -= 4.0
                feedback.append(
                    f"List fragment issue: line {index + 1} looks like a comma-list."
                )

            if words:
                most_common_count = max(
                    words.count(word)
                    for word in set(words)
                )

                if most_common_count >= 3:
                    score -= 4.0
                    feedback.append(
                        f"List fragment issue: line {index + 1} repeats a word too heavily."
                    )

        return self._clip_10(score)

    def _score_repetition_control_10(
        self,
        lines: list,
        feedback: list,
    ) -> float:
        all_words = []

        for line in lines:
            all_words.extend(self._extract_words(line))

        if not all_words:
            feedback.append("Repetition control failed: no usable words.")
            return 0.0

        unique_ratio = len(set(all_words)) / len(all_words)
        score = unique_ratio * 10.0

        repeated_excess = 0
        for word in set(all_words):
            count = all_words.count(word)
            if count > 2:
                repeated_excess += (count - 2)

        if repeated_excess > 0:
            feedback.append(
                f"Repetition issue: repeated words exceed natural usage by {repeated_excess} instance(s)."
            )
            score -= repeated_excess * 0.8

        openings = []
        for line in lines:
            words = self._extract_words(line)
            if words:
                openings.append(words[0])

        duplicated_openings = len(openings) - len(set(openings))
        if duplicated_openings > 0:
            feedback.append(
                f"Repetition issue: repeated line openings detected ({duplicated_openings})."
            )
            score -= duplicated_openings * 1.0

        return self._clip_10(score)

    def _score_transformer_poetic_texture_10(
        self,
        lines: list,
        feedback: list,
    ) -> float:
        if not lines:
            feedback.append("Poetic texture failed: no lines available.")
            return 0.0

        poem_text = "\n".join(lines)
        poem_embedding = self._embed(poem_text)

        poetic_reference = (
            "A lyrical, imagistic, symbolic, emotionally resonant poem with elevated diction."
        )
        plain_reference = (
            "A plain literal statement, instruction, grocery list, or ordinary factual prose."
        )

        poetic_emb = self._embed(poetic_reference)
        plain_emb = self._embed(plain_reference)

        poetic_sim = util.cos_sim(poem_embedding, poetic_emb).item()
        plain_sim = util.cos_sim(poem_embedding, plain_emb).item()

        poeticness = poetic_sim - plain_sim

        if poeticness >= 0.30:
            score = 10.0
        elif poeticness >= 0.20:
            score = 8.0
        elif poeticness >= 0.10:
            score = 6.0
        elif poeticness >= 0.00:
            score = 4.0
        elif poeticness >= -0.10:
            score = 2.0
        else:
            score = 0.0

        lexical_hits = 0
        for word in self._extract_words(poem_text):
            if word in self.poetic_lexicon:
                lexical_hits += 1

        if lexical_hits == 0:
            score -= 2.0
        elif lexical_hits >= 4:
            score += 1.0

        avg_words_per_line = sum(
            len(self._extract_words(line))
            for line in lines
        ) / max(len(lines), 1)

        if avg_words_per_line < 4:
            feedback.append(
                "Poetic texture issue: lines are too fragmentary."
            )
            score -= 2.0

        score = self._clip_10(score)

        if score == 0.0:
            feedback.append(
                f"Poetic texture failed: transformer poeticness={poeticness:.3f}."
            )
        elif score <= 4.0:
            feedback.append(
                f"Poetic texture weak: transformer poeticness={poeticness:.3f}."
            )

        return score

    def _score_rhyme_10(
        self,
        lines: list,
        feedback: list,
    ) -> float:
        if len(lines) < 4:
            feedback.append("Rhyme skipped: fewer than 4 lines.")
            return 0.0

        w1 = self._last_word(lines[0])
        w2 = self._last_word(lines[1])
        w3 = self._last_word(lines[2])
        w4 = self._last_word(lines[3])

        a_rhyme = self._check_rhyme(w1, w3)
        b_rhyme = self._check_rhyme(w2, w4)

        score = 0.0

        if a_rhyme:
            score += 5.0
        else:
            feedback.append(
                f"Rhyme issue: A-rhyme failed ({w1}/{w3})."
            )

        if b_rhyme:
            score += 5.0
        else:
            feedback.append(
                f"Rhyme issue: B-rhyme failed ({w2}/{w4})."
            )

        return self._clip_10(score)

    # ---------------------------------------------------------
    # AGGREGATION
    # ---------------------------------------------------------

    def _score10_to_scalar(
        self,
        score_10: float,
        fatal_zero_keys: list,
    ) -> float:
        if fatal_zero_keys:
            # Fatal failures should always discourage the model.
            # More fatal categories means stronger punishment.
            penalty = -0.5 * len(fatal_zero_keys)
            return round(max(-1.0, penalty), 4)

        # Non-fatal scores map from 0-10 into 0-1.
        # This keeps successful poems positive for RL.
        scalar = score_10 / 10.0

        return round(max(0.0, min(1.0, scalar)), 4)
    
    def _status_from_result(
        self,
        score_10: float,
        fatal_zero_keys: list,
    ) -> str:
        if fatal_zero_keys:
            return "FAILED"
        if score_10 >= 8.0:
            return "NOMINAL"
        if score_10 >= 6.0:
            return "UNSTABLE"
        if score_10 >= 4.0:
            return "WEAK"
        return "FAILED"

    # ---------------------------------------------------------
    # MAIN INTERFACE
    # ---------------------------------------------------------

    def weigh_heart(
        self,
        draft: str,
        query_embedding,
        context_embedding,
    ) -> dict:
        feedback = []

        if not draft or not str(draft).strip():
            subscores_10 = {
                "structure": 0.0,
                "instruction_cleanliness": 0.0,
                "cadence": 0.0,
                "context_vector_alignment": 0.0,
                "query_vector_alignment": 0.0,
                "list_fragment_control": 0.0,
                "repetition_control": 0.0,
                "poetic_texture": 0.0,
                "rhyme": 0.0,
            }

            feedback.append("Empty generation.")

            score_10 = 0.0
            scalar_score = -1.0
            status = "FAILED"

            return {
                "score": scalar_score,
                "score_10": score_10,
                "status": status,
                "subscores_10": subscores_10,
                "feedback": feedback,
                "loss": 1.0,
            }

        lines = [
            line.strip()
            for line in draft.split("\n")
            if line.strip()
        ]

        subscores_10 = {
            "structure": self._score_structure_10(
                lines=lines,
                feedback=feedback,
            ),

            "instruction_cleanliness": self._score_instruction_cleanliness_10(
                draft=draft,
                feedback=feedback,
            ),

            "cadence": self._score_cadence_10(
                lines=lines,
                feedback=feedback,
            ),

            "context_vector_alignment": self._score_context_vector_alignment_10(
                draft=draft,
                context_embedding=context_embedding,
                feedback=feedback,
            ),

            "query_vector_alignment": self._score_query_vector_alignment_10(
                draft=draft,
                query_embedding=query_embedding,
                feedback=feedback,
            ),

            "list_fragment_control": self._score_list_fragment_control_10(
                lines=lines,
                feedback=feedback,
            ),

            "repetition_control": self._score_repetition_control_10(
                lines=lines,
                feedback=feedback,
            ),

            "poetic_texture": self._score_transformer_poetic_texture_10(
                lines=lines,
                feedback=feedback,
            ),

            "rhyme": self._score_rhyme_10(
                lines=lines,
                feedback=feedback,
            ),
        }

        score_10 = round(
            sum(subscores_10.values()) / len(subscores_10),
            3,
        )

        NONFATAL_ZERO_KEYS = {
            "rhyme",
            "query_vector_alignment",
        }

        fatal_zero_keys = [
            key for key, value in subscores_10.items()
            if value == 0.0 and key not in NONFATAL_ZERO_KEYS
        ]

        if fatal_zero_keys:
            feedback.append(
                "Fatal zero subscore(s): " + ", ".join(fatal_zero_keys) + "."
            )

        scalar_score = self._score10_to_scalar(
            score_10=score_10,
            fatal_zero_keys=fatal_zero_keys,
        )

        status = self._status_from_result(
            score_10=score_10,
            fatal_zero_keys=fatal_zero_keys,
        )

        loss = round(1.0 - scalar_score, 4)

        return {
            "score": scalar_score,
            "score_10": score_10,
            "status": status,
            "subscores_10": subscores_10,
            "feedback": feedback if feedback else ["Perfect harmony."],
            "loss": loss,
        }