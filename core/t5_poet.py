import os
import re
import random
import torch
import pronouncing

from dotenv import load_dotenv

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
)

from sentence_transformers import SentenceTransformer, util


load_dotenv()


class T5Poet:
    """
    T5Poet = structured poem generator.

    Responsibilities:
    - Build a short poem-seed prompt from RAG anchors.
    - Generate candidate poetic text.
    - Parse generated text into candidate lines.
    - Validate lines loosely enough to collect useful material.
    - Prefer complete valid 4-line poems.
    - Otherwise assemble a 4-line poem from semantic anchor banks.
    - Use global line-bank recovery if some anchor banks are empty.
    - Only use Python fallback templates as the final emergency layer.
    - Provide compute_logprob() for rl_train.py.

    It does NOT:
    - Call SpinDoctor.
    - Call ArtCritic.
    - Run RL updates directly.
    """

    FALLBACK_TEMPLATES = [
        "{k} drifts beneath the weight of ancient stone",
        "{k} walks beyond the memory of ruined kings",
        "{k} lingers where the silent rivers bend",
        "{k} waits beneath the ashes of old fires",
        "{k} moves through halls abandoned to the rain",
        "{k} fades where shadowed mountains touch the sky",
    ]

    BAD_OUTPUT_PHRASES = [
        "output only",
        "include the word",
        "requirements",
        "no explanations",
        "one poetic line",
        "context summary",
        "write one",
        "must follow",
        "use these",
        "keywords:",
        "themes:",
        "semantic anchors:",
        "separate lines with",
        "format:",
        "rules:",
        "task:",
    ]

    POETIC_WORDS = [
        "shadow",
        "river",
        "stone",
        "ash",
        "wind",
        "memory",
        "moon",
        "silence",
        "fire",
        "mist",
        "crown",
        "oath",
        "throne",
        "dream",
        "blood",
        "rain",
        "forest",
        "sea",
        "night",
        "song",
        "law",
        "king",
        "land",
        "name",
        "grave",
        "gate",
        "storm",
        "bone",
        "dust",
        "star",
        "flame",
        "water",
        "sky",
        "root",
        "door",
        "bridge",
        "hollow",
        "winter",
        "fate",
    ]

    # Tags the model keeps hallucinating. Treat these as separators.
    PSEUDO_LINE_TAGS = [
        "tone",
        "text",
        "title",
        "section",
        "overlay",
        "color",
        "depth",
        "br",
        "image",
        "instance",
        "serial",
        "context",
        "zone",
        "zamble",
        "length",
        "direction",
        "media",
        "lighting",
    ]

    def __init__(self, config):
        poet_cfg = config.get("poet_decoder", {})

        self.model_name = poet_cfg.get(
            "model_name",
            "models/bardic_base",
        )

        self.max_source_length = poet_cfg.get(
            "max_source_length",
            256,
        )

        self.max_target_length = poet_cfg.get(
            "max_target_length",
            128,
        )

        self.max_new_tokens = poet_cfg.get(
            "max_new_tokens",
            80,
        )

        self.temperature = poet_cfg.get(
            "temperature",
            0.95,
        )

        self.top_p = poet_cfg.get(
            "top_p",
            0.92,
        )

        self.num_attempts = poet_cfg.get(
            "num_attempts",
            24,
        )

        self.debug = poet_cfg.get(
            "debug",
            False,
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model_dtype = (
            torch.bfloat16
            if self.device.type == "cuda"
            else torch.float32
        )

        print(f"[Poet] Loading model: {self.model_name}")

        self.tokenizer = T5Tokenizer.from_pretrained(
            self.model_name
        )

        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.model_dtype,
        ).to(self.device)

        self.model.config.use_cache = True
        self.model.config.tie_word_embeddings = False
        self.model.eval()

        semantic_model_name = poet_cfg.get(
            "semantic_encoder_model",
            "all-MiniLM-L6-v2",
        )

        semantic_device = poet_cfg.get(
            "semantic_encoder_device",
            "cpu",
        )

        hf_token = os.getenv("HF_TOKEN")

        self.semantic_encoder = SentenceTransformer(
            semantic_model_name,
            device=semantic_device,
            token=hf_token,
        )

        self._semantic_cache = {}

        self._available_fallbacks = list(self.FALLBACK_TEMPLATES)
        random.shuffle(self._available_fallbacks)

    # ---------------------------------------------------------
    # SYLLABLE COUNTING
    # ---------------------------------------------------------

    def count_syllables_word(self, word):
        word = re.sub(r"[^a-zA-Z']", "", str(word)).lower()

        if not word:
            return 0

        phones = pronouncing.phones_for_word(word)

        if phones:
            return pronouncing.syllable_count(phones[0])

        groups = re.findall(r"[aeiouy]+", word)

        return max(1, len(groups))

    def count_syllables(self, text):
        return sum(
            self.count_syllables_word(word)
            for word in str(text).split()
        )

    # ---------------------------------------------------------
    # TEXT CLEANING / LINE SPLITTING
    # ---------------------------------------------------------

    def clean_generated_text(self, text):
        text = str(text).strip()

        text = text.replace("</s>", "")
        text = text.replace("<pad>", "")

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _marker_pattern(self):
        pseudo_tags = "|".join(
            re.escape(tag)
            for tag in self.PSEUDO_LINE_TAGS
        )

        return (
            r"(?:"
            r"<\s*LINE\s*>|"
            r"â?\s*\bLINES?\s*>|"
            r"/\s*line\s*>|"
            r"/\s*\d+\s*>|"
            rf"\b(?:{pseudo_tags})\s*>"
            r")"
        )

    def split_poem_lines(self, text):
        """
        Split generated text into candidate lines.

        Handles:
        - LINE>
        - line>
        - Line>
        - lines>
        - <LINE>
        - /line>
        - /3>
        - tone>, text>, image>, section>, etc.
        """

        text = self.clean_generated_text(text)

        raw_lines = re.split(
            self._marker_pattern(),
            text,
            flags=re.IGNORECASE,
        )

        lines = []

        for raw in raw_lines:
            line = raw.strip()

            # Remove any remaining pseudo tags that slipped through.
            line = re.sub(
                self._marker_pattern(),
                " ",
                line,
                flags=re.IGNORECASE,
            )

            line = line.replace("<", " ").replace(">", " ")
            line = line.replace("/", " ")

            line = re.sub(r"^\s*[-*]\s*", "", line)
            line = re.sub(r"^\s*\d+[\).\s]+", "", line)
            line = re.sub(r"\s+", " ", line).strip()

            if line:
                lines.append(line)

        return lines[:4]

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    def contains_instruction_echo(self, line):
        line_lower = str(line).lower()

        return any(
            phrase in line_lower
            for phrase in self.BAD_OUTPUT_PHRASES
        )

    def looks_like_keyword_list(self, line, keywords):
        line_lower = str(line).lower()

        literal_hits = sum(
            1 for keyword in keywords
            if str(keyword).lower() in line_lower
        )

        if literal_hits >= 3 and len(line.split()) <= 10:
            return True

        if line.count(",") >= 3 and len(line.split()) <= 10:
            return True

        return False

    def valid_line(self, line, keywords=None):
        """
        Generation-time line validation.

        This is intentionally looser than ArtCritic.
        ArtCritic can punish cadence later. Here, we only need
        usable model-generated lines for bank assembly.
        """

        if keywords is None:
            keywords = []

        line = str(line).strip()

        if not line:
            return False

        if "[fallback]" in line.lower():
            return False

        if self.contains_instruction_echo(line):
            return False

        if "<" in line or ">" in line:
            return False

        if self.looks_like_keyword_list(line, keywords):
            return False

        words = re.findall(r"\b[a-zA-Z']+\b", line)

        if len(words) < 4:
            return False

        if len(words) > 18:
            return False

        syllables = self.count_syllables(line)

        # Loose generation filter. Critic remains stricter at 10-12.
        if syllables < 8 or syllables > 14:
            return False

        return True

    def valid_poem(self, lines, keywords):
        if len(lines) != 4:
            return False

        seen = set()

        for line in lines:
            normalized = line.lower().strip()

            if normalized in seen:
                return False

            seen.add(normalized)

            if not self.valid_line(line, keywords):
                return False

        return True

    # ---------------------------------------------------------
    # SEMANTIC SCORING
    # ---------------------------------------------------------

    def embed_text(self, text):
        key = str(text).strip().lower()

        if key in self._semantic_cache:
            return self._semantic_cache[key]

        emb = self.semantic_encoder.encode(
            str(text),
            convert_to_tensor=True,
        )

        self._semantic_cache[key] = emb

        return emb

    def semantic_similarity(self, left, right):
        left_emb = self.embed_text(left)
        right_emb = self.embed_text(right)

        return util.cos_sim(left_emb, right_emb).item()

    def score_line(self, line, keyword=None, summary=None, all_keywords=None):
        if all_keywords is None:
            all_keywords = []

        line = str(line).strip()
        line_lower = line.lower()

        score = 0.0

        syllables = self.count_syllables(line)

        # Prefer 11 syllables, but this is soft.
        score -= abs(11 - syllables) * 0.75

        words = [
            re.sub(r"[^a-zA-Z']", "", word).lower()
            for word in line.split()
        ]

        words = [
            word for word in words
            if word
        ]

        unique_ratio = len(set(words)) / max(len(words), 1)
        score += unique_ratio * 2.0

        for poetic_word in self.POETIC_WORDS:
            if poetic_word in line_lower:
                score += 0.25

        # Keywords are semantic anchors, not required literal tokens.
        if keyword:
            semantic_score = self.semantic_similarity(line, str(keyword))
            score += semantic_score * 4.0

            # Literal anchor use is allowed, but slightly discouraged.
            if str(keyword).lower() in line_lower:
                score -= 0.15

        # Summary is weakly useful, but too much weight causes prose-copy behavior.
        if summary:
            summary_score = self.semantic_similarity(line, summary)
            score += summary_score * 0.5

        literal_anchor_hits = sum(
            1 for keyword_item in all_keywords
            if str(keyword_item).lower() in line_lower
        )

        if literal_anchor_hits >= 3:
            score -= 4.0
        elif literal_anchor_hits == 2:
            score -= 1.5
        elif literal_anchor_hits == 1:
            score -= 0.15

        return score

    def score_poem(self, lines, summary, keywords):
        score = 0.0

        for index, line in enumerate(lines):
            keyword = keywords[index] if index < len(keywords) else None

            score += self.score_line(
                line=line,
                keyword=keyword,
                summary=summary,
                all_keywords=keywords,
            )

        openings = [
            line.split()[0].lower()
            for line in lines
            if line.split()
        ]

        score -= (
            len(openings) - len(set(openings))
        ) * 1.5

        all_words = []

        for line in lines:
            for word in line.split():
                cleaned = re.sub(r"[^a-zA-Z']", "", word).lower()

                if cleaned:
                    all_words.append(cleaned)

        if all_words:
            repetition_ratio = len(set(all_words)) / len(all_words)
            score += repetition_ratio * 2.0

        return score

    # ---------------------------------------------------------
    # PROMPTING
    # ---------------------------------------------------------

    def build_poem_prompt(self, summary, keywords):
        """
        Short corpus-like seed.

        struct.txt was mostly raw poem text, not instruction examples,
        so this prompt avoids long Wikipedia prose and rigid instructions.
        """

        keyword_text = ", ".join(str(k) for k in keywords)

        return (
            f"theme: {keyword_text}\n"
            "tone: mythic, quiet, symbolic\n"
            "image: shadow, river, stone, memory\n"
            "\n"
        )

    # ---------------------------------------------------------
    # FALLBACK
    # ---------------------------------------------------------

    def fallback_line(self, keyword):
        if not self._available_fallbacks:
            self._available_fallbacks = list(self.FALLBACK_TEMPLATES)
            random.shuffle(self._available_fallbacks)

        template = self._available_fallbacks.pop()

        return f"{template.format(k=keyword)} [FALLBACK]"

    # ---------------------------------------------------------
    # TRAINING SUPPORT
    # ---------------------------------------------------------

    def compute_logprob(self, summary, keywords, poem):
        """
        Compute log probability of an already-generated poem
        under the current model and prompt format.

        Used by rl_train.py.
        Do NOT wrap this in torch.no_grad().
        """

        prompt = self.build_poem_prompt(
            summary=summary,
            keywords=keywords,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_source_length,
        ).to(self.device)

        labels = self.tokenizer(
            poem,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_target_length,
        ).input_ids.to(self.device)

        labels = labels.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
        )

        loss = outputs.loss
        valid_token_count = (labels != -100).sum()

        logprob = -loss * valid_token_count

        return logprob

    # ---------------------------------------------------------
    # GENERATION
    # ---------------------------------------------------------

    def generate_candidate_poem(self, summary, keywords):
        prompt = self.build_poem_prompt(
            summary=summary,
            keywords=keywords,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_source_length,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=1.25,
            no_repeat_ngram_size=3,
            max_new_tokens=self.max_new_tokens,
        )

        decoded = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        ).strip()

        lines = self.split_poem_lines(decoded)

        if self.debug:
            print("\n[DEBUG RAW OUTPUT]")
            print(decoded)

            print("[DEBUG PARSED LINES]")
            for i, line in enumerate(lines, start=1):
                print(
                    f"{i}: {line} | "
                    f"syllables={self.count_syllables(line)} | "
                    f"valid={self.valid_line(line, keywords)}"
                )

        return lines

    def assemble_from_anchor_banks(
        self,
        anchor_banks,
        global_line_bank=None,
    ):
        """
        Assemble one line per semantic anchor.

        First choice:
            choose the best line from that anchor's semantic bank.

        Recovery:
            if an anchor bank is empty, choose the best unused valid line
            from the global generated-line bank.

        This avoids Python fallback when the model produced usable lines
        but semantic assignment clustered them into only some anchors.
        """

        if global_line_bank is None:
            global_line_bank = []

        selected = []
        seen = set()

        global_ranked = sorted(
            global_line_bank,
            key=lambda item: item["score"],
            reverse=True,
        )

        for index in range(4):
            bank = anchor_banks.get(index, [])

            ranked = sorted(
                bank,
                key=lambda item: item["score"],
                reverse=True,
            )

            chosen_line = None

            # First: anchor-specific bank.
            for item in ranked:
                line = item["line"].strip()
                normalized = line.lower()

                if normalized in seen:
                    continue

                chosen_line = line
                seen.add(normalized)
                break

            # Recovery: any best unused valid generated line.
            if chosen_line is None:
                for item in global_ranked:
                    line = item["line"].strip()
                    normalized = line.lower()

                    if normalized in seen:
                        continue

                    chosen_line = line
                    seen.add(normalized)
                    break

            if chosen_line is None:
                return None

            selected.append(chosen_line)

        if len(selected) != 4:
            return None

        return selected

    def generate_poem(self, summary, keywords):
        if len(keywords) != 4:
            raise ValueError(
                "generate_poem requires EXACTLY 4 keywords"
            )

        best_lines = None
        best_score = float("-inf")

        anchor_banks = {
            0: [],
            1: [],
            2: [],
            3: [],
        }

        global_line_bank = []

        for attempt in range(self.num_attempts):
            lines = self.generate_candidate_poem(
                summary=summary,
                keywords=keywords,
            )

            # Best case: complete valid poem from one attempt.
            if self.valid_poem(lines, keywords):
                score = self.score_poem(
                    lines=lines,
                    summary=summary,
                    keywords=keywords,
                )

                if score > best_score:
                    best_score = score
                    best_lines = lines

            # Collect valid lines and assign them to best semantic anchor.
            for source_position, line in enumerate(lines[:4]):
                if not self.valid_line(line, keywords):
                    continue

                best_anchor_index = None
                best_anchor_score = float("-inf")

                for anchor_index, keyword in enumerate(keywords):
                    line_score = self.score_line(
                        line=line,
                        keyword=keyword,
                        summary=summary,
                        all_keywords=keywords,
                    )

                    if line_score > best_anchor_score:
                        best_anchor_score = line_score
                        best_anchor_index = anchor_index

                if best_anchor_index is None:
                    continue

                item = {
                    "line": line,
                    "score": best_anchor_score,
                    "anchor": keywords[best_anchor_index],
                    "source_position": source_position,
                }

                anchor_banks[best_anchor_index].append(item)
                global_line_bank.append(item)

        if self.debug:
            print("\n[DEBUG ANCHOR BANKS]")
            for index, keyword in enumerate(keywords):
                print(
                    f"{index + 1}: {keyword} -> "
                    f"{len(anchor_banks.get(index, []))} valid line(s)"
                )
            print(f"Global valid lines: {len(global_line_bank)}")

        if best_lines:
            return "\n".join(best_lines)

        assembled_lines = self.assemble_from_anchor_banks(
            anchor_banks=anchor_banks,
            global_line_bank=global_line_bank,
        )

        if assembled_lines:
            return "\n".join(assembled_lines)

        fallback_lines = [
            self.fallback_line(keyword)
            for keyword in keywords
        ]

        return "\n".join(fallback_lines)