import re
import random
import pronouncing

try:
    from core.mythic_phonetics import normalize_for_rhyme
except ImportError:
    from mythic_phonetics import normalize_for_rhyme


class SpinDoctor:
    """
    Spin Doctor = Aesthetic/rhyme repair layer.

    Takes structured 4-line output from T5Poet and applies:
    - light style smoothing
    - mythic-name phonetic normalization
    - ABAB rhyme repair when possible

    Core philosophy:
    The Poet writes structure.
    The Spin Doctor makes it sing.
    """

    def __init__(self):
        self.modes = ["lyrical", "minimal", "archaic", "modern"]

        # Safe fallback rhyme pairs.
        # Used only when pronouncing cannot find a useful rhyme.
        self.safe_rhyme_pairs = {
            "stone": "throne",
            "throne": "stone",
            "rain": "plain",
            "plain": "rain",
            "night": "light",
            "light": "night",
            "sea": "tree",
            "tree": "sea",
            "fire": "choir",
            "choir": "fire",
            "hand": "land",
            "land": "hand",
            "grave": "wave",
            "wave": "grave",
            "name": "flame",
            "flame": "name",
            "sky": "lie",
            "lie": "sky",
        }

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------

    def repair_poem(self, poem: str, mode: str | None = None) -> str:
            """
            Main method for main.py.

            Input:
                raw poem string

            Output:
                repaired poem string
            """

            # Do not repair fallback poems.
            # The critic needs the marker intact so RL can punish fallback behavior.
            if "[fallback]" in poem.lower():
                return poem

            result = self.process(poem, mode=mode)
            spun_poem = result["poem"]

            lines = self._split_lines(spun_poem)

            if len(lines) != 4:
                return spun_poem

            lines = self._repair_abab(lines)

            return "\n".join(lines)

    def process(self, poem: str, mode: str | None = None) -> dict:
        """
        Input: 4-line raw poem
        Output: refined poem + metadata
        """

        if not poem:
            return {
                "poem": "",
                "mode": mode or "minimal",
                "notes": "empty input",
            }

        lines = self._split_lines(poem)

        if len(lines) < 4:
            lines += ["..."] * (4 - len(lines))
        elif len(lines) > 4:
            lines = lines[:4]

        mode = mode or random.choice(self.modes)

        if mode == "minimal":
            transformed = self._minimal(lines)
        elif mode == "archaic":
            transformed = self._archaic(lines)
        elif mode == "modern":
            transformed = self._modern(lines)
        else:
            transformed = self._lyrical(lines)

        return {
            "poem": "\n".join(transformed),
            "mode": mode,
            "notes": "spin applied",
        }

    # ---------------------------------------------------------
    # LINE HELPERS
    # ---------------------------------------------------------

    def _split_lines(self, poem: str) -> list[str]:
        return [
            line.strip()
            for line in poem.split("\n")
            if line.strip()
        ]

    def _extract_words(self, text: str) -> list[str]:
        return re.findall(r"\b[a-zA-Z']+\b", text)

    def _last_word(self, line: str) -> str:
        clean_line = re.sub(
            r"\[FALLBACK\]",
            "",
            line,
            flags=re.IGNORECASE,
        )

        words = self._extract_words(clean_line)

        return words[-1] if words else ""

    def _replace_last_word(self, line: str, new_word: str) -> str:
        return re.sub(
            r"\b[a-zA-Z']+\b(?=[^a-zA-Z']*$)",
            new_word,
            line,
        )

    # ---------------------------------------------------------
    # RHYME LOGIC
    # ---------------------------------------------------------

    def _phonetic_word(self, word: str) -> str:
        return normalize_for_rhyme(word).lower().strip()

    def _rhymes(self, word_a: str, word_b: str) -> bool:
        a = self._phonetic_word(word_a)
        b = self._phonetic_word(word_b)

        if not a or not b:
            return False

        rhymes = pronouncing.rhymes(a)

        if b in rhymes:
            return True

        # Crude fallback, useful for unknown mythic words.
        if len(a) >= 3 and len(b) >= 3:
            return a[-3:] == b[-3:]

        return False

    def _find_rhyme_for(self, target_word: str) -> str | None:
        target = self._phonetic_word(target_word)

        if not target:
            return None

        rhymes = pronouncing.rhymes(target)

        for candidate in rhymes:
            if candidate.isalpha() and 3 <= len(candidate) <= 8:
                return candidate

        if target in self.safe_rhyme_pairs:
            return self.safe_rhyme_pairs[target]

        return None

    def _repair_pair(self, line_a: str, line_b: str) -> tuple[str, str]:
        word_a = self._last_word(line_a)
        word_b = self._last_word(line_b)

        if not word_a or not word_b:
            return line_a, line_b

        if self._rhymes(word_a, word_b):
            return line_a, line_b

        replacement = self._find_rhyme_for(word_a)

        if not replacement:
            return line_a, line_b

        repaired_b = self._replace_last_word(line_b, replacement)

        return line_a, repaired_b

    def _repair_abab(self, lines: list[str]) -> list[str]:
        """
        ABAB:
            line 1 rhymes with line 3
            line 2 rhymes with line 4
        """

        if len(lines) != 4:
            return lines

        lines[0], lines[2] = self._repair_pair(lines[0], lines[2])
        lines[1], lines[3] = self._repair_pair(lines[1], lines[3])

        return lines

    # ---------------------------------------------------------
    # STYLE PROFILES
    # ---------------------------------------------------------

    def _minimal(self, lines):
        return [self._compress(line) for line in lines]

    def _archaic(self, lines):
        return [self._archaicify(line) for line in lines]

    def _modern(self, lines):
        return [self._modernize(line) for line in lines]

    def _lyrical(self, lines):
        return [self._smooth(line) for line in lines]

    # ---------------------------------------------------------
    # TRANSFORMS
    # ---------------------------------------------------------

    def _compress(self, line):
        line = re.sub(r"\b(very|really|just|so)\b", "", line)
        return re.sub(r"\s+", " ", line).strip()

    def _archaicify(self, line):
        line = line.replace("you", "thee")
        line = line.replace("your", "thy")
        line = line.replace("are", "art")
        return line

    def _modernize(self, line):
        line = re.sub(r"\s+", " ", line)
        return line.strip()

    def _smooth(self, line):
        words = line.split()

        if len(words) > 10 and random.random() > 0.5:
            mid = len(words) // 2
            words.insert(mid, ",")

        return " ".join(words).replace(" ,", ",")