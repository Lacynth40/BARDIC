import os
import yaml
import re
import nltk
from nltk.corpus import cmudict

class ArtCritic:
    """
    The Deterministic Rule Engine of the BARDIC architecture.
    This class evaluates the raw text output of the T5 Poet and assigns 
    a mathematical reward based on strict structural constraints:
    syllable counts, rhyme schemes, and correct token usage.
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Bootstraps the Critic. Loads the architectural rules from the config 
        and pulls the CMU phonetic dictionary into memory.
        """
        if not os.path.exists(config_path):
            config_path = os.path.join("..", config_path)
            
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
            
        self.constraints = self.config["bardic_constraints"]
        
        # We need the CMU Pronouncing Dictionary to calculate syllables and rhymes.
        # This try-except block downloads it silently on the very first run.
        try:
            nltk.data.find('corpora/cmudict')
        except LookupError:
            print("Downloading the NLTK phonetic dictionary...")
            nltk.download('cmudict', quiet=True)
            
        self.pronouncing_dict = cmudict.dict()

    def _count_syllables(self, word: str) -> int:
        """
        Calculates the exact syllable count of a word.
        """
        # Strip out punctuation and lowercase it to match the dictionary
        word = re.sub(r'[^a-z]', '', word.lower())
        if not word: 
            return 0
        
        # If the word is in the dictionary, count the phonetic vowel stresses (the numbers)
        if word in self.pronouncing_dict:
            return len([p for p in self.pronouncing_dict[word][0] if p[-1].isdigit()])
        
        # Crude architectural fallback for highly unusual words or weird AI hallucinations.
        # Just counts vowel groupings.
        return len(re.findall(r'[aeiouy]+', word))

    def _get_rhyme_ending(self, word: str) -> str:
        """
        Extracts the final phoneme sounds to determine if two words actually rhyme,
        bypassing English spelling tricks (e.g., "weight" vs "late").
        """
        word = re.sub(r'[^a-z]', '', word.lower())
        if not word or word not in self.pronouncing_dict:
            return word 
            
        phonemes = self.pronouncing_dict[word][0]
        # Find the primary vowel stress and return everything after it
        for i, p in enumerate(phonemes):
            if '1' in p: 
                return "-".join(phonemes[i:])
                
        # Fallback to the last two phonemes
        return "-".join(phonemes[-2:]) 

    def score_draft(self, draft: str) -> float:
        """
        The main evaluation loop. Grades the draft and returns a scalar reward 
        between 0.0 (utter garbage) and 1.0 (perfectly constrained poem).
        """
        reward = 0.0
        
        # 1. TOKEN PENALTY: Did it use the special architecture tokens?
        if "<TITLE>" in draft and "<LINE>" in draft:
            reward += 0.2
        else:
            # Instant failure. If it ignored the architecture, it gets nothing.
            return 0.0 
            
        # Parse the lines
        parts = draft.split("<LINE>")
        lines = [line.strip() for line in parts[1:]] 
        
        # 2. STRUCTURE PENALTY: Did it write the correct number of lines?
        target_lines = self.constraints["lines_per_stanza"] * self.constraints["num_stanzas"]
        if len(lines) == target_lines:
            reward += 0.2
        else:
            return reward # Halt grading if the structure is broken
            
        # 3. METER CHECK: Target Syllables (10 per line)
        syllable_score = 0.0
        end_words = []
        for line in lines:
            words = line.split()
            if not words: continue
            end_words.append(words[-1])
            
            line_syllables = sum(self._count_syllables(w) for w in words)
            target_syllables = self.constraints["target_syllables_per_line"]
            
            # Max score if exactly target, decay as it gets further away
            diff = abs(target_syllables - line_syllables)
            syllable_score += max(0, 1.0 - (diff * 0.2)) 
            
        if lines:
            reward += 0.3 * (syllable_score / len(lines))
            
        # 4. RHYME CHECK: Strict ABAB Enforcement
        if len(end_words) >= 4:
            rhyme_a1 = self._get_rhyme_ending(end_words[0])
            rhyme_b1 = self._get_rhyme_ending(end_words[1])
            rhyme_a2 = self._get_rhyme_ending(end_words[2])
            rhyme_b2 = self._get_rhyme_ending(end_words[3])
            
            # If A rhymes with A, B rhymes with B, and A does NOT rhyme with B
            if rhyme_a1 == rhyme_a2 and rhyme_b1 == rhyme_b2 and rhyme_a1 != rhyme_b1:
                reward += 0.3 
        
        # Return a clean float for the REINFORCE algorithm
        return round(reward, 3)

# --- Sanity Check Block ---
if __name__ == "__main__":
    critic = ArtCritic(config_path="config/settings.yaml")
    
    # Test 1: The garbage output our untrained T5 just generated
    bad_draft = "True"
    
    # Test 2: A perfectly formatted, 10-syllable ABAB quatrain 
    good_draft = (
        "<TITLE> A Test of Time "
        "<LINE> The heavy sun begins to fade away "
        "<LINE> A quiet shadow falls across the street "
        "<LINE> Tomorrow brings another working day "
        "<LINE> But for tonight the quiet is complete"
    )
    
    print("\n[TEST 1] Scoring the untrained T5 output:")
    print(f"Draft: '{bad_draft}'")
    print(f"Score: {critic.score_draft(bad_draft)}")
    
    print("\n[TEST 2] Scoring a perfectly architected quatrain:")
    print(f"Draft: '{good_draft}'")
    print(f"Score: {critic.score_draft(good_draft)}")