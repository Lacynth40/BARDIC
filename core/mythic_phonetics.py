import re

# A hardcoded dictionary for the most common/complex names in our datasets
# mapped to English phonetic approximations that standard rhyme engines understand.
MYTHIC_DICTIONARY = {
    # Mabinogion (Welsh)
    "pwyll": "poo-eel",
    "dyved": "duh-ved",
    "llew": "loo",       # Approximating the lateral fricative 'll' to 'l'
    "llaw": "law",
    "gyffes": "guff-ess",
    "gronw": "gron-oo",
    "pebyr": "peb-eer",
    "cynvael": "kin-vile",
    "blodeuwedd": "blod-eh-weth",
    "glyn": "glin",
    "cuch": "keek",      # Approximating the 'ch'
    
    # Poetic Edda (Old Norse / Icelandic)
    "othin": "o-thin",
    "sigurth": "see-gurth",
    "hogni": "hog-nee",
    "yggdrasil": "ig-druh-sil",
    "jotunheim": "yo-tun-hame",
    "valfather": "val-fa-ther",
    "mithgarth": "mith-garth",
    "mjollnir": "myol-neer",
    "valkyrie": "val-keer-ee",
    "glaumvor": "glom-vor",
    "völund": "vo-lund"
}

def translate_mythic_phonetics(word):
    """
    Takes a word. If it's in our hardcoded mythic dictionary, it returns the 
    English phonetic spelling. If not, it applies basic Welsh/Norse regex rules 
    to try and save the rhyme.
    """
    clean_word = word.lower().strip('.,!?;:"()')
    
    # 1. Check the hardcoded matrix first
    if clean_word in MYTHIC_DICTIONARY:
        return MYTHIC_DICTIONARY[clean_word]
        
    # 2. Apply generalized Welsh fallback rules
    # Note: These are rough approximations meant ONLY to trick an English rhyme engine
    fallback = clean_word
    
    # If it looks Welsh (contains 'w' as a vowel, 'll', 'dd', etc.)
    if re.search(r'll|dd|w[^aeiou]', fallback) or fallback.endswith('w'):
        fallback = fallback.replace('ff', 'f')
        fallback = fallback.replace('dd', 'th')
        fallback = fallback.replace('ll', 'l')
        fallback = fallback.replace('f', 'v')
        # Handle 'w' as 'oo'
        fallback = re.sub(r'w$', 'oo', fallback)
        fallback = re.sub(r'w([^aeiouy])', r'oo\1', fallback)
        
    # 3. Apply generalized Old Norse fallback rules
    # If it looks Norse (j acting as y)
    if 'j' in fallback and not fallback.startswith('j'):
        fallback = fallback.replace('j', 'y')
        
    return fallback

def inject_phonetics(draft_lines):
    """
    Scans a draft and builds a phonetic map for the ArtCritic.
    Returns a dictionary mapping the original ancient word to its English pronunciation.
    """
    injected_map = {}
    for line in draft_lines:
        words = line.split()
        for word in words:
            clean_word = word.lower().strip('.,!?;:"()')
            translated = translate_mythic_phonetics(clean_word)
            # Only add to the map if we actually changed the spelling
            if translated != clean_word:
                injected_map[clean_word] = translated
                
    return injected_map

def normalize_for_rhyme(word: str) -> str:
    """
    Compatibility wrapper for rhyme/phonetic tools.
    Converts mythic names into English-ish phonetic forms
    before sending them to rhyme engines.
    """
    return translate_mythic_phonetics(word)