import re
import unicodedata

def tmdb_clean_up_title(title):
    pattern = r"\((?=(?:.*[A-Za-z]){3,}).*?\)"
    title = re.sub(r'\s+\)', '',re.sub(pattern, '', title)).strip()



def normalize(s: str) -> str:
    # strip accents
    decomposed = unicodedata.normalize("NFKD", s)
    no_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    # casefold for full Unicode case-insensitivity
    return no_accents.casefold()
