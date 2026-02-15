import re

def get_intro_text() -> str:
    return (
        "Hello! I'm Olivia. Nice to meet you. "
        "I love hearing people's stories. "
        "Where are you from?"
    )



def extract_location_hint(text: str) -> str:
    if not text:
        return ""
    t = text.strip()

    m = re.search(r"\bfrom\s+([A-Za-z][A-Za-z\s\-]{1,40})\b", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(" .,!?:;")

    if len(t.split()) <= 3:
        return t.strip(" .,!?:;")

    return ""
