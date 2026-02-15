USER_PROFILE = {
    "age_group": "elderly",
    "language": "en",  
}

def get_lang_codes():
    lang = USER_PROFILE.get("language", "en")
    return ("da-DK", "da-DK") if lang == "da" else ("en-US", "en-US")