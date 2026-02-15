from google.cloud import texttospeech

def main():
    client = texttospeech.TextToSpeechClient()
    resp = client.list_voices()

    wanted_prefixes = ("en-GB", "da-DK", "en-IE", "en-AU")
    for v in resp.voices:
        if any(lang.startswith(wanted_prefixes) for lang in v.language_codes) or v.name.startswith(wanted_prefixes):
            langs = ",".join(v.language_codes)
            print(f"{v.name:30}  langs={langs:15}  gender={texttospeech.SsmlVoiceGender(v.ssml_gender).name}")

if __name__ == "__main__":
    main()
