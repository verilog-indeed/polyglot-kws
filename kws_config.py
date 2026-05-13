"""
Shared configuration for the multilingual KWS project.
Imported by both generate_tts.py and the training notebook.
Edit here; changes propagate everywhere automatically.
"""

VOICES = {
    "en": ["en-US-AriaNeural", "en-US-GuyNeural", "en-US-JennyNeural",
           "en-US-AndrewMultilingualNeural", "en-US-EmmaMultilingualNeural",
           "en-GB-RyanNeural", "en-GB-SoniaNeural", "en-GB-ThomasNeural",
           "en-AU-WilliamNeural", "en-AU-NatashaNeural",
           "en-IN-NeerjaNeural", "en-CA-ClaraNeural"],
    "de": ["de-DE-KatjaNeural", "de-DE-ConradNeural", "de-DE-AmalaNeural",
           "de-DE-KillianNeural", "de-DE-MajaNeural", "de-DE-BerndNeural",
           "de-DE-FlorianMultilingualNeural",
           "de-AT-IngridNeural", "de-AT-JonasNeural",
           "de-CH-LeniNeural", "de-CH-JanNeural"],
    "tr": ["tr-TR-AhmetNeural", "tr-TR-EmelNeural"],
    "ar": ["ar-EG-SalmaNeural", "ar-EG-ShakirNeural",
           "ar-SA-HamedNeural", "ar-SA-ZariyahNeural",
           "ar-DZ-AminaNeural", "ar-MA-MounaNeural",
           "ar-KW-FahedNeural", "ar-KW-NouraNeural",
           "ar-LB-RamiNeural", "ar-LB-LaylaNeural"],
    "fr": ["fr-FR-DeniseNeural", "fr-FR-HenriNeural",
           "fr-FR-RemyMultilingualNeural", "fr-FR-VivienneMultilingualNeural",
           "fr-CA-SylvieNeural", "fr-CA-JeanNeural",
           "fr-CH-ArianeNeural", "fr-BE-CharlineNeural", "fr-BE-GerardNeural"],
    "fa": ["fa-IR-DilaraNeural", "fa-IR-FaridNeural"],
}

KEYWORDS = {
    "en": {"activate": ["activate"],    "deactivate": ["deactivate"],    "play": ["play"],      "stop": ["stop"]},
    "de": {"activate": ["aktivieren"],  "deactivate": ["deaktivieren"],  "play": ["abspielen"], "stop": ["stopp"]},
    "tr": {"activate": ["etkinleştir"], "deactivate": ["kapat"],         "play": ["oynat"],     "stop": ["dur"]},
    "ar": {"activate": ["فعّل"],         "deactivate": ["عطّل"],           "play": ["شغّل"],       "stop": ["أوقف"]},
    "fr": {"activate": ["activer"],     "deactivate": ["désactiver"],    "play": ["lancer"],    "stop": ["arrêter"]},
    "fa": {"activate": ["فعال کن"],      "deactivate": ["غیرفعال کن"],    "play": ["پخش کن"],    "stop": ["متوقف کن"]},
}

UNKNOWN_WORDS = {
    "en": ["record", "cancel", "launch", "select", "open", "skip"],
    "de": ["aufnehmen", "abbrechen", "starten", "auswählen", "öffnen", "überspringen"],
    "tr": ["kaydet", "iptal", "başlat", "seç", "aç", "geç"],
    "ar": ["سجّل", "ألغِ", "ابدأ", "اختر", "افتح", "تجاوز"],
    "fr": ["enregistrer", "annuler", "démarrer", "sélectionner", "ouvrir", "ignorer"],
    "fa": ["ضبط کن", "لغو کن", "شروع کن", "انتخاب کن", "باز کن", "رد کن"],
}

# Pitch/rate variants synthesised per (word, voice) pair.
# Variant 0 keeps the base filename (word__voice.wav); variants 1+ get __v1, __v2, … suffixes.
VARIANTS = [
    ("+0%",   "+0Hz"),   # neutral
    ("+12%",  "+8Hz"),   # faster + higher
    ("-12%",  "-8Hz"),   # slower + lower
    ("+8%",   "-6Hz"),   # faster + slightly lower
    ("-8%",   "+6Hz"),   # slower + slightly higher
    ("+20%",  "+0Hz"),   # much faster, neutral pitch
    ("-20%",  "+0Hz"),   # much slower, neutral pitch
    ("+0%",   "+15Hz"),  # neutral speed, higher pitch
    ("+0%",   "-15Hz"),  # neutral speed, lower pitch
]
