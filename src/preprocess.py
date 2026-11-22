"""
Part B text normalization for Persian Digikala reviews (single source of truth).

Used by step_b1_split.py now, and later by the baselines, the Transformer notebooks,
and the offline prediction script, so that training and inference see identical text.

Decisions (each one is evidence-based, see the A4/A5/A6 notes in notebook 01):
  KEEP   emoji, Latin letters, punctuation, negation words, ordinary stop words.
         (A6: 3 of the 50 most frequent words are negation/contrast cues.)
  MAP    Arabic Yeh/Kaf/Teh-marbuta/Alef-hamza -> Persian/plain forms, Arabic-Indic and
         Persian digits -> ASCII digits (digits are normalized, NOT deleted: A6 shows they
         are 2.4x more frequent in not_recommended reviews).
  DROP   HTML tags, diacritics (harakat), tatweel, zero-width/bidi control characters.
  SPACE  ZWNJ -> space (same as eda_utils.normalize_text_minimal, so A5 numbers stay comparable).
  URL    replaced by the token "URL".
  CAP    runs of 4+ identical characters are capped at 3 ("عالیییییی" -> "عالییی").
         This keeps the emphasis signal (A6: elongation is typical for high ratings) but
         removes the long tail of unique spellings. (Proposed choice; easy to change.)

normalize_text() is idempotent and never raises on None.
Run  python src/preprocess.py  for the self-test.
"""
import html
import re
import unicodedata

NORMALIZER_VERSION = "v1"
MAX_REPEAT = 3

_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
_DIACRITICS = re.compile("[\u064B-\u065F\u0670\u0640]")  # harakat, superscript alef, tatweel
_ZERO_WIDTH = re.compile("[\u200B\u200D\u200E\u200F\u2060\uFEFF\u202A-\u202E]")
_REPEAT = re.compile(r"(.)\1{%d,}" % MAX_REPEAT, re.DOTALL)
_WS = re.compile(r"\s+")

_CHAR_MAP = {
    "\u064A": "\u06CC",  # Arabic Yeh -> Persian Yeh
    "\u0649": "\u06CC",  # Alef maksura -> Persian Yeh
    "\u0643": "\u06A9",  # Arabic Kaf -> Persian Keheh
    "\u0629": "\u0647",  # Teh marbuta -> Heh
    "\u0623": "\u0627",  # Alef with hamza above -> Alef
    "\u0625": "\u0627",  # Alef with hamza below -> Alef
    "\u0671": "\u0627",  # Alef wasla -> Alef
    "\u200C": " ",       # ZWNJ -> space
}
for _i, _d in enumerate("0123456789"):
    _CHAR_MAP[chr(0x06F0 + _i)] = _d   # Persian digits
    _CHAR_MAP[chr(0x0660 + _i)] = _d   # Arabic-Indic digits
_TRANSLATION = str.maketrans(_CHAR_MAP)


def normalize_text(text) -> str:
    if text is None:
        return ""
    s = html.unescape(str(text))
    s = unicodedata.normalize("NFKC", s)
    s = _TAG.sub(" ", s)
    s = _URL.sub(" URL ", s)
    s = s.translate(_TRANSLATION)
    s = _DIACRITICS.sub("", s)
    s = _ZERO_WIDTH.sub("", s)
    s = _REPEAT.sub(lambda m: m.group(1) * MAX_REPEAT, s)
    return _WS.sub(" ", s).strip()


def _self_test() -> None:
    assert normalize_text(None) == ""
    assert normalize_text("") == ""
    assert normalize_text(" \u0643\u062a\u0627\u0628\u200c\u0647\u0627\u064a   \u062e\u0648\u0628 ") == \
        "\u06a9\u062a\u0627\u0628 \u0647\u0627\u06cc \u062e\u0648\u0628"
    assert normalize_text("\u06f1\u06f2\u06f3 \u0664\u0665\u0666") == "123 456"
    assert normalize_text("\u0639\u0627\u0644\u06cc\u06cc\u06cc\u06cc\u06cc\u06cc") == "\u0639\u0627\u0644\u06cc\u06cc\u06cc"
    assert normalize_text("<b>good</b> https://digikala.com/x ok") == "good URL ok"
    assert normalize_text("a\u064b\u0640b") == "ab"                      # diacritic + tatweel
    assert normalize_text("a\u200bb\u200fc") == "abc"                    # zero-width removed
    assert normalize_text("\U0001F60D\U0001F60D") == "\U0001F60D\U0001F60D"  # emoji kept
    for t in ["\u0628\u062f \u0646\u06cc\u0633\u062a", "A  b\n\nc", "\u0639\u0627\u0644\u06cc\u06cc\u06cc\u06cc\u06cc"]:
        assert normalize_text(normalize_text(t)) == normalize_text(t)   # idempotent
    # negation is preserved
    assert normalize_text("\u0628\u062f \u0646\u06cc\u0633\u062a") == "\u0628\u062f \u0646\u06cc\u0633\u062a"
    print(f"preprocess.py self-test OK (version {NORMALIZER_VERSION}, max_repeat={MAX_REPEAT})")


if __name__ == "__main__":
    _self_test()
