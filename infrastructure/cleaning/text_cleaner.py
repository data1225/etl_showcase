
import re
import unicodedata

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002700-\U000027BF"  # dingbats
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U00002600-\U000026FF"  # Misc symbols
    "]+",
    flags=re.UNICODE,
)

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)

# 為避免替換後文字黏在一起，所以把目標內容取代成空白，
# 如「段落一https://example.com/段落二」，要變成「段落一 段落二」非「段落一段落二」
def remove_urls(text: str) -> str:
    return URL_RE.sub(" ", text)
def strip_html(text: str) -> str:
    return HTML_TAG_RE.sub(" ", text)
def remove_emojis(text: str) -> str:
    return EMOJI_RE.sub(" ", text)
def remove_noise(text: str) -> str:
    """
    Remove excessive punctuation other than CJK characters
    """
    text = re.sub(r"[^\w\u4e00-\u9fff\u3040-\u30ff\u3400-\u9FFF\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text
def remove_all_punctuation(text: str) -> str:
    """
    移除文字中的所有標點符號及特殊字元，僅保留字母、數字、中日文字元和單一空格。
    """
    text = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff\u3040-\u30ff\u3400-\u9FFF\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()    
    return text

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = normalize_unicode(text)
    text = strip_html(text)
    text = remove_urls(text)
    text = remove_emojis(text)
    text = remove_noise(text)
    return text
