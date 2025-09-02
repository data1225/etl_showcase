
from typing import Optional, List

def safe_str(x: Optional[str]) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        try:
            return str(x)
        except Exception:
            return ""
    return x

def normalize_keywords(keywords: Optional[str]) -> List[str]:
    """
    Accepts a string of comma/space/# separated keywords and turns it into a unique list.
    """
    s = safe_str(keywords)
    # Split on commas, spaces, and hashtags, preserve CJK
    raw = []
    token = ""
    for ch in s:
        if ch in [",", "，", "、"] or ch.isspace() or ch == "#":
            if token:
                raw.append(token)
                token = ""
        else:
            token += ch
    if token:
        raw.append(token)
    # Deduplicate while preserving order
    seen = set()
    result = []
    for t in raw:
        t = t.strip()
        if not t:
            continue
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result

def merge_text_fields(title: Optional[str], description: Optional[str], keywords: Optional[str]) -> str:
    """
    Merge title, description, and keywords into a single corpus string.
    """
    title_s = safe_str(title).strip()
    desc_s = safe_str(description).strip()
    kws = normalize_keywords(keywords)
    parts = [title_s, desc_s, " ".join(kws)]
    # Remove empty segments and deduplicate segments
    merged = " \n".join([p for p in parts if p])
    return merged.strip()
