from typing import Optional, List
import pandas as pd

from .text_merger import safe_str, merge_text_fields
from .text_cleaner import clean_text
from .language_translator import LanguageTranslator

def google_sheet_to_dataframe(values: List[List[str]], column_order: List[str]) -> pd.DataFrame:
    """
    Convert raw values (first row headers) from Google Sheet to a typed DataFrame,
    """
    if not values or len(values) < 2:
        return pd.DataFrame(columns=column_order)
    header = values[0]
    rows = values[1:]

    # Ensure expected column order exists
    df = pd.DataFrame(rows, columns=header)
    # Reindex to expected order when possible
    for col in column_order:
        if col not in df.columns:
            df[col] = None
    df = df[column_order]
    return df


def preprocess_dataframe(
    df: pd.DataFrame,
    target_variant: str = "zh-TW",
    merge_fields: Optional[List[str]] = None
) -> pd.DataFrame:
    translator = LanguageTranslator(target_variant=target_variant)

    print(f'開始合併文字欄位，共 {len(df)} 筆資料')
    # === Step 1: 合併文字欄位 ===
    if merge_fields is None:
        # 預設合併 Video title、Video description
        merge_fields = ["Video title", "Video description"]

    merged_texts: List[str] = []
    for _, row in df.iterrows():
        if merge_fields is None:
            # 沒有 merge_fields，取預設欄位
            title = safe_str(row.get("title"))
            description = safe_str(row.get("description"))
            keywords = safe_str(row.get("keywords"))
        else:
            # 第一個當 title，第二個當 description，其餘當 keywords
            values = [safe_str(row.get(col)) for col in merge_fields]
            title = values[0] if len(values) > 0 else ""
            description = values[1] if len(values) > 1 else ""
            keywords = " ".join(values[2:]) if len(values) > 2 else ""
        merged_texts.append(merge_text_fields(title, description, keywords))

    print(f'開始清理雜訊')
    # === Step 2: 清理雜訊 ===
    cleaned_texts: List[str] = [clean_text(text) for text in merged_texts]
    

    print(f'開始批次翻譯')
    # === Step 3: 批次翻譯 ===
    translated_texts: List[str] = translator.batch_translate(
        cleaned_texts, target_language="zh-TW" if target_variant == "zh-TW" else "zh-CN"
    )

    print(f'開始統一中文變體（繁體/簡體）')
    # === Step 4: 統一中文變體 ===
    translated_texts = [
        translator.to_target_chinese_variant(text) if text else ""
        for text in translated_texts
    ]

    # === Step 5: 回寫至 DataFrame ===
    df = df.copy()
    df["merged_text"] = merged_texts
    df["cleaned_text"] = cleaned_texts
    df["text_translated"] = translated_texts

    return df