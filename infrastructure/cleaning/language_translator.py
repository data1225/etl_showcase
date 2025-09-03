import socket
import pandas as pd
import plotly.graph_objects as go
from typing import List, Optional, Literal
from bertopic import BERTopic
from langdetect import detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from tqdm.notebook import tqdm

# deep-translator 會透過網路連線使用線上翻譯服務，為避免長時間等待，設定了連線 timeout。
socket.setdefaulttimeout(30)

try:
    from opencc import OpenCC
except Exception:
    OpenCC = None

_TRANSLATOR_BACKENDS = []
try:
    from deep_translator import GoogleTranslator as _GT
    _TRANSLATOR_BACKENDS.append("deep_translator")
except Exception:
    _GT = None

# Ensure deterministic detection
DetectorFactory.seed = 42

class LanguageTranslator:
    def __init__(self, target_variant: Literal["zh-TW", "zh-CN"] = "zh-TW"):
        self.target_variant = target_variant
        if OpenCC is not None:
            self.cc_to_tw = OpenCC('s2t')
            self.cc_to_cn = OpenCC('t2s')
        else:
            self.cc_to_tw = self.cc_to_cn = None

    def to_target_chinese_variant(self, text: str) -> str:
        """
        將中文文本轉換為指定的繁體或簡體變體。
        """
        if self.cc_to_tw is None or self.cc_to_cn is None:
            return text
        if self.target_variant == "zh-TW":
            return self.cc_to_tw.convert(text)
        else:
            return self.cc_to_cn.convert(text)

    def batch_detect_lang(self, texts: List[str]) -> List[Optional[str]]:
        """
        批量偵測文字語言，並修正OpenCC部分中文誤判問題。
        """
        langs = []
        for text in texts:
            try:
                if not text:
                    langs.append(None)
                    continue

                result = detect_langs(text)[0].lang
                # --- 修正: 若偵測為韓文但含有大量中文字，視為中文 ---
                if result == "ko":
                    zh_char_count = sum('\u4e00' <= ch <= '\u9fff' for ch in text)
                    if zh_char_count >= max(2, len(text) * 0.3):
                        result = "zh"
                langs.append(result)
            except LangDetectException:
                langs.append(None)
        return langs
    
    def batch_translate(self, texts: List[str], target_language: str) -> List[str]:
        """
        批量將非目標語言的文本翻譯為目標語言。
        """
        if not _GT:
            print("deep_translator 模組未安裝，無法執行批量翻譯。")
            return texts

        print(f"偵測所有文本所屬語言")
        # 偵測所有文本的語言，只呼叫一次
        detected_langs = self.batch_detect_lang(texts)

        print(f"分離目標語言文本和非目標語言文本")
        # 使用單次偵測結果來建立標誌列表，避免在迴圈中重複呼叫
        is_target_lang_flags = [
            (lang is not None and lang.startswith(target_language.split('-')[0])) if text else False
            for text, lang in zip(texts, detected_langs)
        ]
        # 分離目標語言文本和非目標語言文本
        non_target_texts = [text for i, text in enumerate(texts) if not is_target_lang_flags[i]]
        target_texts = [text for i, text in enumerate(texts) if is_target_lang_flags[i]]

        # 批量翻譯非目標語言文本
        translated_non_target = []
        if non_target_texts:
            # 將非目標語言文本分塊處理，降低處理風險
            chunk_size = 50
            non_target_chunks = [non_target_texts[i:i + chunk_size] for i in range(0, len(non_target_texts), chunk_size)]
            
            try:
                translator = _GT(source='auto', target=target_language)
                # 使用tqdm包裹分塊的迴圈，顯示進度
                for chunk in tqdm(non_target_chunks, desc=f'批量翻譯非目標語言文本 (共 {len(non_target_texts)} 筆)'):
                    translated_non_target.extend(translator.translate_batch(chunk))
            except Exception as e:
                print(f"批量翻譯錯誤: {e}")
                translated_non_target = non_target_texts

        # 整合結果
        final_texts = []
        non_target_index = 0
        target_index = 0
        for is_target in is_target_lang_flags:
            if is_target:
                final_texts.append(target_texts[target_index])
                target_index += 1
            else:
                final_texts.append(translated_non_target[non_target_index])
                non_target_index += 1
            
        return final_texts
