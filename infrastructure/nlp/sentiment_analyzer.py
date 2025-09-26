import torch
from transformers import pipeline
from tqdm import tqdm
from collections import Counter
from typing import List, Dict

class SentimentAnalyzer:
    """
    使用 CardiffNLP 的多語情感分析模型
    (twitter-xlm-roberta-base-sentiment-multilingual)
    支援中英文與多語文本，並加入防呆機制
    """
    def __init__(self):
        model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
        self.label_mapping = {
            "LABEL_0": "negative",
            "LABEL_1": "neutral",
            "LABEL_2": "positive"
        }
        
        try:
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=0 if torch.cuda.is_available() else -1,
                use_fast=False  # 🚀 避免 fast tokenizer 轉換失敗
            )
            
            # 從模型組態中動態取得最大長度，並減去特殊 tokens 的數量 (通常為 2)
            self.max_length = self.pipeline.model.config.max_position_embeddings - 2
            
            print("✅ 模型載入成功並已動態取得最大長度。")
            
        except Exception as e:
            print(f"❌ 載入模型時發生錯誤: {e}")
            self.pipeline = None
            self.max_length = None

    def analyze_sentiment(self, text_list: List[str]) -> List[Dict]:
        """
        批次輸入文本 list，回傳每則文本的情感標籤與分數。
        此函式使用內建的截斷功能，已加入防呆機制。
        """
        # --- 1. 防呆檢查：檢查模型是否成功載入 ---
        if not self.pipeline:
            return []

        # --- 2. 防呆檢查：驗證輸入型別 ---
        if not isinstance(text_list, list):
            raise TypeError("輸入參數必須是 List[str] 型別")
        if not all(isinstance(text, str) for text in text_list):
            raise TypeError("輸入 List 中的所有元素都必須是 str 型別")
        
        # --- 3. 防呆檢查：處理空列表 ---
        if not text_list:
            return []

        # 使用 pipeline 內建的截斷與填充功能，簡化程式碼
        results = self.pipeline(
            text_list,
            truncation=True,     # 啟用自動截斷
            max_length=self.max_length,  # 使用動態取得的長度
            padding=True,
            batch_size=32
        )

        # 統一轉換 label
        for r in results:
            if r["label"] in self.label_mapping:
                r["label"] = self.label_mapping[r["label"]]
        return results

    def analyze_sentiment_by_topic(self, topic_data: Dict[str, List[str]]):
        """
        針對不同主題 (dict: {topic_name: [texts...]}) 分析情感比例
        """
        # 防呆檢查：確保模型已載入
        if not self.pipeline:
            return {}
            
        topic_sentiment_results = {}
        for topic_name, texts in topic_data.items():                
            sentiments = self.analyze_sentiment(texts)
            positive_count = sum(1 for s in sentiments if s['label'].lower() == 'positive')
            neutral_count = sum(1 for s in sentiments if s['label'].lower() == 'neutral')
            negative_count = sum(1 for s in sentiments if s['label'].lower() == 'negative')
            total_count = len(sentiments)

            if total_count > 0:
                topic_sentiment_results[topic_name] = {
                    'positive': positive_count / total_count,
                    'neutral': neutral_count / total_count,
                    'negative': negative_count / total_count
                }
        return topic_sentiment_results