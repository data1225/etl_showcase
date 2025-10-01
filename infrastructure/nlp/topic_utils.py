import inspect
import pandas as pd
from bertopic import BERTopic
from umap import UMAP
from tqdm.notebook import tqdm
from typing import Dict

def default_n_neighbors():
    signature = inspect.signature(UMAP.__init__)
    default_n_neighbors = signature.parameters["n_neighbors"].default
    return default_n_neighbors

def get_topic_coordinates(topic_model: BERTopic):
    """
    從 BERTopic 模型的內部取得主題的 UMAP 降維座標。
    """
    # 這裡直接呼叫 visualize_topics 來生成圖表，然後從其數據中提取座標
    fig = topic_model.visualize_topics()
    data = fig.data[0]
    
    # 提取 x 和 y 座標
    x_coords = data['x']
    y_coords = data['y']
    
    # 提取主題數量（用於對應座標）
    topic_info = topic_model.get_topic_info()
    topic_ids = topic_info[topic_info['Topic'] != -1]['Topic'].tolist()
    
    # 將座標與主題 ID 進行映射
    coords_map = {topic_id: (x_coords[i], y_coords[i]) for i, topic_id in enumerate(topic_ids)}
    
    return coords_map

def transform_to_refined_topics_by_culture(topic_info: pd.DataFrame, topic_mapping_list: Dict[str, Dict[str, str]], culture: str):
    """
    精煉主題清單，如果主題包含特定關鍵字，則用預定義的精煉主題取代。

    Args:
        topic_info (pd.DataFrame): 包含 'Name' 列的 DataFrame。'Name' 的內容是由關鍵字組成的字串，例如 '政治_權力_鬥爭'。
        topic_mapping_list (Dict[str, Dict[str, str]]): 包含不同語言或文化的精煉主題映射字典。
                                                         外層的 key 為文化代碼（如 'zh'），內層的 key 為精煉後的主題名稱，
                                                         value 為包含關鍵字的列表。
        culture (str): 用來選擇 topic_mapping_list 中特定語言的 key，例如 'zh' 或 'en'。

    Returns:
        Dict[int, str]: 包含精煉後主題名稱的字典。key 為原始主題 ID，value 為精煉後的主題名稱。
    """
    # 根據 culture 參數從 topic_mapping_list 中取得對應的精煉主題映射
    topic_mapping = topic_mapping_list.get(culture)
    if not topic_mapping:
        raise ValueError(f"Culture '{culture}' not found in topic mapping data.")

    refined_topics = transform_to_refined_topics(topic_info, topic_mapping)
    return refined_topics


def transform_to_refined_topics(topic_info: pd.DataFrame, topic_mapping: Dict[str, str]):
    """
    精煉主題清單，如果主題包含特定關鍵字，則用預定義的精煉主題取代。
    並過濾掉關鍵字均為空值的異常資料。

    Args:
        topic_info (pd.DataFrame): 包含 'Name' 列的 DataFrame。'Name' 的內容是由關鍵字組成的字串，例如 '政治_權力_鬥爭'。
        topic_mapping (Dict[str, str]): 精煉主題映射字典。

    Returns:
        Dict[int, str]: 包含精煉後主題名稱的字典。key 為原始主題 ID，value 為精煉後的主題名稱。
    """

    refined_topics = {}
    
    # 移除 ID = -1 的主題 (通常是 Outlier)
    topic_data = topic_info[topic_info['Topic'] != -1]
    
    for row in tqdm(topic_data.itertuples(), total=len(topic_data), leave=False, desc=f"精煉主題清單"):
        topic_id = row.Topic
        topic_keywords = row.Name # e.g., '19_，_zhan_他_可以' or '復仇_權謀_犧牲'

        # 1. 關鍵檢查：將關鍵詞字串拆分成列表
        # 由於BERTopic的主題名稱格式通常是 "ID_keyword1_keyword2..."
        # 這裡假設 Name 欄位已經包含了關鍵詞，並以 '_' 分隔。
        # 注意：BERTopic的 Name 格式通常是 "ID_keyword1_keyword2"，所以我們需要處理ID。
        
        # 假設 Name 欄位格式是 "TopicID_Keyword1_Keyword2..."，我們先移除 TopicID
        try:
            keywords_str = topic_keywords.split('_', 1)[1] if '_' in topic_keywords else topic_keywords
        except IndexError:
            # 如果主題名稱只有ID，這是一個極端情況，我們視為空
            keywords_str = ""

        # 2. 徹底清理關鍵詞列表：將關鍵詞字串拆分成列表，並移除所有空字串
        keywords = [k.strip() for k in keywords_str.split('_') if k.strip()]
        
        # 3. 檢查關鍵詞列表是否為空
        # 如果經過清理和分隔後，keywords 列表是空的，則跳過此主題。
        if not keywords:
            # print(f"跳過主題 {topic_id}: 關鍵詞已被完全移除或為空。")
            continue

        refined_name = topic_keywords # 預設名稱為原始名稱

        # 4. 進行主題映射精煉
        for key, values in topic_mapping.items():
            # 確保 values 是可迭代的列表或集合
            if isinstance(values, str):
                values = [values]
            
            # 檢查任一關鍵詞是否存在於映射值中
            if any(k in keywords for k in values):
                refined_name = key
                break

        # 5. 加入精煉後的主題
        refined_topics[topic_id] = refined_name
        
    return refined_topics