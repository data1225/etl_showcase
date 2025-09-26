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

    Args:
        topic_info (pd.DataFrame): 包含 'Name' 列的 DataFrame。'Name' 的內容是由關鍵字組成的字串，例如 '政治_權力_鬥爭'。
        topic_mapping (Dict[str, str]): 精煉主題映射字典。

    Returns:
        Dict[int, str]: 包含精煉後主題名稱的字典。key 為原始主題 ID，value 為精煉後的主題名稱。
    """

    refined_topics = {}
    for row in tqdm(topic_info.itertuples(), total=len(topic_info), leave=False, desc=f"精煉主題清單"):
        topic_id = row.Topic
        topic_keywords = row.Name

        if topic_id == -1:
            continue

        keywords = topic_keywords.split('_')
        refined_name = topic_keywords

        for key, values in topic_mapping.items():
            if any(k in keywords for k in values):
                refined_name = key
                break

        refined_topics[topic_id] = refined_name
    return refined_topics