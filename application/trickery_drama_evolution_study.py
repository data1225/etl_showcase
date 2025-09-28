#!/usr/bin/env python
# coding: utf-8

# In[3]:


from path_setup import setup_project_root
root = setup_project_root()

import openpyxl, os, json, inspect, hdbscan, warnings, torch
import pandas as pd
from tqdm.notebook import tqdm
from umap import UMAP
from collections import defaultdict
from typing import Dict
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Local modules
from etl_showcase.infrastructure.utils.file_utils import (
    save_large_dataframe_to_excel,
    read_and_combine_excel_sheets,
)
from etl_showcase.infrastructure.cleaning.language_translator import LanguageTranslator
from etl_showcase.infrastructure.cleaning.text_tokenizer import (
    jieba_tokenizer
)
from etl_showcase.infrastructure.cleaning.text_cleaner import (
    clean_text,
    remove_non_chinese_and_noise,
    remove_non_english_and_noise,
    remove_all_punctuation,
)
from etl_showcase.infrastructure.nlp.sentiment_analyzer import SentimentAnalyzer
from etl_showcase.infrastructure.nlp.topic_utils import (
    default_n_neighbors,
    get_topic_coordinates,
    transform_to_refined_topics_by_culture,
)
from etl_showcase.infrastructure.reporting.html_export import (
    save_plotly_html,
    save_html,
)
from etl_showcase.infrastructure.reporting.general_visualizer import (
    visualize_heatmap,
    visualize_four_quadrants_violin_plot,
    visualize_radar_chart,
)
from etl_showcase.infrastructure.reporting.topic_visualizer import (
    generate_bubble_chart_html,
    generate_topic_sentiment_html,
)
from etl_showcase.infrastructure.reporting.wordcloud_visualizer import generate_word_clouds_html

def get_BERTopic_model(culture:str):
    umap_model = UMAP(
        n_neighbors=default_n_neighbors(), 
        n_components=2, 
        min_dist=0.0, 
        metric='cosine', 
        random_state=42
    )
    
    if culture == 'zh':
        embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")
        vectorizer_model = CountVectorizer(tokenizer=jieba_tokenizer, min_df=5, max_df=0.85)   
        return BERTopic(
            embedding_model=embedding_model,
            language="chinese",
            vectorizer_model=vectorizer_model,
            umap_model=umap_model,
            verbose=True,
        )
    else:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        vectorizer_model = CountVectorizer(stop_words='english', min_df=5, max_df=0.85)
        return BERTopic(
            embedding_model=embedding_model,
            language="english",
            vectorizer_model=vectorizer_model,
            umap_model=umap_model,
            verbose=True,
        )



# 資料清理完，偶爾還是會有奇怪符號或非純中文、英文的文本，但最後主題建模出來的主題。
# 只有極少數會出現非純中文、英文的主題，考慮每個任務的CP值，該程式暫時先優化至此。
def clean_and_translate(translator:LanguageTranslator, text: str, target_language: str):
    detect_lang = translator.detect_lang(text)
    if detect_lang is None:
        return None  

    if target_language == "zh":
        target_language = "zh-TW"
    elif target_language == "en":
        target_language = "en"
    else:
        target_language = "auto"
        
    if detect_lang.startswith('zh'):
        detect_lang = "zh-TW"
        text = remove_non_chinese_and_noise(text)
    elif detect_lang.startswith('en'):
        detect_lang = "zh-TW"
        text = remove_non_english_and_noise(text)
    else:
        detect_lang = "auto"
        text = remove_all_punctuation(text)        
        
    if not detect_lang.startswith(target_language):
        text = translator.translate(text=text, source_language=detect_lang, target_language=target_language)
    if target_language.startswith('zh'):
        text = translator.to_target_chinese_variant(
            text=text, 
            target_variant="zh-TW", 
        )

    return text

def load_data(file_path: str) -> (pd.DataFrame, Dict[str, pd.DataFrame]):
    """
    載入主工作表和所有留言工作表的資料，並根據 Screenwork 欄位進行分組讀取。

    Args:
        file_path (str): Excel 檔案的路徑。

    Returns:
        (pd.DataFrame, Dict[str, pd.DataFrame]):
        - main_df: 主工作表 '男頻高流量權謀爽劇影評影片資料' 的 DataFrame。
        - comments_data: 字典，鍵為 Screenwork 名稱，值為對應的留言 DataFrame。
    """
    xls = pd.ExcelFile(file_path)
    main_df = pd.read_excel(xls, '男頻高流量權謀爽劇影評影片資料')
    
    comments_data = {}
    
    # 從 main_df 讀取所有不重複的 Screenwork 名稱
    if 'Screenwork' in main_df.columns:
        screenworks = main_df['Screenwork'].unique()
    else:
        print("主工作表中找不到 'Screenwork' 欄位。")
        return main_df, {}

    # 根據 Screenwork 名稱尋找並讀取留言工作表
    matching_sheets = []
    for screenwork_name in screenworks:
        for sheet_name in xls.sheet_names:
            if screenwork_name in sheet_name:
                matching_sheets.append(sheet_name)
        
    # 遍歷符合條件的工作表並讀取資料
    for sheet_name in matching_sheets:
        # 提取劇作標題，並替換 '留言' 和 'reviews'
        drama_title = sheet_name.replace('留言', '').replace('reviews', '').strip()
        
        # 將工作表資料存入字典，以處理同一個劇作有多個留言工作表的情況
        if drama_title not in comments_data:
            comments_data[drama_title] = pd.read_excel(xls, sheet_name)
        else:
            # 如果該劇作已存在，將新的留言資料附加到現有的 DataFrame
            new_data = pd.read_excel(xls, sheet_name)
            comments_data[drama_title] = pd.concat([comments_data[drama_title], new_data], ignore_index=True)
                
    return main_df, comments_data 

def preprocess_and_merge_data(main_df: pd.DataFrame, comments_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    預處理資料並將影片內容和留言合併。
    """
    print('Start to preprocess and merge data')
    
    # 設定資料上限（每個文化領域的上限）
    MAX_RECORDS = 5000

    # -------------------
    # 第一步：預估每個文化領域的總資料量
    # -------------------
    total_likes_by_culture = {}
    
    # 預估影片資料的總按讚數
    main_df_likes = main_df.groupby('Cultural sphere')['Like count'].sum()
    for culture, likes in main_df_likes.items():
        total_likes_by_culture[culture] = total_likes_by_culture.get(culture, 0) + likes

    # 預估留言資料的總按讚數
    for screenwork, comments_df in comments_data.items():
        if not comments_df.empty:
            comments_df_likes = comments_df.groupby('Video ID')['Like count'].sum()
            for video_id, likes in comments_df_likes.items():
                # 找到對應的影片，取得其文化領域
                cultural_sphere = main_df[main_df['Video ID'] == video_id]['Cultural sphere'].iloc[0]
                total_likes_by_culture[cultural_sphere] = total_likes_by_culture.get(cultural_sphere, 0) + likes

    print(f"預估每個文化領域的總資料量 (按讚數總和): {total_likes_by_culture}")
    
    # -------------------
    # 第二步：根據總量限制調整每個文化領域的重複次數
    # -------------------
    scaling_factors = {}
    for culture, total_likes in total_likes_by_culture.items():
        if total_likes > MAX_RECORDS:
            scaling_factor = MAX_RECORDS / total_likes
            print(f"文化領域 '{culture}' 總量超過 {MAX_RECORDS} 筆，將使用縮放比例: {scaling_factor:.4f}")
        else:
            scaling_factor = 1
            print(f"文化領域 '{culture}' 總量未超過上限，不需要縮放。")
        scaling_factors[culture] = scaling_factor
    
    # -------------------
    # 第三步：按比例分配與合併
    # -------------------
    translator = LanguageTranslator()
    merged_data = []

    for _, row in tqdm(main_df.iterrows(), total=len(main_df), desc="處理影片資料"):
        video_id = row['Video ID']
        cultural_sphere = row['Cultural sphere']
        screenwork = row['Screenwork']
        scaling_factor = scaling_factors.get(cultural_sphere, 1) # 取得對應的縮放比例
        doc = {
            'Video ID': video_id,
            'Publish year': row['Publish year'],
            'Cultural sphere': cultural_sphere,
            'Category': row['Category'],
            'Screenwork': screenwork,
            'text': '',
        }
        
        ### 加入影片文本
        video_title = str(row['Video title']) if not pd.isna(row['Video title']) else ''
        video_description = str(row['Video description']) if not pd.isna(row['Video description']) else ''
        video_content = str(row['Video content']) if not pd.isna(row['Video content']) else ''
        video_content = video_content.replace('\n', ' ').replace('\r', '').replace('\t', '')
        video_text = video_title + ' ' + video_description + ' ' + video_content
        
        video_text = clean_and_translate(
            translator=translator,
            text=video_text,
            target_language=cultural_sphere
        )
        if video_text is not None and video_text != '':
            video_doc = doc.copy()
            video_doc['text'] = video_text
            
            # 使用四捨五入計算重複次數
            reps = round(row['Like count'] * scaling_factor)
            merged_data.extend([video_doc] * reps)
            
        ### 加入留言文本
        comments_df = comments_data.get(screenwork, pd.DataFrame())
        if comments_df.empty:
            continue
        
        for _, comment_row in tqdm(comments_df.iterrows(), total=len(comments_df), leave=False, desc=f"處理 {screenwork} 留言"):
            if comment_row['Video ID'] == video_id:
                comment_text = str(comment_row['Text']) if not pd.isna(str(comment_row['Text'])) else ''
                comment_text = clean_and_translate(
                    translator=translator,
                    text=comment_text,
                    target_language=cultural_sphere
                )
                if comment_text is not None and comment_text != '':
                    comment_doc = doc.copy()
                    comment_doc['text'] = comment_text

                    # 使用四捨五入計算重複次數
                    reps = round(comment_row['Like count'] * scaling_factor)
                    merged_data.extend([comment_doc] * reps)

    print(f"最終資料集大小: {len(merged_data)} 筆")
    return pd.DataFrame(merged_data)


# In[4]:


# 撈取或宣告所需資料
# 影評影片內容及留言
data_file_path = "./data/raw/trickery_drama_evolution_study_data.xls"
data_file_path_preprocessed = './data/processed/trickery_drama_evolution_study_data.xlsx'
if not os.path.exists(data_file_path):
    print(f"Error: Data file not found at {data_file_path}")
    sys.exit()        
main_df, comments_data = load_data(data_file_path)
# 共用變數
time_periods = {
    '2015–2020': (2015, 2020),
    '2021–2023': (2021, 2023),
    '2024–2025': (2024, 2025)
}
cultures = ['zh', 'en']
docs_dir = os.path.join(os.getcwd(), '..', 'docs/trickery_drama_evolution_study')
os.makedirs(docs_dir, exist_ok=True)
# 常見權謀元素 mapping 清單
with open('../resources/trickery_element_mapping.json', 'r', encoding='utf-8') as f:
    topic_mapping_list = json.load(f)

if os.path.exists(data_file_path_preprocessed):
    all_data_df = read_and_combine_excel_sheets(data_file_path_preprocessed)
else:
    # 將所有影片資料和留言資料合併並進行預處理
    all_data_df = preprocess_and_merge_data(main_df, comments_data)
    # 將資料寫入csv 避免一直重複預處理大量文本
    save_large_dataframe_to_excel(all_data_df, data_file_path_preprocessed)

print(f'zh資料 {len(all_data_df[all_data_df['Cultural sphere'] == 'zh'])} 筆')
print(f'en資料 {len(all_data_df[all_data_df['Cultural sphere'] == 'en'])} 筆')


# In[31]:


def generate_surface_keywords_report(docs_dir, data, cultures, time_periods):
    """
    根據文化區分，生成表層關鍵字文字雲報告，並移除停用詞及在過多文本中出現的詞語。
    """
    print("Generating surface keywords report...")
    word_cloud_data = defaultdict(dict)

    # 定義常用的中文停用詞，把作品名也加進去，以避免噪音，
    chinese_stop_words = [
        "琅琊榜","琅","琊","榜","慶餘年","贅婿","雪中悍刀行","藏海傳","海傳",
        "的","了","著","之","地","得","和","與","或","但是","而且","所以",
        "是","在","我","他","她","你","它","們","這","那","個","不","也","都","對","而","但",
        "因為","雖然","然後","從","到","由","於","為","以","上","下","中","前","後",
        "哪","哪裡","一","二","三","些","某些","每","全部",
        "啊","真的","很","非常","較","稍","只","還","有","沒有",
        "吧","呢","嘛","呀","看","就是","就","這個","來","把","讓","被","給","將","其","則",
        "那樣","這樣","那麼","什麼","還是","已經","能","又","要","嗎","還有","說","沒",
        "說","是","不是","這是","為","追","一部"
    ]

    # 確保 'text' 欄位沒有 NaN 值
    data['text'] = data['text'].fillna('')
    
    for culture in cultures:
        for period_name, (start, end) in time_periods.items():
            df_filtered = data[
                (data['Cultural sphere'] == culture) &
                (data['Publish year'] >= start) &
                (data['Publish year'] <= end)
            ]
            
            if not df_filtered.empty:
                corpus = df_filtered['text'].tolist()
                
                if culture == 'zh':
                    vectorizer = TfidfVectorizer(
                        max_features=100,
                        tokenizer=jieba_tokenizer,
                        stop_words=chinese_stop_words,
                        max_df=0.95
                    )
                else:
                    vectorizer = TfidfVectorizer(
                        max_features=100,
                        stop_words='english',
                        max_df=0.95
                    )
                
                try:
                    tfidf_matrix = vectorizer.fit_transform(corpus)
                    feature_names = vectorizer.get_feature_names_out()
                    
                    # 計算每個關鍵字的總 TF-IDF 權重
                    word_weights = {}
                    for col in range(len(feature_names)):
                        word_weights[feature_names[col]] = tfidf_matrix[:, col].sum()
                    
                    word_cloud_data[culture][period_name] = word_weights
                except ValueError as e:
                    print(f"Error processing {culture} - {period_name}: {e}")
                    continue

    # 生成文字雲並儲存為 HTML
    html_content = generate_word_clouds_html(word_cloud_data, cultures, time_periods.keys())
    output_path = os.path.join(docs_dir, "timing_comparison/surface_keywords/all_generation_cultural.html")
    save_html(html_content, output_path)
    print(f"Surface keywords report is saved.")

generate_surface_keywords_report(docs_dir, all_data_df, cultures, time_periods)


# In[37]:


def generate_deep_topics_report(docs_dir, data, topic_mapping_list, cultures, time_periods):
    """
    生成深層權謀元素氣泡圖報告。
    """
    print("Generating deep topics report...")
    
    # 在一開始篩選並儲存所有需要的 DataFrame
    filtered_dataframes = {}
    for culture in cultures:
        for period_name, (start, end) in time_periods.items():
            df_filtered_copy = data[
                (data['Cultural sphere'] == culture) &
                (data['Publish year'] >= start) &
                (data['Publish year'] <= end)
            ].copy()
            filtered_dataframes[(culture, period_name)] = df_filtered_copy
            
    for culture in cultures:
        print(f"Processing BERTopic for {culture}...")
        
        # 將每個時段的資料合併後再建模
        all_texts = []
        for period_name, (start, end) in time_periods.items():
            df_filtered = filtered_dataframes.get((culture, period_name))
            if df_filtered is not None:
                all_texts.extend(df_filtered['text'].tolist())

        if not all_texts:
            print(f"No data for {culture}. Skipping.")
            continue
        # 檢查文件數量是否足夠， UMAP 降維時樣本數量需大於預設值，否則會報錯
        if len(all_texts) < default_n_neighbors():
            warnings.warn(f"BERTopic 建模所需文件數不足 (至少{default_n_neighbors()}個)，目前只有 {len(all_texts)} 個。已跳過 {culture} 的主題報告生成。", UserWarning)
            continue
        
        # 訓練模型，並從所有文本中萃取出主題集合。
        topic_model = get_BERTopic_model(culture=culture)
        _, _ = topic_model.fit_transform(all_texts)
        # 從模型中取得主題的二維座標。
        topic_coords = get_topic_coordinates(topic_model)
        
        # 精煉主題清單
        topic_info = topic_model.get_topic_info()
        refined_topics = transform_to_refined_topics_by_culture(topic_info, topic_mapping_list, culture)

        # 將主題分組
        bubble_chart_data = defaultdict(list)
        for period_name, (start, end) in time_periods.items():
            df_filtered = filtered_dataframes.get((culture, period_name))
            if df_filtered is None or df_filtered.empty:
                continue

            texts_in_period = df_filtered['text'].tolist()
            topics_in_period, _ = topic_model.transform(texts_in_period)
            
            topic_counts = pd.Series(topics_in_period).value_counts().to_dict()
            
            period_data = []
            for topic_id, count in topic_counts.items():
                if topic_id == -1:
                    continue
                topic_name = refined_topics.get(topic_id, f"Topic {topic_id}")
                keywords = [str(item[0]) for item in topic_model.get_topic(topic_id)] 
                x_coord, y_coord = topic_coords.get(topic_id, (None, None))
    
                period_data.append({
                    'id': topic_id,
                    'name': topic_name,
                    'count': count,
                    'keywords': ', '.join(keywords) ,
                    'x': x_coord, 
                    'y': y_coord,
                })
            bubble_chart_data[period_name] = period_data

        html_content = generate_bubble_chart_html(bubble_chart_data)
        output_path = os.path.join(docs_dir, f"timing_comparison/topics_bubble/{culture}/all_generation.html")
        save_html(html_content, output_path)
        print(f"Deep topics report is saved.")

generate_deep_topics_report(docs_dir, all_data_df, topic_mapping_list, cultures, time_periods)


# In[39]:


def generate_sentiment_reports(docs_dir, data, topic_mapping, cultures, time_periods):
    print("Generating sentiment reports...")

    analyzer = SentimentAnalyzer()

    filtered_dataframes = {}
    for culture in cultures:
        for period_name, (start, end) in time_periods.items():
            df_filtered_copy = data[
                (data['Cultural sphere'] == culture) &
                (data['Publish year'] >= start) &
                (data['Publish year'] <= end)
            ].copy()
            filtered_dataframes[(culture, period_name)] = df_filtered_copy

    for culture in cultures:
        for period_name, (start, end) in time_periods.items():
            print(f"Processing BERTopic for {culture} - {period_name}...")
            
            df_filtered_period = filtered_dataframes.get((culture, period_name))
            if df_filtered_period is None or df_filtered_period.empty:
                print(f"No data for {culture} in {period_name}. Skipping.")
                continue

            texts_for_period = df_filtered_period['text'].tolist()
            if len(texts_for_period) < default_n_neighbors():
                print(f"BERTopic 建模所需文件數不足 (至少{default_n_neighbors()}個)，目前只有 {len(texts_for_period)} 個。已跳過 {culture} {period_name} 的主題報告生成。")
                continue

            # 建立 BERTopic
            topic_model = get_BERTopic_model(culture=culture)
            topics_in_period, _ = topic_model.fit_transform(texts_for_period)
            
            # 取得主題資訊
            topic_info = topic_model.get_topic_info()
            topic_id_to_name = {row['Topic']: row['Name'] for _, row in topic_info.iterrows()}

            # 取前 20 個主題 (不包含 -1)
            sorted_topic_info = topic_info.sort_values('Count', ascending=False)
            top_20_topics = sorted_topic_info[sorted_topic_info['Topic'] != -1].head(20)['Topic'].tolist()
            
            # 根據主題名稱分組文本
            topic_texts = defaultdict(list)
            for topic_id, text in zip(topics_in_period, texts_for_period):
                if topic_id in top_20_topics:
                    topic_name = topic_id_to_name.get(topic_id, f"Topic {topic_id}")
                    topic_texts[topic_name].append(text)
            
            if not topic_texts:
                print(f"No topics found for {culture} in {period_name}. Skipping.")
                continue

            # 這裡可以用 analyzer 計算情感
            period_sentiment_data = analyzer.analyze_sentiment_by_topic(
                topic_data=topic_texts,
            )
            html_content = generate_topic_sentiment_html(
                sentiment_data=period_sentiment_data,
                display_language='zh',
            )
            output_path = os.path.join(docs_dir, f"timing_comparison/topic_sentiment/{culture}/{period_name.replace('–', '-')}.html")
            save_html(html_content, output_path)
            print(f"Sentiment report for {period_name}  is saved.")

generate_sentiment_reports(docs_dir, all_data_df, topic_mapping_list, cultures, time_periods)


# In[41]:


def generate_drama_category_reports(docs_dir, data, topic_mapping_list, cultures, time_periods):
    """
    生成復仇劇 vs 一般權謀爽劇比對報告。
    """
    print("Generating drama category comparison reports...")
        
    analyzer = SentimentAnalyzer()

    for culture in cultures:
        print(f"Processing drama category reports for {culture}...")

        # 共用變數
        trickery_elements = set[str]
        # 熱力圖資料
        heatmap_data = defaultdict(dict)
        # 小提琴圖資料
        violin_plot_data = defaultdict(dict)

        for period_name, (start, end) in time_periods.items():
            for category in ['復仇劇', '一般權謀劇']:
                df_filtered = data[
                    (data['Cultural sphere'] == culture) &
                    (data['Publish year'] >= start) &
                    (data['Publish year'] <= end) &
                    (data['Category'] == category)
                ]
                
                if df_filtered.empty:
                    print(f"No data for {culture} in {period_name} and {category}. Skipping.")
                    continue
                
                texts_in_period = df_filtered['text'].tolist()
                # 檢查文件數量是否足夠， UMAP 降維時樣本數量需大於預設值，否則會報錯
                if len(texts_in_period) < default_n_neighbors():
                    print(f"BERTopic 建模所需文件數不足 (至少{default_n_neighbors()}個)，目前只有 {len(texts_in_period)} 個。已跳過 [{culture}][{period_name}][{category}] 的主題報告生成。")
                    continue       
                    
                # BERTopic for Heatmap
                topic_model = get_BERTopic_model(culture=culture)
                topics, _ = topic_model.fit_transform(texts_in_period)

                # 精鍊主題清單
                topic_info = topic_model.get_topic_info()
                refined_topics = transform_to_refined_topics_by_culture(topic_info, topic_mapping_list, culture)
                trickery_elements = trickery_elements.union(set(refined_topics.values()))

                # 計算每部作品中權謀元素的比重
                drama_names = df_filtered['Screenwork'].unique()
                for drama_name in tqdm(drama_names, total=len(drama_names), leave=False, desc=f"計算每部作品中權謀元素的比重"):
                    drama_texts = df_filtered[df_filtered['Screenwork'] == drama_name]['text'].tolist()
                    drama_topics, _ = topic_model.transform(drama_texts)
                    
                    topic_counts = pd.Series(drama_topics).value_counts().to_dict()
                    total_count = sum(topic_counts.values())
                    
                    drama_topic_weights = {}
                    for topic_id, count in topic_counts.items():
                        if topic_id == -1: continue
                        topic_name = refined_topics.get(topic_id, f"Topic {topic_id}")
                        drama_topic_weights[topic_name] = count / total_count
                    
                    heatmap_data[drama_name].update(drama_topic_weights)
                    
                # Sentiment analysis for Violin Plot
                sentiment_results = analyzer.analyze_sentiment(texts_in_period)
                sentiment_scores = [res['score'] if res['label'] == 'positive' else -res['score'] for res in sentiment_results]
                
                violin_plot_data[period_name][category] = sentiment_scores
        
        # 生成熱力圖
        if heatmap_data: 
            topic_mapping = topic_mapping_list.get(culture)
            fig_heatmap = visualize_heatmap(heatmap_data, trickery_elements)
            fig_heatmap.update_layout(title_text='權謀元素熱力圖', xaxis_title='權謀元素名稱', yaxis_title='作品名稱')
            output_path = os.path.join(docs_dir, f"drama_category_comparison/topic_heatmap/{culture}/all_generation.html")
            save_plotly_html(fig_heatmap,output_path)
            print(f"Topic heatmap report is saved.")

        # 生成小提琴圖
        if violin_plot_data:
            fig_violin_plot = visualize_four_quadrants_violin_plot(violin_plot_data)
            fig_violin_plot.update_layout(
                title_text='復仇劇與一般權謀劇情緒波動',
                yaxis_title='情緒分數',
                violinmode='group'
            )
            output_path = os.path.join(docs_dir, f"drama_category_comparison/violin_plot/{culture}/all_generation.html")
            save_plotly_html(fig_violin_plot,output_path)
            print(f"Violin plot report is saved.")

generate_drama_category_reports(docs_dir, all_data_df, topic_mapping_list, cultures, time_periods)


# In[7]:


def generate_per_drama_reports(docs_dir, data, topic_mapping_list, cultures):
    """
    生成各劇雷達圖報告。
    """
    print("Generating per-drama radar chart reports...")
    
    for culture in cultures:
        df_filtered = data[data['Cultural sphere'] == culture]
        dramas = df_filtered['Screenwork'].unique()
        texts_culture = df_filtered['text'].tolist()
        # 檢查文件數量是否足夠， UMAP 降維時樣本數量需大於預設值，否則會報錯
        if len(texts_culture) < default_n_neighbors():
            print(f"BERTopic 建模所需文件數不足 (至少{default_n_neighbors()}個)，目前只有 {len(texts_culture)} 個。已跳過 {culture} 的主題報告生成。")
            continue        
        topic_model = get_BERTopic_model(culture=culture)
        _ = topic_model.fit_transform(texts_culture)
        
        # 精鍊主題清單
        topic_info = topic_model.get_topic_info()
        refined_topics = transform_to_refined_topics_by_culture(topic_info, topic_mapping_list, culture)
        
        for drama_name in dramas:
            print(f"Processing radar chart for {drama_name}...")
            df_filtered = data[
                (data['Cultural sphere'] == culture) &
                (data['Screenwork'] == drama_name)
            ]
            
            if df_filtered.empty:
                continue

            drama_texts = df_filtered['text'].tolist()
            drama_topics, _ = topic_model.transform(drama_texts)

            # 根據文本數排序，選擇前10個主題，再計算選出來的主題的總文本數 
            top_num = 10
            topic_counts = pd.Series(drama_topics).value_counts()         
            top_topics = topic_counts[topic_counts.index != -1].head(top_num)
            total_count = top_topics.sum()

            topic_proportions = defaultdict(float)
            for topic_id, count in top_topics.items():
                refined_name = refined_topics.get(topic_id, f"Topic {topic_id}")
                topic_proportions[refined_name] += count / total_count

            fig_radar = visualize_radar_chart(topic_proportions)
            fig_radar.update_layout(
                title=f'{drama_name} Top {top_num} 權謀元素'
            )          
            safe_drama_name = drama_name.replace(' ', '_').replace('?', '').replace(':', '_')
            output_path = os.path.join(docs_dir, f"drama_analysis/radar_chart/{culture}/{safe_drama_name}.html")
            save_plotly_html(fig_radar,output_path)
            print(f"Radar chart for {drama_name} is saved.")
generate_per_drama_reports(docs_dir, all_data_df, topic_mapping_list, cultures)


# In[ ]:



