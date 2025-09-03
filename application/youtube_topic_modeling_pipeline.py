#!/usr/bin/env python
# coding: utf-8

# # YouTube 熱門影片主題建模（BERTopic）Pipeline
# 
# 以中文分析為核心，使用 `shibing624/text2vec-base-chinese` 產生向量嵌入，
# 並輸出主題結構（Hierarchy / Bubble）與 Sankey（Topic ↔ Search keyword）。

# In[ ]:


from path_setup import setup_project_root
root = setup_project_root()

import os, sys, jieba
import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Local modules
from etl_showcase.infrastructure.cleaning.data_preprocessor import (
    google_sheet_to_dataframe, 
    preprocess_dataframe,
)
from etl_showcase.infrastructure.cleaning.text_cleaner import (
    remove_all_punctuation
)
from etl_showcase.infrastructure.cleaning.text_tokenizer import (
    jieba_tokenizer
)
from etl_showcase.infrastructure.reporting.topic_visualizer import (
    visualize_topics_bubble, 
    visualize_dendrogram_like_hierarchy, 
    visualize_sankey, 
    save_plotly_html,
)
from etl_showcase.config.youtube import (
    YOUTUBE_SPREADSHEET_ID,
    YOUTUBE_SEARCH_VIDEOS_FUNCTION_NAME,
    VIDEO_COLUMN_ORDER,
)
from etl_showcase.infrastructure.datasource.google_sheets_api import (
    write_secret_json,
    delete_secret_json,
    get_full_google_sheet,
)


# In[ ]:


youtube_videos = [[]]
write_secret_json()
try:
    youtube_videos = get_full_google_sheet(
        spreadsheet_id=YOUTUBE_SPREADSHEET_ID,
        sheet_name=YOUTUBE_SEARCH_VIDEOS_FUNCTION_NAME
    )
finally:
    delete_secret_json()
df_raw = google_sheet_to_dataframe(youtube_videos, VIDEO_COLUMN_ORDER)
print("Loaded rows:", len(df_raw))
df_raw.head()


# In[ ]:


# ==== 前處理 ====
# Search keyword加入主題建模，內容會過於重複
df_raw['Empty keywords'] = ''
df = preprocess_dataframe(
    df_raw, 
    target_variant="zh-TW",
    merge_fields=["Video title", "Video description", "Empty keywords"]
)
df = df.dropna(subset=["text_translated"]).reset_index(drop=True)
# 此次分析，不想留逗號、句號等有語意的符號
df['text_translated'] = df['text_translated'].apply(remove_all_punctuation)
print(df.head()['text_translated'])


# In[ ]:


from hdbscan import HDBSCAN
embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")
vectorizer = CountVectorizer(tokenizer=jieba_tokenizer, lowercase=False, min_df=2)   
data_source_text = "資料來源：蒐集 YouTube 上相關關鍵字前 300 名影片標題與描述，使用 BERTopic 生成熱門主題。"

# ==== 依 Topic 分組建模，並產生各自報表 ==== 
grouped_by_topic = df.groupby('Topic')
for topic_name, group_df in grouped_by_topic: 
    print(f"--- 處理主題: {topic_name} ---")
    
    docs = group_df["text_translated"].tolist()
    if not docs:
        print(f"主題 '{topic_name}' 沒有資料，跳過。")
        continue

    # ==== 所有圖表共通變數與前置處理 ====
    # 複製df，以避免 SettingWithCopyWarning 錯誤
    group_df = group_df[['Topic', 'Search keyword', 'merged_text', 'cleaned_text', 'text_translated']].copy()

    # 依每圈資料訓練新模型
    topic_model = BERTopic(
        embedding_model=embedding_model,
        language="chinese",
        vectorizer_model=vectorizer,
        verbose=True
    )
    topics, probs = topic_model.fit_transform(docs)
    group_df["topic_id"] = topics
    # 移除未在任何文本出現的主題
    group_df = group_df[group_df['topic_id'] > -1]

    topic_info = topic_model.get_topic_info()    
    topic_info = topic_info[topic_info['Topic'] > -1]

    docs_dir = os.path.join(os.getcwd(), '..', 'docs', topic_name)
    os.makedirs(docs_dir, exist_ok=True)
    bubble_path = os.path.join(docs_dir, "topics_bubble.html")
    hier_path = os.path.join(docs_dir, "topics_hierarchy.html")
    sankey_path = os.path.join(docs_dir, "topic_keyword_sankey.html")
    
    if len(topic_info) < 3:
        print('主題數量太少，無法繪圖')

        warning_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>資料不足</title>
            <style>
                body { font-family: sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; }
                p { font-size: 1rem; color: gray; }
            </style>
        </head>
        <body>
            <p>該議題分析出來的主題數量過少，無法視覺化。</p>
        </body>
        </html>
        """
        for path in [bubble_path, hier_path, sankey_path]:
            with open(path, "w", encoding="utf-8") as f:
                f.write(warning_html)
        print("Saved:", bubble_path, hier_path, sankey_path)
        
        continue

    # ==== Bubble 圖 ====
    fig_bubble = visualize_topics_bubble(topic_model)
    fig_bubble.update_layout(title="")
    save_plotly_html(fig_bubble, bubble_path)
    
    # 準備html中 主題ID vs 主題內容 對照表資料
    topic_table_df = pd.DataFrame({
        'Topic': topic_info['Topic'],
        'Keywords': topic_info['Name'].apply(lambda x: ' | '.join(x.split('_')[:5]))
    })
    table_data = go.Table(
        header=dict(values=["主題代號", "主題內容"],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[topic_table_df.Topic, topic_table_df.Keywords],
                    fill_color='lavender',
                    align='left')
    )
    # 準備html中 Bubble Chart 資料
    bubble_data = fig_bubble.to_json()
    # 整合並寫入新html內容
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Topic Visualization</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: sans-serif; }}
            .container {{ display: flex; width: 100%; height: 80vh; }}
            #bubble-chart {{ flex: 2; border: 1px solid #ccc; box-sizing: border-box; }}
            #topic-table {{ flex: 1; overflow-y: auto; box-sizing: border-box; }}
            .hover-cell {{ cursor: pointer; font-weight: bold; text-decoration: underline; }}
            h3.title {{ margin-block-end: .1em; font-weight: 320; }}
            p.subtitle {{ margin-block-start: 0em; margin-block-end: .4em; font-size: .95rem; font-weight: 300; }}
            p.source {{ font-size: .8rem; font-weight: 300; color: gray; }}
        </style>
    </head>
    <body>
        <h3 class="title">熱門主題聲量與相似度（Bubble Chart）</h3>
        <p class="subtitle">主題氣泡愈大，代表相關影片數量愈多；主題氣泡間愈近，代表相似性愈高。</p>
        <div class="container">
            <div id="bubble-chart"></div>
            <div id="topic-table"></div>
        </div>
        <div><p class="source">{data_source_text}</p></div>
        <script>
            document.addEventListener('DOMContentLoaded', function() {{
                var bubbleData = {bubble_data};
                var tableData = {table_data.to_json()};
                
                // Fix for topic_id being null
                for (var i = 0; i < bubbleData.data.length; i++) {{
                    if (bubbleData.data[i].topic_id === undefined) {{
                        bubbleData.data[i].topic_id = bubbleData.data[i].name;
                    }}
                }}
    
                Plotly.newPlot('bubble-chart', bubbleData.data, bubbleData.layout);
    
                var tableDiv = document.getElementById('topic-table');
                Plotly.newPlot(tableDiv, [tableData], {{}});
    
                tableDiv.on('plotly_hover', function(data) {{
                    if (data.points.length > 0) {{
                        var hoveredTopicId = data.points[0].cells.values[0][data.points[0].pointIndex];
                        var bubblePlot = document.getElementById('bubble-chart');
                        var traceIndex = bubblePlot.data.findIndex(d => d.topic_id === hoveredTopicId);
                        
                        if (traceIndex !== -1) {{
                            Plotly.restyle(bubblePlot, {{
                                'marker.line.width': [3],
                                'marker.line.color': ['black']
                            }}, [traceIndex]);
                        }}
                    }}
                }});
    
                tableDiv.on('plotly_unhover', function(data) {{
                    if (data.points.length > 0) {{
                        var unhoveredTopicId = data.points[0].cells.values[0][data.points[0].pointIndex];
                        var bubblePlot = document.getElementById('bubble-chart');
                        var traceIndex = bubblePlot.data.findIndex(d => d.topic_id === unhoveredTopicId);
                        
                        if (traceIndex !== -1) {{
                            Plotly.restyle(bubblePlot, {{
                                'marker.line.width': [1],
                                'marker.line.color': ['rgba(0,0,0,0)']
                            }}, [traceIndex]);
                        }}
                    }}
                }});
            }});
        </script>
    </body>
    </html>
    """
    with open(bubble_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    # ==== Hierarchy 圖 ====
    fig_h = visualize_dendrogram_like_hierarchy(topic_model)
    fig_h.update_layout(
        title="熱門主題層級關聯（Hierarchical Clustering）",
        annotations=[dict(
            text=data_source_text,
            x=0.5, y=-0.15,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=12, color="gray")
        )]
    )
    save_plotly_html(fig_h, hier_path)
    
    # ==== Sankey 圖 ====
    # 建立圖中主題格式: "ID_關鍵字1_關鍵字2..."
    topic_label_map = {}
    for index, row in topic_info.iterrows():
        topic_id = row['Topic']
        keywords = '_'.join(row['Name'].split('_')[1:])
        topic_label_map[topic_id] = f"{topic_id}_{keywords}"
    group_df["topic_label"] = group_df["topic_id"].map(topic_label_map)    
    fig_sk = visualize_sankey(
        group_df, 
        topic_column="topic_label", 
        keyword_column="Search keyword",
        topic_prefix="",
    )
    fig_sk.update_layout(
        title=go.layout.Title(
            text='熱門主題與搜尋關鍵字之關聯（Sankey Diagram）<br /><sub>左側為熱門主題，右側為搜尋關鍵字</sub>'
        ),
        annotations=[dict(
            text=data_source_text,
            x=0.5, y=-0.1,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=12, color="gray")
        )],
    )
    save_plotly_html(fig_sk, sankey_path)
    
    # ==== 儲存模型與標註資料 ====
    topic_model.save(os.path.join(docs_dir, "bertopic_model"), serialization="safetensors")
    group_df.to_csv(os.path.join(docs_dir, "docs_with_topics.csv"), index=False, encoding="utf-8-sig")
    
    print("Saved:", bubble_path, hier_path, sankey_path)


# In[ ]:



