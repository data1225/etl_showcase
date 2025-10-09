import io, base64
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict
from bertopic import BERTopic

matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang TC', 'Noto Sans CJK TC', 'SimHei']  
matplotlib.rcParams['axes.unicode_minus'] = False 

def visualize_topics_bubble(model: BERTopic) -> go.Figure:
    return model.visualize_topics()

def visualize_hierarchical_clustering(model: BERTopic) -> go.Figure:
    return model.visualize_hierarchy()

def visualize_sankey(
    df: pd.DataFrame,                     # The DataFrame containing topic and keyword information.
    topic_column: str,                    # The name of the column containing topic IDs or labels.
    keyword_column: str,                  # The name of the column containing search keywords.
    topic_prefix: str = "Topic ",         # The prefix for topic's name on the figure.
    figure_title: str = "Sankey Diagram"  # The title shown on the figure.
) -> go.Figure:
    
    # Use all data without filtering out -1
    data = df.copy()
    
    # Counts of topic-keyword pairs
    counts = data.groupby([topic_column, keyword_column]).size().reset_index(name="count")
    
    # Build unique nodes list
    unique_topics = list(counts[topic_column].unique())
    unique_keywords = list(counts[keyword_column].unique())

    # Format the labels for the left side of the Sankey chart
    # Example: "Topic 0"
    topic_labels = [f"{topic_prefix}{t}" for t in unique_topics]

    # Create a mapping from node label to ID
    label_to_id = {label: i for i, label in enumerate(topic_labels + unique_keywords)}
    
    # Create source, target, and value lists for the Sankey diagram
    source_nodes = counts[topic_column].apply(lambda x: f"{topic_prefix}{x}").map(label_to_id).tolist()
    target_nodes = counts[keyword_column].map(label_to_id).tolist()
    values = counts["count"].tolist()
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=topic_labels + unique_keywords,
        ),
        link=dict(
            source=source_nodes,
            target=target_nodes,
            value=values,
        )
    )])
    
    fig.update_layout(
        title_text=figure_title, 
        font=dict(size=12),
        autosize=True,
    )
    
    return fig

def generate_bubble_chart_html(data, display_language: str = "en", source_description: str = ""):
    """
    生成氣泡圖，並將其嵌入 HTML 檔案。

    Args:
        data (dict): 各時期的主題資料
        display_language (str): 語言設定 ("zh"=中文, "en"=英文)
    """

    html_content = ""

    if display_language == "zh":
        report_name = "議題氣泡圖"
        xaxis_title = "議題相似度維度 1"
        yaxis_title = "議題相似度維度 2"
    else:
        report_name = "Topic Bubble Chart"
        xaxis_title = "Topic similarity dimension 1"
        yaxis_title = "Topic similarity dimension 2"

    for period, topics in data.items():
        if topics:
            df = pd.DataFrame(topics)
            df = df.dropna(subset=['x', 'y'])

            fig = px.scatter(
                df,
                x="x",
                y="y",
                size="count",
                hover_name="name",
                hover_data={"keywords": True, "count": True},
                title=f"{period} {report_name}",
                height=500,
            )
            fig.update_traces(text=df['name'], textposition='top center')
            fig.update_layout(xaxis_title=xaxis_title, yaxis_title=yaxis_title)

            # 創建主題內容比對表
            table_html = df[['id', 'name', 'keywords', 'count', 'x', 'y']].to_html(index=False)

            html_content += f"""
            <div>
                <h2>{period}</h2>
                <div style="display: flex; flex-direction: column;">
                    <div style="width: 100%;">{fig.to_html(full_html=False, include_plotlyjs='cdn')}</div>
                    <div style="width: 100%;">{table_html}</div>
                </div>
            </div>
            """

    full_html = f"""
    <html>
        <head>
            <title>{report_name}</title>
        </head>    
        <body>
            {html_content}
            <br />
            <div>{source_description}</div>
        </body>
    </html>"""
    return full_html

def generate_topic_sentiment_html(sentiment_data: Dict[str, Dict[str, float]], display_language: str = "en", source_description: str = "") -> str:
    chart_html_snippets = []

    # 多語對照表
    sentiment_map = {
        'positive': {'zh': '正面', 'en': 'Positive'},
        'neutral': {'zh': '中性', 'en': 'Neutral'},
        'negative': {'zh': '負面', 'en': 'Negative'}
    }

    labels = {
        'topic': {'zh': '議題', 'en': 'Topic'},
        'sentiment': {'zh': '情感種類', 'en': 'Sentiment Types'},
        'ratio': {'zh': '比例', 'en': 'Proportion'},
        'report_title': {'zh': '議題情感報告', 'en': 'Topic Sentiment Report'},
        'chart_title': {'zh': '{topic}', 'en': '{topic}'}
    }

    lang = display_language if display_language in ['zh','en'] else "en"

    # 準備資料
    data_list = []
    for topic_name, proportions in sentiment_data.items():
        for sentiment, proportion in proportions.items():
            data_list.append({
                labels['topic'][lang]: topic_name,
                labels['sentiment'][lang]: sentiment,
                labels['ratio'][lang]: proportion
            })
    df = pd.DataFrame(data_list)

    unique_topics = df[labels['topic'][lang]].unique()
    sentiments = ['positive', 'neutral', 'negative']
    all_combinations = pd.MultiIndex.from_product(
        [unique_topics, sentiments], names=[labels['topic'][lang], labels['sentiment'][lang]]
    ).to_frame(index=False)
    df = all_combinations.merge(df, on=[labels['topic'][lang], labels['sentiment'][lang]], how='left').fillna({labels['ratio'][lang]: 0})

    for topic_name in df[labels['topic'][lang]].unique():
        topic_df = df[df[labels['topic'][lang]] == topic_name].copy()
        topic_df[labels['sentiment'][lang]] = topic_df[labels['sentiment'][lang]].map(lambda s: sentiment_map[s][lang])

        # 顏色對應
        palette = {
            sentiment_map['positive'][lang]: '#4BC0C0',
            sentiment_map['neutral'][lang]: '#FFCE56',
            sentiment_map['negative'][lang]: '#FF6384'
        }

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(
            x=labels['ratio'][lang],
            y=labels['sentiment'][lang],
            data=topic_df,
            ax=ax,
            palette=palette
        )

        ax.set_title(labels['chart_title'][lang].format(topic=topic_name), fontsize=16)
        ax.set_xlabel(labels['ratio'][lang], fontsize=12)
        ax.set_ylabel(labels['sentiment'][lang], fontsize=12)
        ax.set_xlim(0, 1.0)
        sns.despine(top=True, right=True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)

        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        chart_html_snippets.append(img_str)

    full_html = f"""
    <!DOCTYPE html>
    <html lang="{ 'zh-TW' if lang == 'zh' else 'en' }">
    <head>
        <meta charset="UTF-8">
        <title>{labels['report_title'][lang]}</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-100 p-8">
        <h1 class="text-4xl font-bold text-center mb-8">{labels['report_title'][lang]}</h1>
        <div class="container mx-auto grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
            {''.join([f'<div class="bg-white rounded-xl shadow p-6"><img src="data:image/png;base64,{img_str}" alt="chart"></div>' for img_str in chart_html_snippets])}
        </div>
        <div class="container mx-auto grid grid-cols-1 sm:grid-cols-1 lg:grid-cols-1 pt-5">
            {source_description}
        </div>
    </body>
    </html>
    """
    return full_html
