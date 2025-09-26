import io, base64 
from wordcloud import WordCloud

def generate_word_clouds_html(data, cultures, period_names):
    """
    生成六格文字雲，並將其嵌入單一 HTML 檔案。
    """
    html_parts = []
    
    for culture in cultures:
        for period_name in period_names:
            if culture in data and period_name in data[culture]:
                word_weights = data[culture][period_name]
                if word_weights:
                    wordcloud = WordCloud(
                        font_path="simsun.ttc",  # 假設有中文字體
                        width=800, height=400, background_color='white'
                    ).generate_from_frequencies(word_weights)
                    
                    img_buffer = io.BytesIO()
                    wordcloud.to_image().save(img_buffer, format='PNG')
                    img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                    
                    html_parts.append(f"""
                    <div style="float: left; width: 33%;">
                        <h3>{culture.upper()} {period_name}</h3>
                        <img src="data:image/png;base64,{img_str}" style="width:100%; height:auto;">
                    </div>
                    """)
    
    full_html = f"""
    <html>
    <head>
        <title>Surface Keywords Word Clouds</title>
    </head>
    <body>
        <h1>表層關鍵字文字雲</h1>
        <div>
            {''.join(html_parts)}
        </div>
    </body>
    </html>
    """
    return full_html