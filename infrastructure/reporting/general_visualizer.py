import pandas as pd
import plotly.graph_objects as go

def visualize_heatmap(data, x_labels):
    """
    生成熱力圖。
    """
    df = pd.DataFrame(data).T.fillna(0)
    df = df.reindex(columns=x_labels, fill_value=0)
    
    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.columns,
        y=df.index,
        colorscale='Viridis'
    ))
    return fig

def visualize_four_quadrants_violin_plot(data):
    """
    生成小提琴圖，四象限的圖放在同一圖中。
    """
    fig = go.Figure()
    
    for period, categories in data.items():
        for category, scores in categories.items():
            fig.add_trace(go.Violin(
                y=scores,
                name=f'{period} - {category}',
                box_visible=True,
                meanline_visible=True
            ))
    return fig

def visualize_radar_chart(data):
    """
    生成雷達圖。
    """
    categories = list(data.keys())
    values = list(data.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.2 if max(values) > 0 else 1]
            )),
        showlegend=True,
    )
    return fig