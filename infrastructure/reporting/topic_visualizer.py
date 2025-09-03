
from typing import List, Optional
import pandas as pd
import plotly.graph_objects as go
from bertopic import BERTopic

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

def save_plotly_html(fig, path: str):
    fig.write_html(path, include_plotlyjs="cdn")
    return path
