# This file will contain the plotting functions using Plotly.
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

def plot_burstiness(sent_lengths: list):
    """
    Creates an interactive histogram of sentence lengths using Plotly.
    """
    if not sent_lengths:
        # Return an empty figure placeholder
        return px.scatter(title="句長分布 (Sentence Length Distribution)").update_layout(
            xaxis_title="句子中的詞數 (Words per Sentence)",
            yaxis_title="句子數量 (Number of Sentences)"
        )

    df = pd.DataFrame({'Sentence Length': sent_lengths})
    fig = px.histogram(
        df, 
        x='Sentence Length', 
        title='<b>句長分布 (Sentence Length Distribution)</b>',
        labels={'Sentence Length': '句子中的詞數 (Words per Sentence)'},
        marginal="box", # Add a box plot to see distribution statistics
    )
    fig.update_layout(
        yaxis_title='句子數量 (Number of Sentences)',
        bargap=0.1,
        title_x=0.5
    )
    return fig

def plot_pos_distribution(pos_dist: dict):
    """
    Creates an interactive bar chart of the Part-of-Speech distribution.
    """
    if not pos_dist:
        return px.bar(title="<b>詞性分布 (Part-of-Speech Distribution)</b>").update_layout(
            xaxis_title="詞性 (Part of Speech)",
            yaxis_title="百分比 (Percentage)"
        )

    # Sort by value for better visualization
    sorted_pos = sorted(pos_dist.items(), key=lambda x: x[1], reverse=True)
    
    df = pd.DataFrame(sorted_pos, columns=['POS', 'Percentage'])
    
    fig = px.bar(
        df,
        x='POS',
        y='Percentage',
        title='<b>詞性分布 (Part-of-Speech Distribution)</b>',
        labels={'POS': '詞性 (Part of Speech)', 'Percentage': '百分比 (%)'},
        color='POS'
    )
    fig.update_layout(
        title_x=0.5,
        showlegend=False
    )
    return fig

def plot_zipf(zipf_data: dict):
    """
    Creates an interactive log-log plot of word rank vs. frequency.
    """
    if not zipf_data:
        return go.Figure().update_layout(
            title_text="<b>Zipf's Law 分布 (Zipf's Law Distribution)</b>",
            xaxis_type="log", yaxis_type="log",
            xaxis_title="詞頻排名 (Log Rank)", yaxis_title="詞語頻率 (Log Frequency)"
        )

    ranks = zipf_data["ranks"]
    frequencies = zipf_data["frequencies"]
    words = zipf_data["words"]

    df = pd.DataFrame({
        "Rank": ranks,
        "Frequency": frequencies,
        "Word": words
    })

    # Create the scatter plot for the actual data
    fig = px.scatter(
        df,
        x="Rank",
        y="Frequency",
        log_x=True,
        log_y=True,
        title="<b>Zipf's Law 分布 (Zipf's Law Distribution)</b>",
        labels={"Rank": "詞頻排名 (Log Rank)", "Frequency": "詞語頻率 (Log Frequency)"},
        hover_data=["Word"]
    )
    fig.update_traces(marker=dict(color='blue', opacity=0.7), name="Actual Distribution")

    # Add the ideal Zipf's Law line
    # The ideal line is y = c/x, where c is the frequency of the most frequent word.
    c = frequencies[0]
    ideal_freqs = [c / r for r in ranks]
    
    fig.add_trace(go.Scatter(
        x=ranks, 
        y=ideal_freqs, 
        mode='lines', 
        name="Ideal Zipf's Law",
        line=dict(color='red', dash='dash')
    ))

    fig.update_layout(
        title_x=0.5,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig

def plot_semantic_drift(semantic_data: dict):
    """
    Creates a 2D scatter plot showing the semantic trajectory of sentences.
    """
    if not semantic_data:
        return go.Figure().update_layout(
            title_text="<b>語意軌跡 (Semantic Trajectory)</b>",
            xaxis_title="PCA Component 1", yaxis_title="PCA Component 2"
        )
    
    pca_data = semantic_data["pca_data"]
    df = pd.DataFrame({
        "x": pca_data["x"],
        "y": pca_data["y"],
        "sentence": pca_data["sentences"]
    })

    fig = go.Figure()

    # Add the trajectory line
    fig.add_trace(go.Scatter(
        x=df['x'], 
        y=df['y'],
        mode='lines',
        line=dict(width=1, color='lightgrey'),
        name='Trajectory'
    ))

    # Add the points (sentences)
    fig.add_trace(go.Scatter(
        x=df['x'], 
        y=df['y'],
        mode='markers',
        marker=dict(
            size=8,
            color=df.index, # Color by sentence order
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Sentence Order")
        ),
        hoverinfo='text',
        hovertext=df['sentence'],
        name='Sentences'
    ))

    
    return fig

def plot_perplexity(ppl_scores: list, avg_ppl: float):
    """
    Creates a line chart of perplexity scores over text chunks.
    """
    if not ppl_scores:
        return go.Figure().update_layout(
            title_text="<b>Perplexity (困惑度) 時間序列圖</b>",
            xaxis_title="文本區塊 (Text Chunk)",
            yaxis_title="Perplexity"
        )
    
    df = pd.DataFrame({'Perplexity': ppl_scores})
    df.index.name = "Text Chunk"
    
    fig = px.line(
        df,
        y='Perplexity',
        title='<b>Perplexity (困惑度) 時間序列圖</b>',
        labels={'index': '文本區塊 (Text Chunk)', 'Perplexity': 'Perplexity Score'}
    )
    
    # Add average line
    fig.add_hline(
        y=avg_ppl, 
        line_dash="dot", 
        annotation_text=f"Average: {avg_ppl:.2f}",
        annotation_position="bottom right",
        line_color='red'
    )
    
    fig.update_layout(title_x=0.5)
    
    return fig


