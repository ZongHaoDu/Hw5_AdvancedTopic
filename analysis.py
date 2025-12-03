import nltk
import numpy as np
import streamlit as st
from collections import Counter
import string
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import nltk.downloader

# Helper to download nltk data silently
@st.cache_resource
def download_nltk_data():
    resources = {
        "tokenizers/punkt": "punkt",
        "taggers/averaged_perceptron_tagger": "averaged_perceptron_tagger",
        "corpora/stopwords": "stopwords"
    }
    for resource_path, resource_id in resources.items():
        try:
            nltk.data.find(resource_path)
        except nltk.downloader.DownloadError:
            nltk.download(resource_id, quiet=True)

def calculate_burstiness(text: str):
    """
    Calculates the burstiness of a text, defined as the coefficient of variation of sentence lengths.
    
    Returns:
        - burstiness_score (float): The calculated burstiness.
        - sent_lengths (list): A list of sentence lengths (number of words).
    """
    download_nltk_data()
    
    sentences = nltk.sent_tokenize(text)
    
    if not sentences:
        return 0, []

    # Using word_tokenize for more accurate word counts per sentence
    sent_lengths = [len(nltk.word_tokenize(s)) for s in sentences]
    
    if not sent_lengths:
        return 0, []

    mean_length = np.mean(sent_lengths)
    std_dev = np.std(sent_lengths)
    
    burstiness_score = std_dev / mean_length if mean_length > 0 else 0
    
    return burstiness_score, sent_lengths

def calculate_stylometry(text: str):
    """
    Calculates stylometric features: Type-Token Ratio (TTR) and POS distribution.

    Returns:
        - ttr (float): The Type-Token Ratio.
        - pos_dist (dict): A dictionary with the distribution of major POS tags.
    """
    download_nltk_data()
    
    # Use word_tokenize for TTR and POS tagging
    tokens = nltk.word_tokenize(text.lower())
    
    if not tokens:
        return 0, {}

    # Calculate TTR
    ttr = len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0

    # POS Tagging and Distribution
    pos_tags = nltk.pos_tag(tokens)
    
    # Simplify tags to major categories
    pos_counts = {
        "Noun": 0, "Verb": 0, "Adjective": 0, "Adverb": 0, 
        "Pronoun": 0, "Preposition": 0, "Conjunction": 0, "Determiner": 0,
        "Other": 0
    }
    
    for _, tag in pos_tags:
        if tag.startswith('NN'):
            pos_counts["Noun"] += 1
        elif tag.startswith('VB'):
            pos_counts["Verb"] += 1
        elif tag.startswith('JJ'):
            pos_counts["Adjective"] += 1
        elif tag.startswith('RB'):
            pos_counts["Adverb"] += 1
        elif tag.startswith('PRP') or tag.startswith('WP'):
            pos_counts["Pronoun"] += 1
        elif tag.startswith('IN'):
            pos_counts["Preposition"] += 1
        elif tag.startswith('CC'):
            pos_counts["Conjunction"] += 1
        elif tag.startswith('DT') or tag.startswith('WDT'):
            pos_counts["Determiner"] += 1
        else:
            pos_counts["Other"] += 1
            
    # Convert counts to percentages
    total_tags = len(pos_tags)
    pos_dist = {k: (v / total_tags) * 100 for k, v in pos_counts.items()} if total_tags > 0 else {}
    
    return ttr, pos_dist

def calculate_zipf(text: str):
    """
    Calculates word frequency distribution for Zipf's Law analysis.
    
    Returns:
        - A dictionary containing ranks, frequencies, and words.
    """
    download_nltk_data()
    
    # Tokenize, remove punctuation and stopwords
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(nltk.corpus.stopwords.words('english'))
    punct = set(string.punctuation)
    
    clean_tokens = [
        token for token in tokens 
        if token.isalpha() and token not in stop_words and token not in punct
    ]
    
    if not clean_tokens:
        return None

    # Calculate frequencies
    freqs = Counter(clean_tokens)
    sorted_freqs = freqs.most_common()
    
    ranks = list(range(1, len(sorted_freqs) + 1))
    frequencies = [count for _, count in sorted_freqs]
    words = [word for word, _ in sorted_freqs]
    
    return {"ranks": ranks, "frequencies": frequencies, "words": words}

# --- Semantic Drift ---
@st.cache_resource
def load_embedding_model():
    """Loads the sentence-transformer model and caches it."""
    return SentenceTransformer('all-MiniLM-L6-v2')

def calculate_semantic_drift(text: str):
    """
    Calculates semantic drift and variance using sentence embeddings.

    Returns:
        - A dictionary containing avg_drift, variance, and pca_data.
    """
    download_nltk_data()
    model = load_embedding_model()
    
    sentences = nltk.sent_tokenize(text)
    
    if len(sentences) < 2:
        return None

    # Generate embeddings
    embeddings = model.encode(sentences)
    
    # Calculate drift (distance between adjacent sentences)
    drifts = [
        cosine_distances([embeddings[i]], [embeddings[i+1]])[0][0] 
        for i in range(len(embeddings) - 1)
    ]
    avg_drift = np.mean(drifts) if drifts else 0
    
    # Calculate overall variance of embeddings
    variance = np.mean(np.var(embeddings, axis=0))
    
    # Reduce to 2D with PCA for plotting
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)
    
    pca_data = {
        "x": pca_result[:, 0],
        "y": pca_result[:, 1],
        "sentences": sentences
    }
    
    return {
        "avg_drift": avg_drift,
        "variance": variance,
        "pca_data": pca_data
    }

# --- Perplexity ---
@st.cache_resource
def load_perplexity_model():
    """Loads the GPT-2 model and tokenizer for perplexity calculation."""
    return GPT2LMHeadModel.from_pretrained('gpt2'), GPT2TokenizerFast.from_pretrained('gpt2')

def calculate_perplexity(text: str):
    """
    Calculates perplexity of a text using a sliding window approach with GPT-2.
    """
    model, tokenizer = load_perplexity_model()
    
    encodings = tokenizer(text, return_tensors='pt')
    
    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
            
    ppl_scores = [torch.exp(nll) for nll in nlls]
    avg_ppl = torch.exp(torch.stack(nlls).mean()).item() if nlls else 0
    
    # Convert tensors to numbers for plotting
    ppl_scores_float = [score.item() for score in ppl_scores]

    return avg_ppl, ppl_scores_float

# --- Final Score Calculation ---
def calculate_final_score(metrics: dict):
    """
    Calculates a heuristic 'AI Likelihood' score based on all metrics.
    This is a simplified model and can be expanded.
    """
    # Normalize scores (0-1, where 1 is more "AI-like")
    # These are heuristics and can be fine-tuned.
    
    # Perplexity: Lower is more AI-like. Assume avg human PPL is ~60, AI is ~30.
    ppl_score = 1 - min(metrics.get('avg_perplexity', 60) / 60, 1.0)
    
    # Burstiness: Lower is more AI-like. Assume human B is ~0.8, AI is ~0.5.
    burstiness_score = 1 - min(metrics.get('burstiness', 0.5) / 0.8, 1.0)
    
    # TTR: Very low or very high can be AI. We'll simplify: lower is more AI.
    ttr_score = 1 - min(metrics.get('ttr', 0.5) / 0.5, 1.0)
    
    # Semantic Drift: Lower is more AI-like. Assume human drift is ~0.4, AI is ~0.2.
    drift_score = 1 - min(metrics.get('avg_drift', 0.2) / 0.4, 1.0)
    
    # Weights for each metric
    weights = {
        'ppl': 0.4,
        'burstiness': 0.2,
        'ttr': 0.2,
        'drift': 0.2
    }
    
    final_score = (
        ppl_score * weights['ppl'] +
        burstiness_score * weights['burstiness'] +
        ttr_score * weights['ttr'] +
        drift_score * weights['drift']
    )
    
    return final_score * 100 # Return as a percentage