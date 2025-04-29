from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import pandas as pd
import numpy as np

def build_recommender(df):
    """
    Builds a recommendation engine based on content features.
    
    Args:
        df (pd.DataFrame): The dataset containing entertainment content
        
    Returns:
        tuple: (cosine_similarity_matrix, indices)
    """
    # Reset index to ensure continuous indices
    df = df.reset_index(drop=True)
    
    # Print debug information
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {df.columns.tolist()}")
    
    # Create a more comprehensive content feature by combining multiple columns
    content_features = []
    
    # Handle genres column - use the original 'genres' column
    df['content'] = df['genres'].fillna('')
    
    # Add platform information if available
    if 'platform' in df.columns:
        df['platform'] = df['platform'].fillna('')
        content_features.append('platform')
    
    # Add other features with different weights if available
    for feature in content_features:
        df['content'] = df['content'] + ' ' + df[feature]
    
    # Create TF-IDF matrix with n-grams for better feature extraction
    tfidf = TfidfVectorizer(
        stop_words='english',
        min_df=2,                 # Ignore terms that appear in less than 2 documents
        max_df=0.85,              # Ignore terms that appear in more than 85% of documents
        ngram_range=(1, 2),       # Include both unigrams and bigrams
        max_features=10000        # Limit the number of features to reduce memory usage
    )
    
    # Transform the content feature into TF-IDF matrix
    tfidf_matrix = tfidf.fit_transform(df['content'])
    
    # Compute cosine similarity matrix in batches to save memory
    batch_size = 1000
    n_items = tfidf_matrix.shape[0]
    cosine_sim = np.zeros((n_items, n_items))
    
    for i in range(0, n_items, batch_size):
        end = min(i + batch_size, n_items)
        batch_sim = linear_kernel(tfidf_matrix[i:end], tfidf_matrix)
        cosine_sim[i:end] = batch_sim
    
    # Create a Series with title indices for quick lookup
    # Convert titles to lowercase for case-insensitive matching
    # Handle duplicate titles by keeping the first occurrence
    indices = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()
    
    # Print debug information
    print(f"Number of unique titles: {len(indices)}")
    print(f"Cosine similarity matrix shape: {cosine_sim.shape}")
    
    return cosine_sim, indices

def recommend_content(title, df, cosine_sim, indices, top_n=10):
    """
    Recommends content similar to the provided title.
    
    Args:
        title (str): The title to find recommendations for
        df (pd.DataFrame): The dataset containing entertainment content
        cosine_sim (np.ndarray): The cosine similarity matrix
        indices (pd.Series): The mapping of titles to indices
        top_n (int): Number of recommendations to return
        
    Returns:
        tuple: (original_content, recommendations)
    """
    # Convert the title to lowercase for case-insensitive matching
    title = title.lower()
    
    # Print debug information
    print(f"Looking for title: {title}")
    print(f"Available titles: {len(indices)}")
    print(f"DataFrame shape: {df.shape}")
    print(f"Cosine similarity matrix shape: {cosine_sim.shape}")
    print(f"DataFrame columns: {df.columns.tolist()}")
    
    # Check if the title exists in our dataset
    if title not in indices:
        return None, None
    
    # Get the index of the title
    idx = indices[title]
    
    # Print debug information
    print(f"Found index: {idx}")
    
    # Get similarity scores for all titles
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort titles based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: float(x[1]), reverse=True)
    
    # Get top N most similar titles (excluding the input title itself)
    sim_scores = sim_scores[1:top_n + 1]
    
    # Get indices of recommended titles
    recommended_indices = [i[0] for i in sim_scores]
    
    # Print debug information
    print(f"Recommended indices: {recommended_indices}")
    
    # Get similarity scores
    similarity_scores = [float(i[1]) for i in sim_scores]
    
    # Create recommendations dataframe with similarity score
    # First, get all available columns
    available_columns = df.columns.tolist()
    print(f"Available columns: {available_columns}")
    
    # Define required columns
    required_columns = ['title', 'genres', 'rating']
    
    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in available_columns]
    if missing_columns:
        print(f"Warning: Missing required columns: {missing_columns}")
        # Use only available columns
        columns_to_include = [col for col in required_columns if col in available_columns]
    else:
        columns_to_include = required_columns
    
    # Add optional columns if available
    optional_columns = ['year', 'type', 'platform']
    for col in optional_columns:
        if col in available_columns:
            columns_to_include.append(col)
    
    # Print debug information
    print(f"Columns to include: {columns_to_include}")
    
    # Get recommendations with selected columns
    recommendations = df.iloc[recommended_indices][columns_to_include].copy()
    
    # Add similarity score
    recommendations['similarity'] = similarity_scores
    
    # Sort by similarity score (highest first)
    recommendations = recommendations.sort_values('similarity', ascending=False)
    
    # Round similarity score and format it as percentage
    recommendations['similarity'] = (recommendations['similarity'] * 100).round(1)
    
    # Get the original content
    original_content = df.iloc[idx].to_dict()
    
    # Print debug information
    print(f"Recommendations columns: {recommendations.columns.tolist()}")
    
    return original_content, recommendations.reset_index(drop=True)
