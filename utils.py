import pandas as pd
import os
import numpy as np
import re

def load_data(path):
    """
    Loads and validates the dataset from the given path with enhanced error handling
    and data preprocessing.
    
    Args:
        path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded and validated DataFrame, or empty DataFrame on failure
    """
    if not os.path.exists(path):
        print(f"❌ Data file not found at: {path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Looking for file: {os.path.basename(path)}")
        print(f"In directory: {os.path.dirname(path)}")
        print(f"Files in directory: {os.listdir(os.path.dirname(path)) if os.path.exists(os.path.dirname(path)) else 'Directory not found'}")
        raise FileNotFoundError(f"❌ Data file not found at: {path}")
    
    try:
        # Print file details for debugging
        print(f"Found file: {path}")
        print(f"File size: {os.path.getsize(path)} bytes")
        
        # Try to read file directly first
        try:
            with open(path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                print(f"First line of file: {first_line}")
        except Exception as e:
            print(f"Error reading first line: {str(e)}")
            
        # Try multiple encodings and delimiters to maximize chances of successful loading
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        delimiters = [',', ';', '\t', '|']
        df = None
        
        for encoding in encodings:
            for delimiter in delimiters:
                try:
                    print(f"Trying to load with encoding={encoding}, delimiter='{delimiter}'")
                    df = pd.read_csv(path, encoding=encoding, delimiter=delimiter, engine='python')
                    if not df.empty:
                        print(f"✅ Successfully loaded with encoding={encoding}, delimiter='{delimiter}'")
                        print(f"Columns found: {df.columns.tolist()}")
                        print(f"Data shape: {df.shape}")
                        break
                except Exception as e:
                    print(f"Failed with encoding={encoding}, delimiter='{delimiter}': {str(e)}")
                    continue
            if df is not None and not df.empty:
                break
        
        if df is None or df.empty:
            raise ValueError("❌ Loaded DataFrame is empty or unreadable.")
        
        # Print raw column names before cleaning
        print("Raw column names:", df.columns.tolist())
        
        # Clean up column names
        df.columns = [clean_column_name(col) for col in df.columns]
        
        # Print cleaned column names
        print("Cleaned column names:", df.columns.tolist())
        
        # Normalize columns and rename for consistency
        column_mapping = {
            'score': 'rating',
            'ratings': 'rating',
            'rate': 'rating',
            'imdb_rating': 'rating',
            'imdb_score': 'rating',
            'name': 'title',
            'movie_title': 'title',
            'show_title': 'title',
            'film_title': 'title',
            'content_type': 'type',
            'category': 'type',
        }
        
        # Apply column mapping
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                print(f"Renaming column {old_col} to {new_col}")
                df.rename(columns={old_col: new_col}, inplace=True)
        
        # Print columns after mapping
        print("Columns after mapping:", df.columns.tolist())
        
        # Check for required columns
        required_columns = ['title', 'genres', 'rating']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            print(f"Available columns: {df.columns.tolist()}")
            raise ValueError(f"❌ Missing required columns: {missing_cols}")
        
        # Preprocess data for better recommendations
        df = preprocess_data(df)
        
        # Drop rows with missing values in essential columns
        df = df.dropna(subset=required_columns)
        
        if df.empty:
            raise ValueError("❌ All rows dropped after NA removal")
        
        # Return clean, validated dataframe
        return df
    
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return pd.DataFrame()


def clean_column_name(col):
    """Cleans and normalizes column names"""
    if not isinstance(col, str):
        return str(col).lower().strip()
    
    # Just convert to lowercase and strip whitespace
    return col.lower().strip()


def preprocess_data(df):
    """
    Preprocesses the dataframe to improve recommendation quality.
    
    Args:
        df (pd.DataFrame): The raw dataframe
        
    Returns:
        pd.DataFrame: The processed dataframe
    """
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Convert ratings to numeric (0-10 scale)
    if 'rating' in processed_df.columns:
        # Convert to string first to handle various formats
        processed_df['rating'] = processed_df['rating'].astype(str)
        
        # Extract numeric values (handles formats like "8.5/10")
        processed_df['rating'] = processed_df['rating'].apply(extract_rating)
        
        # Convert to float
        processed_df['rating'] = pd.to_numeric(processed_df['rating'], errors='coerce')
        
        # Normalize ratings to 0-10 scale if they seem to be on a different scale
        if processed_df['rating'].max() <= 5:
            processed_df['rating'] = processed_df['rating'] * 2
        elif processed_df['rating'].max() <= 100:
            processed_df['rating'] = processed_df['rating'] / 10
    
    # Clean genres column
    if 'genres' in processed_df.columns:
        # Convert to string and clean up
        processed_df['genres'] = processed_df['genres'].astype(str)
        
        # Normalize format (handle lists, separators)
        processed_df['genres'] = processed_df['genres'].apply(clean_genre)
    
    # Ensure title is string
    if 'title' in processed_df.columns:
        processed_df['title'] = processed_df['title'].astype(str)
    
    # Add content type if missing
    if 'type' not in processed_df.columns:
        # Try to infer type from other columns
        if 'anime_type' in processed_df.columns:
            processed_df['type'] = 'Anime'
        elif 'movie_id' in processed_df.columns:
            processed_df['type'] = 'Movie'
        elif 'tv_id' in processed_df.columns:
            processed_df['type'] = 'TV Show'
        elif 'kdrama_id' in processed_df.columns:
            processed_df['type'] = 'K-Drama'
        else:
            # Try to infer from genres
            processed_df['type'] = processed_df['genres'].apply(infer_content_type)
    
    return processed_df


def extract_rating(rating_str):
    """Extracts numeric rating from various formats"""
    if not rating_str or pd.isna(rating_str):
        return np.nan
    
    # Extract numbers from the string
    numbers = re.findall(r'\d+\.?\d*', str(rating_str))
    
    if not numbers:
        return np.nan
    
    # Return the first number found
    return float(numbers[0])


def clean_genre(genre_str):
    """
    Cleans and normalizes genre strings in various formats
    Examples: [Action, Adventure] -> Action, Adventure
              Action|Adventure -> Action, Adventure
    """
    if not genre_str or pd.isna(genre_str) or genre_str.lower() == 'nan':
        return ''
    
    # Remove brackets, quotes, etc.
    cleaned = re.sub(r'[\[\]\'"{}\(\)]', '', str(genre_str))
    
    # Normalize separators (| or / or ; to commas)
    cleaned = re.sub(r'[|;/]', ',', cleaned)
    
    # Ensure consistent spacing around commas
    cleaned = re.sub(r'\s*,\s*', ', ', cleaned)
    
    # Remove trailing comma if it exists
    cleaned = re.sub(r',\s*$', '', cleaned)
    
    return cleaned


def infer_content_type(genre_str):
    """Attempts to infer content type from genre"""
    genre_lower = str(genre_str).lower()
    
    if 'anime' in genre_lower:
        return 'Anime'
    elif 'kdrama' in genre_lower or 'korean drama' in genre_lower:
        return 'K-Drama'
    elif 'series' in genre_lower or 'tv show' in genre_lower:
        return 'TV Show'
    else:
        return 'Movie'  # Default type
