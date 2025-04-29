import pandas as pd
import logging
from pathlib import Path
import sys

# Configure logging
logger = logging.getLogger(__name__)

def load_master_dataset():
    """Load the master dataset from CSV."""
    try:
        df = pd.read_csv('master_dataset.csv')
        logger.info("Successfully loaded master dataset")
        return df
    except Exception as e:
        logger.error(f"Error loading master dataset: {str(e)}")
        raise

def preprocess_data(df):
    """Preprocess the dataset for recommendation system."""
    try:
        # Handle missing values
        df['genres'] = df['genres'].fillna('')
        df['rating'] = df['rating'].fillna(0)
        df['platform'] = df['platform'].fillna('Unknown')
        
        # Convert title to lowercase for better matching
        df['title_lower'] = df['title'].str.lower()
        
        # Clean genres (remove special characters, etc.)
        df['genres'] = df['genres'].str.replace('[^\w\s]', '')
        
        logger.info("Successfully preprocessed data")
        return df
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

def combine_features(df):
    """Combine relevant features for similarity calculation."""
    try:
        df['combined_features'] = df['title'] + ' ' + df['genres']
        logger.info("Successfully combined features")
        return df
    except Exception as e:
        logger.error(f"Error combining features: {str(e)}")
        raise

def save_processed_data(df, output_path):
    """Save processed data to CSV."""
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved processed data to {output_path}")
    except Exception as e:
        logger.error(f"Error saving processed data: {str(e)}")
        raise 