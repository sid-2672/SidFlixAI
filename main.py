import streamlit as st
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Now import the modules
from modules.utils import load_data
from modules.recommender import build_recommender, recommend_content
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define full path to dataset - with more flexible path options
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Try different possible data paths
POSSIBLE_DATA_PATHS = [
    os.path.join(BASE_DIR, "data", "master_dataset.csv"),
    os.path.join(BASE_DIR, "master_dataset.csv"),
    os.path.join(os.path.dirname(__file__), "master_dataset.csv"),
    os.path.join(os.path.dirname(__file__), "data", "master_dataset.csv"),
    "master_dataset.csv"  # Just in current directory
]

# Print working directory and available paths for debugging
print(f"Current working directory: {os.getcwd()}")
print(f"Script directory: {os.path.dirname(__file__)}")
print(f"Base directory: {BASE_DIR}")
print("Searching for data file in these locations:")
for possible_path in POSSIBLE_DATA_PATHS:
    print(f" - {possible_path} (exists: {os.path.exists(possible_path)})")

# Find first existing data path
DATA_PATH = next((path for path in POSSIBLE_DATA_PATHS if os.path.exists(path)), POSSIBLE_DATA_PATHS[0])

# Page configuration
st.set_page_config(
    page_title="SidFlix AI", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS - with fallback
css_path = os.path.join(BASE_DIR, "assets", "styles.css")
if not os.path.exists(css_path):
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")

try:
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Inline minimal CSS if file not found
        st.markdown("""
        <style>
        body { background-color: #111827; color: #ffffff; }
        .stTextInput > div > div > input { background-color: #1f2937; color: white; }
        [data-testid="stDataFrame"] { background-color: #1f2937; }
        thead tr th { background-color: #374151; color: white; }
        </style>
        """, unsafe_allow_html=True)
except Exception as e:
    print(f"Error loading CSS: {str(e)}")

# UI Header
st.title("🎬 SidFlix AI – Ultimate Recommendation Engine")
st.markdown("A god-tier AI that recommends **Anime, Movies, TV Shows, and K-Dramas** based on your input.")

# Data loading with better error handling and auto-create dummy data if needed
try:
    with st.spinner("Loading entertainment database..."):
        # Check if data file exists
        if not os.path.exists(DATA_PATH):
            st.warning("⚠️ Dataset not found! Creating a dummy dataset for demonstration...")
            
            # Create very basic dataset as last resort
            import pandas as pd
            dummy_data = {
                'title': ['Stranger Things', 'Breaking Bad', 'Attack on Titan', 'Inception'],
                'genres': ['Sci-Fi, Horror', 'Crime, Drama', 'Anime, Action', 'Sci-Fi, Action'],
                'rating': [8.7, 9.5, 9.0, 8.8],
                'platform': ['Netflix', 'Netflix', 'Crunchyroll', 'Netflix']
            }
            df = pd.DataFrame(dummy_data)
            df.to_csv('master_dataset.csv', index=False)
            DATA_PATH = 'master_dataset.csv'
            st.info("Created minimal dummy dataset as fallback")
        
        # Load the data directly with pandas
        print(f"Loading data from: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        print("Raw DataFrame columns:", df.columns.tolist())
        
        # Ensure required columns exist
        required_columns = ['title', 'genres', 'rating']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Clean and preprocess the data
        df['title'] = df['title'].astype(str)
        df['genres'] = df['genres'].astype(str)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        
        # Drop rows with missing values
        df = df.dropna(subset=required_columns)
        
        if df.empty:
            raise ValueError("All rows dropped after NA removal")
        
        # Reset index to ensure continuous indices
        df = df.reset_index(drop=True)
        
        # Build the recommender
        cosine_sim, indices = build_recommender(df)
        st.success(f"✅ Loaded {len(df)} titles successfully!")
        
except Exception as e:
    st.error(f"❌ Failed to load data: {str(e)}")
    st.text("Detailed error information:")
    st.code(str(e))
    st.stop()

# Add content type filter
content_types = ["All"] + sorted(df["type"].unique().tolist()) if "type" in df.columns else ["All"]
selected_type = st.selectbox("🎭 Filter by content type:", content_types)

# Enhanced search input
col1, col2 = st.columns([3, 1])
with col1:
    title = st.text_input("🔍 Enter a Show/Anime/Movie/K-Drama title:", help="Try titles like 'Stranger Things' or 'Attack on Titan'")
with col2:
    num_recommendations = st.slider("Number of recommendations:", 5, 20, 10)

# Filter data by content type if selected
filtered_df = df.copy()  # Create a copy to avoid modifying the original
if selected_type != "All" and "type" in df.columns:
    filtered_df = df[df["type"] == selected_type].copy()
    if filtered_df.empty:
        st.warning(f"No content available for type: {selected_type}")
        st.stop()
    # Reset index to ensure continuous indices
    filtered_df = filtered_df.reset_index(drop=True)
    # Rebuild recommender with filtered data
    cosine_sim, indices = build_recommender(filtered_df)
else:
    # Use the original recommender
    cosine_sim, indices = build_recommender(df)

# Show recommendations
if title:
    # Convert title to lowercase for case-insensitive matching
    title = title.lower()
    
    # Check if the title exists in our dataset
    if title not in indices:
        st.warning("😕 Couldn't find that exact title. Did you mean one of these?")
        # Find closest matches
        import difflib
        close_matches = difflib.get_close_matches(title, indices.index.tolist(), n=5, cutoff=0.6)
        
        if close_matches:
            for i, match in enumerate(close_matches):
                if st.button(f"🔍 {match.title()}", key=f"match_button_{i}"):
                    st.experimental_rerun()
        else:
            st.error("No similar titles found. Please try a different search.")
    else:
        result, recs = recommend_content(title, filtered_df, cosine_sim, indices, top_n=num_recommendations)
        
        if result is not None:
            # Display result in a nice card
            st.markdown("### 🎯 You searched for:")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"## {result['title']}")
                st.markdown(f"**Genres:** {result['genres']}")
                st.markdown(f"**Rating:** ⭐ {result['rating']}/10")
                
                # Show additional metadata if available
                if "platform" in result:
                    st.markdown(f"**Platform:** {result['platform']}")
                if "type" in result:
                    st.markdown(f"**Type:** {result['type']}")
            
            # Recommendations section
            st.markdown("### 🔮 You might also like:")
            
            # Add similarity score to recommendations
            if not recs.empty:
                # Print debug information
                print(f"Recommendations DataFrame columns: {recs.columns.tolist()}")
                
                # Format the recommendations table
                display_cols = []
                
                # Add required columns if they exist
                if 'title' in recs.columns:
                    display_cols.append('title')
                if 'genres' in recs.columns:
                    display_cols.append('genres')
                if 'rating' in recs.columns:
                    display_cols.append('rating')
                
                # Add optional columns if they exist
                if 'platform' in recs.columns:
                    display_cols.append('platform')
                if 'type' in recs.columns:
                    display_cols.append('type')
                
                # Print debug information
                print(f"Display columns: {display_cols}")
                
                # Style the dataframe
                st.dataframe(
                    recs[display_cols],
                    use_container_width=True,
                    column_config={
                        "title": "Title",
                        "genres": "Genres",
                        "rating": st.column_config.NumberColumn("Rating", format="%.1f ⭐"),
                        "platform": "Platform",
                        "type": "Type"
                    }
                )
            else:
                st.info("No similar content found. Try another title!")

# Footer
st.markdown("---")
st.markdown("### 🚀 SidFlix AI - Powered by Advanced Recommendation Algorithms")
st.markdown("Made with ❤️ for entertainment lovers everywhere")
