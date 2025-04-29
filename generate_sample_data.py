import pandas as pd
import numpy as np
import random
from pathlib import Path

def generate_sample_data(n_samples=1000):
    """Generate a sample dataset for testing."""
    # Sample titles
    anime_titles = [
        "Attack on Titan", "Death Note", "Naruto", "One Piece", "Dragon Ball Z",
        "My Hero Academia", "Demon Slayer", "Jujutsu Kaisen", "Tokyo Ghoul", "Fullmetal Alchemist"
    ]
    
    movie_titles = [
        "The Shawshank Redemption", "The Godfather", "The Dark Knight", "Pulp Fiction",
        "Forrest Gump", "Inception", "The Matrix", "Goodfellas", "The Silence of the Lambs"
    ]
    
    kdrama_titles = [
        "Squid Game", "Crash Landing on You", "Descendants of the Sun", "Goblin",
        "Itaewon Class", "Vincenzo", "Start-Up", "Hotel del Luna", "It's Okay to Not Be Okay"
    ]
    
    # Sample genres
    genres = [
        "Action", "Adventure", "Comedy", "Drama", "Fantasy", "Horror",
        "Mystery", "Romance", "Sci-Fi", "Thriller", "Slice of Life", "Supernatural"
    ]
    
    # Sample platforms
    platforms = ["Netflix", "Hulu", "Amazon Prime", "Disney+", "Crunchyroll", "Viki"]
    
    # Generate data
    data = []
    all_titles = anime_titles + movie_titles + kdrama_titles
    
    for _ in range(n_samples):
        title = random.choice(all_titles)
        title_genres = random.sample(genres, k=random.randint(1, 3))
        rating = round(random.uniform(1, 10), 1)
        platform = random.choice(platforms)
        
        data.append({
            'title': title,
            'genres': ', '.join(title_genres),
            'rating': rating,
            'platform': platform
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_path = Path('master_dataset.csv')
    df.to_csv(output_path, index=False)
    print(f"Generated sample dataset with {n_samples} entries")
    return df

if __name__ == "__main__":
    generate_sample_data() 