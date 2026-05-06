import streamlit as st
import pandas as pd
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    import urllib.request
    import zipfile
    import os

    # Download dataset if not already present
    if not os.path.exists('ml-1m/ratings.dat'):
        st.info("Downloading MovieLens dataset for first time...")
        urllib.request.urlretrieve(
            'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
            'ml-1m.zip'
        )
        with zipfile.ZipFile('ml-1m.zip', 'r') as z:
            z.extractall('.')
        os.remove('ml-1m.zip')

    ratings = pd.read_csv(
        'ml-1m/ratings.dat', sep='::',
        names=['userId','movieId','rating','timestamp'],
        engine='python')
    movies = pd.read_csv(
        'ml-1m/movies.dat', sep='::',
        names=['movieId','title','genres'],
        engine='python', encoding='latin-1')
    users = pd.read_csv(
        'ml-1m/users.dat', sep='::',
        names=['userId','gender','age','occupation','zip'],
        engine='python')

    movie_stats = ratings.groupby('movieId')['rating'].agg(
        avg_rating='mean', num_ratings='count').reset_index()
    movies = pd.merge(movies, movie_stats, on='movieId')
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
    movies['year'] = pd.to_numeric(movies['year'], errors='coerce')
    return movies, users

movies, users = load_data()

# Header
st.title("🎬 AI Movie Recommendation System")
st.markdown("Tell us what you like and I wil recommend the perfect movies according to your preference")
st.divider()

# Input section
col1, col2, col3 = st.columns(3)

with col1:
    genre = st.selectbox("Favourite Genre", [
        "Action", "Comedy", "Drama", "Romance", 
        "Thriller", "Horror", "Animation", "Documentary",
        "Adventure", "Sci-Fi", "Fantasy", "Crime", "Mystery"
    ])

with col2:
    actor = st.text_input("Favourite Actor/Actress", 
        placeholder="e.g. Tom Hanks, Leonardo DiCaprio")

with col3:
    time_period = st.selectbox("Preferred Time Period", [
        "Any", "Classic (before 1970)", 
        "70s & 80s", "90s", "2000s"
    ])

mood = st.text_area("Describe what you're in the mood for",
    placeholder="e.g. I want something funny and light hearted, or an intense thriller that keeps me on edge...",
    height=100)

st.divider()

# Filter movies based on inputs
def filter_movies(genre, time_period, min_ratings=50):
    filtered = movies[
        (movies['genres'].str.contains(genre, case=False)) &
        (movies['num_ratings'] >= min_ratings)
    ].copy()
    
    if time_period == "Classic (before 1970)":
        filtered = filtered[filtered['year'] < 1970]
    elif time_period == "70s & 80s":
        filtered = filtered[
            (filtered['year'] >= 1970) & (filtered['year'] < 1990)]
    elif time_period == "90s":
        filtered = filtered[
            (filtered['year'] >= 1990) & (filtered['year'] < 2000)]
    elif time_period == "2000s":
        filtered = filtered[filtered['year'] >= 2000]
    
    return filtered.sort_values('avg_rating', ascending=False).head(30)

# Get AI recommendation
def get_ai_recommendation(genre, actor, time_period, mood, movie_list):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    movies_text = "\n".join([
        f"- {row['title']} | Genres: {row['genres']} | "
        f"Avg Rating: {row['avg_rating']:.1f} | "
        f"Votes: {row['num_ratings']}"
        for _, row in movie_list.iterrows()
    ])
    
    prompt = f"""
You are a movie expert recommender. Based on the user's preferences, 
recommend the TOP 5 most suitable movies from the list below.

User Preferences:
- Favourite Genre: {genre}
- Favourite Actor/Actress: {actor if actor else 'Not specified'}
- Preferred Time Period: {time_period}
- Mood/Description: {mood if mood else 'Not specified'}

Available Movies:
{movies_text}

Return exactly 5 movie recommendations in this format for each:
🎬 **Movie Title (Year)**
⭐ Rating: X.X/5
🎭 Why you'll love it: [2 sentence personalised reason based on user preferences]
---
"""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1000
    )
    
    return response.choices[0].message.content

# Recommend button
if st.button("🎬 Get My Recommendations", type="primary"):
    if not os.getenv("GROQ_API_KEY"):
        st.error("API key not found! Check your .env file.")
    else:
        with st.spinner("Finding perfect movies for you..."):
            # Filter dataset
            filtered_movies = filter_movies(genre, time_period)
            
            if filtered_movies.empty:
                st.warning("No movies found with these filters. Try different options!")
            else:
                # Get AI recommendations
                result = get_ai_recommendation(
                    genre, actor, time_period, mood, filtered_movies)
                
                st.subheader("🍿 Your Personalised Recommendations")
                st.markdown(result)
                st.divider()
                
                # Show full filtered list
                with st.expander("See all matching movies from dataset"):
                    st.dataframe(
                        filtered_movies[['title','genres','avg_rating','num_ratings','year']] \
                            .rename(columns={
                                'title':'Title',
                                'genres':'Genres', 
                                'avg_rating':'Avg Rating',
                                'num_ratings':'No. of Ratings',
                                'year':'Year'
                            }).reset_index(drop=True),
                        use_container_width=True
                    )

st.divider()
st.caption("Built with Streamlit · Powered by Groq AI · Dataset: MovieLens 1M")