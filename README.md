# CineAI – Setup Guide

## 1. Install dependencies
```
pip install -r requirements.txt
```

## 2. Create a `.env` file in the same folder:
```
TMDB_API_KEY=your_tmdb_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

Get a free TMDB key at: https://www.themoviedb.org/settings/api
Get an Anthropic key at: https://console.anthropic.com/

## 3. Place your pickle files in the same folder:
- `df.pkl`           — pandas DataFrame with movie metadata
- `indices.pkl`      — Series or dict mapping title → row index
- `tfidf_matrix.pkl` — scipy sparse TF-IDF matrix

## 4. Run the app:
```
streamlit run cineai_app.py
```

## Features
| Feature              | Requires         |
|----------------------|-----------------|
| Browse / Search      | TMDB API key    |
| Genre filter         | TMDB API key    |
| Movie details+cast   | TMDB API key    |
| ✨ AI Insight button  | Anthropic key   |
| 🎭 Mood Finder       | Anthropic key   |
| 🤖 TF-IDF Recs       | Pickle files    |
