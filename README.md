# 🎬 Cineflix • AI Movie Discovery & Recommendation Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://movie-suggestion-project-rk9z9h6u3dhydfuie2pzx7.streamlit.app/)
[![Author](https://img.shields.io/badge/Author-Atharv%20Shukla-E50914?style=flat&logo=github&logoColor=white)](https://github.com/atharvshukla76)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Personalized Cinematic Intelligence, Global Search & Real-Time Recommendations**  
> Architected and Developed by **[Atharv Shukla](https://github.com/atharvshukla76)**

---

## 🚀 Live Demo & Deployment

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://movie-suggestion-project-rk9z9h6u3dhydfuie2pzx7.streamlit.app/)

- 🌐 Try the interactive web application: **[Live Demo App](https://movie-suggestion-project-rk9z9h6u3dhydfuie2pzx7.streamlit.app/)**
- 💻 **Run Locally**:
```bash
streamlit run netflix_ui.py
```

---

## 📖 Project Overview

**Cineflix** is a next-generation movie discovery and recommendation platform inspired by Netflix and modern streaming architectures. It bridges content-based machine learning with live global cinema metadata to deliver instant, high-fidelity movie recommendations.

Unlike traditional static recommenders, Cineflix provides:
1. **Universal Search Coverage**: Search for **any** movie in the world (e.g. *Avengers: Endgame*, *Dune*, *The Batman*, *Inception*, trailers) and receive high-res posters, ratings, synopsis, streaming providers, and official YouTube trailer links.
2. **Dual-Core AI Recommendations**: Combines local TF-IDF cosine similarity and Random Forest genre classification with live TMDB user preference graphs.
3. **Dynamic Search-Tuned Notifications**: The notification popover (`🔔`) recalibrates in real-time based on what you search, alerting you to high-affinity companions (e.g., searching *Avengers: Endgame* immediately recommends *Captain America: Civil War*).

---

## ✨ Key Features

### 🔍 Universal Movie & Franchise Search
- Instant typo-tolerant search across millions of global titles via TMDB.
- Automatically handles noise words (e.g. searching `"Avengers trailer"` strips noise and resolves directly to *The Avengers* with an official YouTube trailer button).
- Fallback integration with local ML models ensures **zero** "Movie not found" dead ends.

### 🔔 Search-Tuned Dynamic Notifications
- **AI Resonance Matches**: Direct recommendations triggered by your latest search (`"Because you searched for 'Avengers: Endgame', our AI flagged Captain America: Civil War as a 98% match!"`).
- **Trending in Your Genre**: 4K release alerts based on your recent genre interactions.
- **Taste Calibration**: Real-time updates to your neural taste profile.
- Unread notification counter badge (`🔔 (3)`) that resets with "✓ Mark All as Read" and reactivates on new searches.

### 🧭 Domain-Specific Curated Collections
- Curated carousels with 100% verified TMDB posters across all major categories:
  - 🔥 **Trending Now** (*Oppenheimer*, *Avengers: Endgame*, *Interstellar*, *Across the Spider-Verse*, *The Batman*, *Inception*)
  - 📅 **New Releases 2024** (*Dune: Part Two*, *Deadpool & Wolverine*, *Civil War*, *Inside Out 2*, *Furiosa*, *Godzilla x Kong*)
  - 🎭 **Genre Rows**: Action, Sci-Fi, Thriller, Comedy, Romance, Horror, Drama, Animation. Each genre strictly features its own verified domain titles.

### 🤍 "Your Space" Personalization Hub
- **My Watchlist**: Interactive queue with 1-click addition and deletion.
- **Watch & Search History**: Chronological tracking of every search and watched movie. Click "Re-Search" or "Rewatch" to instantly reload any journey.
- **My Ratings**: Community & personal rating impressions with reviews.
- **Cinematic DNA Slate**: High-tech neural tablet showing cluster affinity scores across Sci-Fi, Noir, Psychological Thrillers, and Action.

### 🔀 Smart Tools
- **😊 Mood Explorer**: Filter films by vibe (*Dark & Gritty*, *Mind-Bending*, *Chill & Relaxed*, *Romantic*).
- **🎲 Surprise Me Roulette**: AI-driven cinema roulette for indecisive movie nights.
- **⚖️ Side-by-Side Comparison**: Comparative showdown analyzing runtimes, ratings, genres, and audience resonance.

### 🌌 Design & Visual Aesthetics
- Dark liquid-cosmic space theme with ambient particle backdrop.
- Custom translucent glassmorphism cards (`backdrop-filter: blur(16px)`).
- Neon cyan accents (`#00f2fe`) and Netflix red interactive hover glows (`#e50914`).
- Streamlit BaseWeb popover styling with dark translucent dropdowns.

---

## 🗂️ File & Directory Structure

```text
movie-suggestion-project/
│
├── .streamlit/
│   └── config.toml                   # Theme tokens (dark cosmic) & server config (headless = false)
├── data_processing_of_netflix_recommendation.py  # Backend: TF-IDF, Cosine Similarity & Random Forest model
├── netflix_ui.py                     # Frontend: Complete Streamlit interface & view controllers
├── tmdb_service.py                   # TMDB API Client with DoH (DNS-over-HTTPS) & CloudFront fallback
├── netflix_titles-2.csv.xlsx         # Dataset: Primary Netflix movie & TV catalog (612 KB)
├── netflix_clean.csv                 # Dataset: Cleaned, preprocessed dataset for rapid boot (1.78 MB)
├── requirements.txt                  # Python package dependencies
├── .env.example                      # Environment variables template
├── .gitignore                        # Git ignore rules (protects API keys & ignores 456MB model_cache.pkl)
└── README.md                         # Project documentation
```

> **Note on Datasets**: Both `netflix_titles-2.csv.xlsx` and `netflix_clean.csv` are tracked in the repository (~2 MB combined) so the app can build its recommendation models cleanly on fresh clones or cloud platforms. Large binary cache files (`model_cache.pkl`, ~456 MB) are ignored via `.gitignore` to adhere to GitHub's 100 MB file limit.

---

## 💻 Tech Stack

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/) (v1.65+)
- **Core Language**: Python 3.10+
- **Machine Learning & NLP**:
  - `scikit-learn`: TF-IDF Vectorizer, Cosine Similarity, Random Forest Classifier
  - `joblib`: Fast model persistence and caching
- **Data Engineering**: `pandas`, `numpy`, `openpyxl`
- **External Movie APIs & Networking**:
  - [The Movie Database (TMDB) API](https://www.themoviedb.org/documentation/api)
  - `requests`: HTTP client with Google DNS-over-HTTPS (DoH) resolution and CloudFront edge routing to bypass ISP sinkholes
  - YouTube search query engine for trailer playback

---

## ⚡ Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/atharvshukla76/movie-suggestion-project.git
cd movie-suggestion-project
```

### 2. Set Up a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure TMDB API Key (Optional but Recommended)
Copy the example environment file and add your free TMDB API key:
```bash
cp .env.example .env
```
Open `.env` and set:
```env
TMDB_API_KEY=your_actual_tmdb_api_key_here
```
*(If omitted, the app gracefully falls back to the local Scikit-Learn content engine and curated high-definition catalogs).*

### 5. Launch the Application
```bash
streamlit run netflix_ui.py
```
*Your browser will launch automatically at `http://localhost:8501`.*

---

## 🛠️ How the Recommendation Engine Works

1. **Local Content-Based Filtering**:
   - Features (`listed_in`, `description`, `country`, `rating`) are vectorized into an n-gram TF-IDF matrix.
   - Pairwise cosine similarity computes affinity scores across all titles.
   - A Random Forest classifier predicts content type and genre distribution.
2. **Global Preference Graph (TMDB)**:
   - Live TMDB integration queries real-time user co-watch habits and thematic similarity vectors.
   - Resilient DNS-over-HTTPS resolver routes API calls through AWS CloudFront edge IPs, ensuring high availability.
3. **Typo & Keyword Normalization**:
   - `difflib` provides fuzzy matching for misspelt titles (e.g., `"stanger things"` → *Stranger Things*).
   - Noise-stripping regex cleans tokens like `"trailer"`, `"movie"`, `"official"`, and `"4k"`.

---

## 👨‍💻 Author & Engineering

Designed, engineered, and maintained by **[Atharv Shukla](https://github.com/atharvshukla76)**.

- **GitHub**: [@atharvshukla76](https://github.com/atharvshukla76)
- **Repository**: [atharvshukla76/movie-suggestion-project](https://github.com/atharvshukla76/movie-suggestion-project)
- **Live Demo**: [Cineflix on Streamlit Community Cloud](https://movie-suggestion-project-rk9z9h6u3dhydfuie2pzx7.streamlit.app/)

Feel free to open an issue or pull request to contribute, suggest features, or report bugs!

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
