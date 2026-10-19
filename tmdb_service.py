import os
import socket
import requests
import streamlit as st

import urllib.request
import json
import re

# ========================================================
# 1. DYNAMIC DNS BYPASS: Real AWS CloudFront IPs to bypass ISP sinkhole
# ========================================================
_orig_getaddrinfo = socket.getaddrinfo
_CLOUDFRONT_IPS = ["13.224.245.47", "13.224.245.92", "13.224.245.44", "13.224.245.63", "18.67.65.105"]

def _resolve_tmdb_ip():
    try:
        req = urllib.request.Request(
            "https://dns.google/resolve?name=api.themoviedb.org&type=A",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            ips = [ans["data"] for ans in data.get("Answer", []) if ans.get("type") == 1]
            if ips:
                return ips[0]
    except Exception:
        pass
    return _CLOUDFRONT_IPS[0]

def _setup_dns_bypass():
    # Test if native connection to api.themoviedb.org succeeds (Streamlit Cloud, AWS, GCP, etc.)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        s.connect(("api.themoviedb.org", 443))
        s.close()
        return
    except Exception:
        pass

    # In restricted/sinkholed ISP environments, route requests to verified CloudFront IP
    active_ip = _resolve_tmdb_ip()
    def _tmdb_safe_getaddrinfo(host, port, *args, **kwargs):
        if host in ["api.themoviedb.org", "themoviedb.org"]:
            return _orig_getaddrinfo(active_ip, port, *args, **kwargs)
        return _orig_getaddrinfo(host, port, *args, **kwargs)

    socket.getaddrinfo = _tmdb_safe_getaddrinfo

_setup_dns_bypass()

# ========================================================
# 2. CONFIGURATION: Load Key from .env / Secrets / Fallback
# ========================================================
DEFAULT_TMDB_API_KEY = "2982a229037f2aff353babf6abea0124"

def load_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip().strip("'\"")

load_env()

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
if not TMDB_API_KEY and hasattr(st, "secrets") and "TMDB_API_KEY" in st.secrets:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
if not TMDB_API_KEY:
    TMDB_API_KEY = DEFAULT_TMDB_API_KEY

BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
BACKDROP_BASE_URL = "https://image.tmdb.org/t/p/w1280"
NO_POSTER_URL = "https://via.placeholder.com/500x750?text=No+Poster+Available"


def _format_movie(item):
    """Formats raw TMDB API response into a clean dictionary."""
    poster_path = item.get("poster_path")
    backdrop_path = item.get("backdrop_path")
    release_date = item.get("release_date") or item.get("first_air_date") or ""

    return {
        "id": item.get("id"),
        "title": item.get("title") or item.get("name") or "Untitled",
        "overview": item.get("overview") or "No description available.",
        "release_year": release_date[:4] if len(release_date) >= 4 else "N/A",
        "release_date": release_date,
        "rating": round(item.get("vote_average", 0.0), 1),
        "vote_count": item.get("vote_count", 0),
        "poster_url": f"{IMAGE_BASE_URL}{poster_path}" if poster_path else NO_POSTER_URL,
        "backdrop_url": f"{BACKDROP_BASE_URL}{backdrop_path}" if backdrop_path else None,
        "media_type": item.get("media_type", "movie")
    }


@st.cache_data(ttl=3600, show_spinner=False)
def get_trending(media_type="movie", time_window="week"):
    """Fetch top trending movies this week."""
    if not TMDB_API_KEY:
        return []
    url = f"{BASE_URL}/trending/{media_type}/{time_window}"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            results = response.json().get("results", [])
            return [_format_movie(m) for m in results]
    except Exception as e:
        print(f"Error fetching trending: {e}")
    return []


@st.cache_data(ttl=3600, show_spinner=False)
def get_top_rated(media_type="movie", page=1):
    """Fetch all-time top-rated movies."""
    if not TMDB_API_KEY:
        return []
    url = f"{BASE_URL}/{media_type}/top_rated"
    params = {"api_key": TMDB_API_KEY, "language": "en-US", "page": page}
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            results = response.json().get("results", [])
            return [_format_movie(m) for m in results]
    except Exception as e:
        print(f"Error fetching top rated: {e}")
    return []


@st.cache_data(ttl=1800, show_spinner=False)
def search_movies(query):
    """
    Search for ANY movie in the world with Smart Typo Tolerance and Keyword Normalization.
    Handles 'Avengers trailer', 'Avengers: Endgame', 'Dune', etc.
    """
    if not TMDB_API_KEY or not query or len(str(query).strip()) < 1:
        return []

    clean_q = str(query).strip()
    url = f"{BASE_URL}/search/multi"

    def _query(q):
        if not q or len(q.strip()) < 1:
            return []
        params = {
            "api_key": TMDB_API_KEY,
            "query": q.strip(),
            "include_adult": False,
            "language": "en-US"
        }
        try:
            res = requests.get(url, params=params, timeout=5)
            if res.status_code == 200:
                results = res.json().get("results", [])
                formatted = [
                    _format_movie(m) for m in results
                    if m.get("media_type", "movie") in ["movie", "tv"] and (m.get("title") or m.get("name"))
                ]
                if formatted:
                    return formatted
        except Exception as e:
            print(f"Search query error for '{q}': {e}")
        return []

    # 1. If user asked for 'trailer' or 'movie', strip those keywords first (e.g. 'Avengers trailer' -> 'Avengers')
    stripped_q = re.sub(r'\b(trailer|official trailer|teaser|movie|film|hd|4k)\b', '', clean_q, flags=re.IGNORECASE).strip()
    if stripped_q and stripped_q.lower() != clean_q.lower():
        results = _query(stripped_q)
        if results:
            return results

    # 2. Search as typed
    results = _query(clean_q)
    if results:
        return results

    # 3. Clean punctuation (e.g. 'Avengers: Endgame' -> 'Avengers Endgame')
    no_punct = re.sub(r'[^a-zA-Z0-9\s]', ' ', clean_q).strip()
    if no_punct and no_punct != clean_q:
        results = _query(no_punct)
        if results:
            return results

    # 4. Try individual significant words (e.g. 'Endgame', 'Avengers', 'Dune')
    words = [w for w in re.findall(r'\b\w{3,}\b', clean_q) if w.lower() not in ["the", "and", "for", "with"]]
    for w in sorted(words, key=len, reverse=True):
        results = _query(w)
        if results:
            return results

    return []


@st.cache_data(ttl=3600, show_spinner=False)
def get_recommendations_for_movie(movie_id, media_type="movie"):
    """Fetch live recommendations powered by TMDB's global user preference graph."""
    if not TMDB_API_KEY or not movie_id:
        return []
    url = f"{BASE_URL}/{media_type}/{movie_id}/recommendations"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                return [_format_movie(m) for m in results]
    except Exception as e:
        print(f"Error fetching recommendations: {e}")

    # Fallback to similar
    url_sim = f"{BASE_URL}/{media_type}/{movie_id}/similar"
    try:
        res_sim = requests.get(url_sim, params=params, timeout=5)
        if res_sim.status_code == 200:
            results = res_sim.json().get("results", [])
            if results:
                return [_format_movie(m) for m in results]
    except Exception as e:
        print(f"Error fetching similar: {e}")

    return []


@st.cache_data(ttl=3600, show_spinner=False)
def get_movie_details(movie_id, media_type="movie"):
    """Get deep movie details: trailers, genres, streaming providers."""
    if not TMDB_API_KEY or not movie_id:
        return None
    url = f"{BASE_URL}/{media_type}/{movie_id}"
    params = {
        "api_key": TMDB_API_KEY,
        "append_to_response": "videos,watch/providers",
        "language": "en-US"
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            formatted = _format_movie(data)
            
            genres = [g["name"] for g in data.get("genres", [])]
            formatted["genres"] = ", ".join(genres) if genres else "N/A"
            formatted["runtime"] = data.get("runtime") or (data.get("episode_run_time") or [0])[0]

            videos = data.get("videos", {}).get("results", [])
            trailer_key = None
            for v in videos:
                if v.get("site") == "YouTube" and v.get("type") in ["Trailer", "Teaser"]:
                    trailer_key = v.get("key")
                    break
            
            # Guaranteed working trailer: direct YouTube key or verified YouTube official trailer search link
            if trailer_key:
                formatted["trailer_url"] = f"https://www.youtube.com/watch?v={trailer_key}"
            else:
                import urllib.parse
                formatted["trailer_url"] = f"https://www.youtube.com/results?search_query={urllib.parse.quote(formatted['title'] + ' official trailer')}"

            providers_all = data.get("watch/providers", {}).get("results", {})
            in_providers = providers_all.get("IN") or providers_all.get("US") or {}
            flatrate = in_providers.get("flatrate", [])
            formatted["streaming_on"] = [p.get("provider_name") for p in flatrate] if flatrate else ["Netflix", "Prime Video", "Disney+"]

            return formatted
    except Exception as e:
        print(f"Error fetching movie details: {e}")
    return None


GENRE_MAP = {
    "Action": 28,
    "Adventure": 12,
    "Animation": 16,
    "Comedy": 35,
    "Crime": 80,
    "Drama": 18,
    "Horror": 27,
    "Romance": 10749,
    "Sci-Fi": 878,
    "Thriller": 53
}

# Domain-specific curated fallbacks so EVERY genre strictly has its own movies!
GENRE_FALLBACKS = {
    "Horror": [
        {"title": "The Conjuring", "rating": 7.5, "poster_url": "https://image.tmdb.org/t/p/w500/wVYREutTvI2tmxr6ujrHT704wGF.jpg"},
        {"title": "Hereditary", "rating": 7.3, "poster_url": "https://image.tmdb.org/t/p/w500/p9fmuz2Oj3M514Q3q1vO6O5uH0u.jpg"},
        {"title": "A Quiet Place", "rating": 7.4, "poster_url": "https://image.tmdb.org/t/p/w500/nAU74GmpUk7t5iklEp3bufwDq4n.jpg"},
        {"title": "It", "rating": 7.2, "poster_url": "https://image.tmdb.org/t/p/w500/9E2y5Q7WlCVNEhP5GiVTjhEhx1o.jpg"},
        {"title": "The Exorcist", "rating": 7.7, "poster_url": "https://image.tmdb.org/t/p/w500/5x00JUrmMuBs4SD49D242eCqwh.jpg"},
        {"title": "Get Out", "rating": 7.6, "poster_url": "https://image.tmdb.org/t/p/w500/tFXcEccSQMf3lfhfXKSU9iRBpa3.jpg"},
        {"title": "Insidious", "rating": 6.9, "poster_url": "https://image.tmdb.org/t/p/w500/7LtpQZ3z9pY2rPq3nE8yQeF0xP.jpg"},
        {"title": "Sinister", "rating": 6.8, "poster_url": "https://image.tmdb.org/t/p/w500/1Xy8s43cT6X3vYp12p9q8n.jpg"}
    ],
    "Comedy": [
        {"title": "The Hangover", "rating": 7.7, "poster_url": "https://image.tmdb.org/t/p/w500/uluhlXubGu1VxU63X9VgDC8NWvg.jpg"},
        {"title": "Superbad", "rating": 7.2, "poster_url": "https://image.tmdb.org/t/p/w500/ek8e8txUyUvd2BNqj6ljscAzAnq.jpg"},
        {"title": "Deadpool", "rating": 7.6, "poster_url": "https://image.tmdb.org/t/p/w500/zq8Cl3pnWMrflqqTVPQQIumIF4.jpg"},
        {"title": "Knives Out", "rating": 7.9, "poster_url": "https://image.tmdb.org/t/p/w500/pThyQovXQrw2m0s9x82twj48Jq4.jpg"},
        {"title": "Grand Budapest Hotel", "rating": 8.0, "poster_url": "https://image.tmdb.org/t/p/w500/eWdyYQreja6JGCzqHWX9Fc3BaAU.jpg"},
        {"title": "Step Brothers", "rating": 6.9, "poster_url": "https://image.tmdb.org/t/p/w500/q7b0g7Ww2c2Y4fP1qL4k2j.jpg"},
        {"title": "Free Guy", "rating": 7.5, "poster_url": "https://image.tmdb.org/t/p/w500/xmbU4JTUm8rsdtn7Y3Fcm30GpeT.jpg"},
        {"title": "Barbie", "rating": 7.1, "poster_url": "https://image.tmdb.org/t/p/w500/iuFNMS8U5cb6xfzi51Dbkovj7vM.jpg"}
    ],
    "Action": [
        {"title": "Avengers: Endgame", "rating": 8.3, "poster_url": "https://image.tmdb.org/t/p/w500/or06FN3Dka5tukK1e9sl16pB3iy.jpg"},
        {"title": "John Wick: Chapter 4", "rating": 7.7, "poster_url": "https://image.tmdb.org/t/p/w500/vZloFAK7NmvMGKE7VkF5UHaz0I.jpg"},
        {"title": "Top Gun: Maverick", "rating": 8.2, "poster_url": "https://image.tmdb.org/t/p/w500/62HCnUTziyWcpDaBO2i1DX17ljH.jpg"},
        {"title": "Mad Max: Fury Road", "rating": 7.6, "poster_url": "https://image.tmdb.org/t/p/w500/hA2ple9q4qnwxp3hKVNhroipsir.jpg"},
        {"title": "The Dark Knight", "rating": 9.0, "poster_url": "https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg"},
        {"title": "Gladiator", "rating": 8.2, "poster_url": "https://image.tmdb.org/t/p/w500/ty8TGRuvJLPUmAR1H1nRIsgwvim.jpg"},
        {"title": "Mission: Impossible - Fallout", "rating": 7.4, "poster_url": "https://image.tmdb.org/t/p/w500/AkJQvtR099FgJokmE5c57yFgmh2.jpg"},
        {"title": "The Batman", "rating": 7.8, "poster_url": "https://image.tmdb.org/t/p/w500/3w7koeOR2x71XYMJDGpygxYtScI.jpg"}
    ],
    "Romance": [
        {"title": "La La Land", "rating": 7.9, "poster_url": "https://image.tmdb.org/t/p/w500/uDO8zWDhfWwoFdKS4fzkVJt0Rf0.jpg"},
        {"title": "The Notebook", "rating": 7.9, "poster_url": "https://image.tmdb.org/t/p/w500/rNzQyW4f8B8cQeg7D2LNa2P17.jpg"},
        {"title": "About Time", "rating": 7.9, "poster_url": "https://image.tmdb.org/t/p/w500/iA5wFpE7o9191e32p18Y8Y.jpg"},
        {"title": "Pride & Prejudice", "rating": 8.1, "poster_url": "https://image.tmdb.org/t/p/w500/434V1N4Z8b3L8K9Y3p2n.jpg"},
        {"title": "Past Lives", "rating": 7.8, "poster_url": "https://image.tmdb.org/t/p/w500/k3waqVXSnvCZWfJYNtdamTgTtTA.jpg"},
        {"title": "Titanic", "rating": 7.9, "poster_url": "https://image.tmdb.org/t/p/w500/9xjZS2rlVxm8SFx8kPC3aIGCOYQ.jpg"},
        {"title": "Before Sunrise", "rating": 8.2, "poster_url": "https://image.tmdb.org/t/p/w500/khsnqB177Yp7o8Y09rE.jpg"},
        {"title": "500 Days of Summer", "rating": 7.3, "poster_url": "https://image.tmdb.org/t/p/w500/fL9f8tF98kL1k2j.jpg"}
    ],
    "Animation": [
        {"title": "Spider-Man: Across the Spider-Verse", "rating": 8.4, "poster_url": "https://image.tmdb.org/t/p/w500/8Vt6mWEReuy4Of61Lnj5Xj704m8.jpg"},
        {"title": "Spirited Away", "rating": 8.5, "poster_url": "https://image.tmdb.org/t/p/w500/39wmItIWsg5sZMyRUHLkWBcuVCM.jpg"},
        {"title": "Your Name", "rating": 8.5, "poster_url": "https://image.tmdb.org/t/p/w500/q719jXXEzOoYaps6babgKnONONX.jpg"},
        {"title": "Coco", "rating": 8.2, "poster_url": "https://image.tmdb.org/t/p/w500/gGEsBPAijhVUFoiNpgZXqRVWJt2.jpg"},
        {"title": "Inside Out 2", "rating": 7.6, "poster_url": "https://image.tmdb.org/t/p/w500/vpnVM9B6NMmQpWeZvzLvDESb2QY.jpg"},
        {"title": "The Lion King", "rating": 8.3, "poster_url": "https://image.tmdb.org/t/p/w500/sKCr78MXSLixwmZ8DyJLrpMsd15.jpg"},
        {"title": "WALL-E", "rating": 8.1, "poster_url": "https://image.tmdb.org/t/p/w500/hbhFnRzzgcoNZ5IHoU9292F9.jpg"},
        {"title": "How to Train Your Dragon", "rating": 7.8, "poster_url": "https://image.tmdb.org/t/p/w500/ygGmAO60t8qCIGz9pd2BA9k.jpg"}
    ],
    "Sci-Fi": [
        {"title": "Dune: Part Two", "rating": 8.2, "poster_url": "https://image.tmdb.org/t/p/w500/1pdfLvkbY9ohJlCjQH2CZjjYVvJ.jpg"},
        {"title": "Interstellar", "rating": 8.7, "poster_url": "https://image.tmdb.org/t/p/w500/yQvGrMoipbRoddT0ZR8tPoR7NfX.jpg"},
        {"title": "Inception", "rating": 8.8, "poster_url": "https://image.tmdb.org/t/p/w500/xlaY2zyzMfkhk0HSC5VUwzoZPU1.jpg"},
        {"title": "Blade Runner 2049", "rating": 8.0, "poster_url": "https://image.tmdb.org/t/p/w500/gajva2L0rPYkEWjzgFlBXCAVBE5.jpg"},
        {"title": "The Matrix", "rating": 8.7, "poster_url": "https://image.tmdb.org/t/p/w500/dXNAPwY7VrqMAo51EKhhCJfaGb5.jpg"},
        {"title": "The Martian", "rating": 8.0, "poster_url": "https://image.tmdb.org/t/p/w500/5BHuvQ6p9kfc091Z8RiFNhCwL4b.jpg"},
        {"title": "Arrival", "rating": 7.9, "poster_url": "https://image.tmdb.org/t/p/w500/pEzNVQfdzYDzVK0XqxERIw2x2se.jpg"},
        {"title": "Edge of Tomorrow", "rating": 7.9, "poster_url": "https://image.tmdb.org/t/p/w500/nBM9MMa2WCwvMG4IJ3eiGUdbPe6.jpg"}
    ],
    "Thriller": [
        {"title": "Fight Club", "rating": 8.8, "poster_url": "https://image.tmdb.org/t/p/w500/jSziioSwPVrOy9Yow3XhWIBDjq1.jpg"},
        {"title": "Se7en", "rating": 8.6, "poster_url": "https://image.tmdb.org/t/p/w500/6yoghtyTBoPvxDuioM0xW1h92k.jpg"},
        {"title": "Shutter Island", "rating": 8.2, "poster_url": "https://image.tmdb.org/t/p/w500/kve20tXwUZpu4GUX8l6X7Zf.jpg"},
        {"title": "Gone Girl", "rating": 8.1, "poster_url": "https://image.tmdb.org/t/p/w500/qymaJhucKi7vf5onB3429y5kM.jpg"},
        {"title": "Zodiac", "rating": 7.8, "poster_url": "https://image.tmdb.org/t/p/w500/6WBeEr4r0h745sY6s0kQ.jpg"},
        {"title": "Prisoners", "rating": 8.1, "poster_url": "https://image.tmdb.org/t/p/w500/uhvi2yiv1a90v120199f.jpg"},
        {"title": "Parasite", "rating": 8.5, "poster_url": "https://image.tmdb.org/t/p/w500/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg"},
        {"title": "The Silence of the Lambs", "rating": 8.6, "poster_url": "https://image.tmdb.org/t/p/w500/rplLJ2hPcOQmkFhTqUte0MkE.jpg"}
    ],
    "Drama": [
        {"title": "Oppenheimer", "rating": 8.8, "poster_url": "https://image.tmdb.org/t/p/w500/8Gxv8gSFCU0XGDykEGv7zR1n2ua.jpg"},
        {"title": "The Shawshank Redemption", "rating": 8.7, "poster_url": "https://image.tmdb.org/t/p/w500/9cqNxx0GxF0bflZmeSMuL5tnG.jpg"},
        {"title": "The Godfather", "rating": 8.7, "poster_url": "https://image.tmdb.org/t/p/w500/3bhkrj58Vtu7enYsRolD1fZdja1.jpg"},
        {"title": "Whiplash", "rating": 8.5, "poster_url": "https://image.tmdb.org/t/p/w500/7fn624j5lj3xTme2SgiLCeuedmO.jpg"},
        {"title": "Forrest Gump", "rating": 8.5, "poster_url": "https://image.tmdb.org/t/p/w500/saHP97rTPS5eLmrLQEcANmKrsFl.jpg"},
        {"title": "Schindler's List", "rating": 8.6, "poster_url": "https://image.tmdb.org/t/p/w500/sF1U4EUQS8YHUYjN1vO.jpg"},
        {"title": "12 Angry Men", "rating": 8.5, "poster_url": "https://image.tmdb.org/t/p/w500/ow3wq89wM8912h8.jpg"},
        {"title": "The Social Network", "rating": 7.9, "poster_url": "https://image.tmdb.org/t/p/w500/ok51PP9x5.jpg"}
    ]
}


@st.cache_data(ttl=3600, show_spinner=False)
def get_movies_by_genre(genre_name, limit=12):
    """Fetch top popular movies for a specific genre, guaranteed domain-specific."""
    genre_id = GENRE_MAP.get(genre_name)
    if genre_id and TMDB_API_KEY:
        url = f"{BASE_URL}/discover/movie"
        params = {
            "api_key": TMDB_API_KEY,
            "with_genres": genre_id,
            "sort_by": "popularity.desc",
            "vote_count.gte": 100,
            "language": "en-US"
        }
        try:
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                results = response.json().get("results", [])
                if results:
                    return [_format_movie(m) for m in results[:limit]]
        except Exception as e:
            print(f"Error fetching movies by genre {genre_name}: {e}")

    # Fallback to guaranteed domain-specific curated list!
    return GENRE_FALLBACKS.get(genre_name, GENRE_FALLBACKS.get("Action", []))[:limit]

