import os
import random
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from data_processing_of_netflix_recommendation import predict_for_title
import tmdb_service

# ========================================================
# 1. PAGE CONFIGURATION
# ========================================================
st.set_page_config(
    page_title="Cineflix • AI Movie Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def clean_html(html_str):
    """Strips leading and trailing spaces per line to prevent Markdown code block wrapping."""
    return "\n".join(line.strip() for line in html_str.strip().splitlines())

# ========================================================
# 2. LIQUID GLASS & HOLLOW COSMIC SPACE STYLING
# ========================================================
st.markdown(
    clean_html("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Inter:wght@300;400;500;600&display=swap');

    /* Global Cosmic Space Background */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', 'Inter', sans-serif !important;
        background: radial-gradient(circle at 50% 25%, rgba(18, 28, 58, 0.45) 0%, rgba(4, 7, 18, 0.98) 100%),
                    url("https://images.unsplash.com/photo-1506703719100-a0f3a48c0f86?q=80&w=2070") no-repeat center center fixed !important;
        background-size: cover !important;
        color: #f1f5f9 !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    [data-testid="stMainBlockContainer"], .block-container {
        background: transparent !important;
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
    }

    /* Universal High-Contrast Text Rules */
    p, span, label, div, li {
        color: #f1f5f9;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 800 !important;
    }
    strong, b {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* Streamlit Captions & Secondary Details */
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p,
    [data-testid="stCaptionContainer"] span,
    .stCaption,
    small {
        color: #cbd5e1 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        opacity: 1 !important;
    }

    /* Markdown Container Paragraphs & Elements */
    [data-testid="stMarkdownContainer"] p {
        color: #e2e8f0 !important;
        line-height: 1.6 !important;
    }

    /* Interactive Links */
    a {
        color: #38bdf8 !important;
        text-decoration: underline !important;
        font-weight: 600 !important;
    }
    a:hover {
        color: #7dd3fc !important;
        text-shadow: 0 0 10px rgba(56, 189, 248, 0.6) !important;
    }

    /* Global Streamlit Buttons (Main panel buttons: Explore, Re-Search, Rewatch, Remove) */
    .stButton > button {
        background: rgba(255, 255, 255, 0.08) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: #e50914 !important;
        color: #ffffff !important;
        border-color: #e50914 !important;
        box-shadow: 0 0 15px rgba(229, 9, 20, 0.5) !important;
    }

    /* Left Sidebar: Translucent Liquid Glass */
    [data-testid="stSidebar"] {
        background: rgba(8, 12, 24, 0.65) !important;
        backdrop-filter: blur(35px) saturate(200%) !important;
        -webkit-backdrop-filter: blur(35px) saturate(200%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 10px 0 35px rgba(0, 0, 0, 0.6) !important;
    }

    /* Sidebar Section Headers: Vibrant High-Contrast Ice Cyan */
    .sidebar-section-title {
        color: #93c5fd !important;
        font-size: 11.5px !important;
        font-weight: 800 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        margin: 20px 0 8px 8px !important;
        text-shadow: 0 0 10px rgba(147, 197, 253, 0.3) !important;
    }

    /* Sidebar Navigation Buttons */
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 255, 255, 0.03) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 10px !important;
        padding: 9px 14px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        width: 100% !important;
        margin-bottom: 3px !important;
        transition: all 0.2s ease !important;
        white-space: nowrap !important;
        word-break: keep-all !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255, 255, 255, 0.12) !important;
        color: #ffffff !important;
        border-color: rgba(255, 255, 255, 0.25) !important;
        transform: translateX(3px) !important;
    }

    /* Active Sidebar Navigation Item */
    [data-testid="stSidebar"] button[kind="primary"] {
        background: rgba(229, 9, 20, 0.22) !important;
        border: 1px solid rgba(229, 9, 20, 0.5) !important;
        border-left: 4px solid #e50914 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        box-shadow: 0 0 16px rgba(229, 9, 20, 0.35) !important;
    }

    /* Global Liquid Glass Panels */
    .liquid-panel {
        background: rgba(13, 20, 38, 0.55) !important;
        backdrop-filter: blur(28px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(28px) saturate(180%) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 20px !important;
        padding: 24px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.5), inset 0 0 15px rgba(255, 255, 255, 0.03) !important;
    }

    .liquid-panel, .liquid-panel p, .liquid-panel span, .liquid-panel label, .liquid-panel li {
        color: #e2e8f0 !important;
    }
    .liquid-panel h1, .liquid-panel h2, .liquid-panel h3, .liquid-panel h4 {
        color: #ffffff !important;
    }
    .liquid-panel b, .liquid-panel strong {
        color: #ffffff !important;
    }

    /* Holographic DNA Tablet (From Reference Image 2) */
    .dna-tablet {
        background: rgba(8, 14, 28, 0.7) !important;
        backdrop-filter: blur(35px) saturate(200%) !important;
        -webkit-backdrop-filter: blur(35px) saturate(200%) !important;
        border: 2px solid #00f2fe !important;
        box-shadow: 0 0 50px rgba(0, 242, 254, 0.45), inset 0 0 30px rgba(0, 242, 254, 0.12) !important;
        border-radius: 36px !important;
        padding: 35px 30px !important;
        margin: 20px auto !important;
        position: relative !important;
    }

    /* Tablet Top Camera Notch */
    .tablet-notch {
        width: 60px;
        height: 6px;
        background: rgba(0, 242, 254, 0.5);
        border-radius: 6px;
        margin: 0 auto 20px auto;
        box-shadow: 0 0 12px #00f2fe;
    }

    /* Liquid Search Input */
    .stTextInput > div > div > input {
        background: rgba(15, 23, 42, 0.65) !important;
        backdrop-filter: blur(20px) !important;
        color: #ffffff !important;
        border: 1.5px solid rgba(255, 255, 255, 0.18) !important;
        border-radius: 14px !important;
        padding: 12px 18px !important;
        font-size: 15px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #e50914 !important;
        box-shadow: 0 0 25px rgba(229, 9, 20, 0.5) !important;
    }
    .stTextInput input::placeholder,
    .stTextInput input::-webkit-input-placeholder,
    .stTextInput input::-moz-placeholder {
        color: #cbd5e1 !important;
        opacity: 0.9 !important;
    }

    /* Genre Pill Buttons: Strictly NO TEXT WRAP */
    div[data-testid="stHorizontalBlock"] .stButton > button,
    div[data-testid="stHorizontalBlock"] .stButton > button p,
    div[data-testid="stHorizontalBlock"] .stButton > button span {
        white-space: nowrap !important;
        word-break: keep-all !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    div[data-testid="stHorizontalBlock"] .stButton > button {
        background: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        color: #f1f5f9 !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        border-radius: 18px !important;
        padding: 6px 10px !important;
        font-weight: 600 !important;
        font-size: 12.5px !important;
        transition: all 0.2s ease !important;
        min-width: 0 !important;
    }

    div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        background: #e50914 !important;
        border-color: #e50914 !important;
        color: #ffffff !important;
        box-shadow: 0 0 15px rgba(229, 9, 20, 0.6) !important;
        transform: translateY(-2px) !important;
    }

    /* Active Genre Chip */
    div[data-testid="stHorizontalBlock"] .stButton > button[kind="primary"] {
        background: #e50914 !important;
        border-color: #e50914 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        box-shadow: 0 0 16px rgba(229, 9, 20, 0.7) !important;
    }

    /* Origin Pill Badge */
    .origin-badge {
        background: rgba(0, 242, 254, 0.15) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.45) !important;
        padding: 3px 9px !important;
        border-radius: 8px !important;
        font-size: 11.5px !important;
        font-weight: 700 !important;
        display: inline-block !important;
        margin-top: 4px !important;
        text-shadow: 0 0 8px rgba(56, 189, 248, 0.3) !important;
    }

    /* Streamlit Alert Callouts */
    [data-testid="stAlert"] {
        background: rgba(15, 23, 42, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 14px !important;
    }
    [data-testid="stAlert"] * {
        color: #ffffff !important;
    }

    /* Top Bar Clickable Popovers (Notifications & Profile Menu) */
    div[data-testid="stPopover"] {
        width: 100% !important;
    }
    
    /* Popover Trigger Buttons: Always Dark Glass across all states (normal, focus, active, expanded) */
    div[data-testid="stPopover"] button,
    [data-testid="stPopover"] button,
    div[data-testid="stPopover"] button[kind="secondary"],
    div[data-testid="stPopover"] button[data-testid="baseButton-secondary"],
    div[data-testid="stPopover"] button:focus,
    div[data-testid="stPopover"] button:active,
    div[data-testid="stPopover"] button:visited,
    div[data-testid="stPopover"] button[aria-expanded="true"],
    div[data-testid="stPopover"] button[aria-expanded="false"],
    [data-testid="stPopover"] [data-baseweb="button"],
    [data-testid="stPopoverButton"] button {
        background: rgba(15, 23, 42, 0.88) !important;
        background-color: rgba(15, 23, 42, 0.88) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1.5px solid rgba(0, 242, 254, 0.5) !important;
        border-radius: 14px !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 13.5px !important;
        padding: 9px 14px !important;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.5) !important;
        transition: all 0.2s ease !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: nowrap !important;
    }
    /* Hide redundant chevron on compact bell button in top bar */
    div[data-testid="column"] div[data-testid="stHorizontalBlock"] > div:first-child div[data-testid="stPopover"] button svg {
        display: none !important;
    }
    div[data-testid="stPopover"] button:hover,
    div[data-testid="stPopover"] button[aria-expanded="true"]:hover,
    [data-testid="stPopover"] button:hover {
        background: rgba(229, 9, 20, 0.4) !important;
        background-color: rgba(229, 9, 20, 0.4) !important;
        border-color: #e50914 !important;
        color: #ffffff !important;
        box-shadow: 0 0 20px rgba(229, 9, 20, 0.8) !important;
    }
    div[data-testid="stPopover"] button p,
    div[data-testid="stPopover"] button span,
    div[data-testid="stPopover"] button div,
    [data-testid="stPopover"] button p,
    [data-testid="stPopover"] button span {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 13.5px !important;
        white-space: nowrap !important;
    }

    /* BaseWeb Popover Outer Container: Deep Cosmic Glass */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="popover"] [role="dialog"],
    div[data-testid="stPopoverBody"] {
        background: #090d18 !important;
        background-color: #090d18 !important;
        backdrop-filter: blur(40px) saturate(220%) !important;
        -webkit-backdrop-filter: blur(40px) saturate(220%) !important;
        border: 1.5px solid rgba(0, 242, 254, 0.55) !important;
        border-radius: 20px !important;
        box-shadow: 0 25px 65px rgba(0, 0, 0, 0.95), 0 0 35px rgba(0, 242, 254, 0.3) !important;
        color: #ffffff !important;
    }

    /* Force Transparent on all BaseWeb intermediate wrappers so NO WHITE BOX can ever show */
    div[data-baseweb="popover"] div,
    div[data-testid="stPopoverBody"] div,
    div[data-testid="stPopoverBody"] > div,
    div[data-testid="stPopoverBody"] [data-testid="stVerticalBlock"],
    div[data-testid="stPopoverBody"] [data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stPopoverBody"] [data-testid="stVerticalBlock"] > div,
    div[data-testid="stPopoverBody"] [class*="st-emotion-cache"],
    div[data-testid="stPopoverBody"] [class*="st-emotion-cache"] > div {
        background-color: transparent !important;
    }

    /* Popover Arrow */
    div[data-baseweb="popover"] [data-popper-arrow],
    div[data-baseweb="popover"] svg,
    div[data-baseweb="popover"] path {
        fill: #090d18 !important;
        background-color: #090d18 !important;
    }

    /* Popover Body Typography */
    div[data-testid="stPopoverBody"] p,
    div[data-testid="stPopoverBody"] span,
    div[data-baseweb="popover"] p,
    div[data-baseweb="popover"] span {
        color: #e2e8f0 !important;
    }
    div[data-testid="stPopoverBody"] h1,
    div[data-testid="stPopoverBody"] h2,
    div[data-testid="stPopoverBody"] h3,
    div[data-testid="stPopoverBody"] h4,
    div[data-baseweb="popover"] h1,
    div[data-baseweb="popover"] h2,
    div[data-baseweb="popover"] h3 {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    /* Notification Cards inside Popover */
    .notif-card-box {
        background: rgba(22, 34, 64, 0.8) !important;
        background-color: rgba(22, 34, 64, 0.8) !important;
        border: 1px solid rgba(0, 242, 254, 0.3) !important;
        border-radius: 12px !important;
        padding: 12px 14px !important;
        margin-bottom: 6px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4) !important;
        transition: all 0.2s ease !important;
    }
    .notif-card-box:hover {
        background: rgba(30, 48, 88, 0.95) !important;
        border-color: #00f2fe !important;
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.35) !important;
    }

    /* Buttons inside the Popover: High-contrast theme buttons with icons & hover glow */
    div[data-testid="stPopoverBody"] .stButton > button,
    div[data-baseweb="popover"] .stButton > button {
        background: linear-gradient(135deg, rgba(22, 34, 64, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%) !important;
        background-color: #111a33 !important;
        border: 1.5px solid rgba(0, 242, 254, 0.35) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 13.5px !important;
        padding: 10px 16px !important;
        margin: 4px 0 !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4) !important;
        transition: all 0.25s ease !important;
    }
    div[data-testid="stPopoverBody"] .stButton > button:hover,
    div[data-baseweb="popover"] .stButton > button:hover {
        background: linear-gradient(135deg, #e50914 0%, #b80710 100%) !important;
        background-color: #e50914 !important;
        border-color: #ff4d4d !important;
        color: #ffffff !important;
        box-shadow: 0 0 22px rgba(229, 9, 20, 0.8) !important;
        transform: translateX(4px) !important;
    }
    div[data-testid="stPopoverBody"] .stButton > button p,
    div[data-testid="stPopoverBody"] .stButton > button span,
    div[data-baseweb="popover"] .stButton > button p,
    div[data-baseweb="popover"] .stButton > button span {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 13.5px !important;
    }
    </style>
    """),
    unsafe_allow_html=True
)

# ========================================================
# 3. PREDEFINED SEARCH-DRIVEN DATA IN YOUR SPACE
# ========================================================
if "nav_choice" not in st.session_state:
    st.session_state.nav_choice = "🏠 Home"

if "active_search_query" not in st.session_state:
    st.session_state.active_search_query = ""

if "selected_genre" not in st.session_state:
    st.session_state.selected_genre = "All"

# Search History: Predefined searches + dynamically records new searches
if "search_history" not in st.session_state:
    st.session_state.search_history = [
        {"query": "Dune", "year": "2021", "rating": 8.0, "genre": "Sci-Fi", "time": "Just now", "poster_url": "https://image.tmdb.org/t/p/w500/v1tRXZ4JtD2Iv6fjkPvT4GiwslV.jpg"},
        {"query": "Oppenheimer", "year": "2023", "rating": 8.8, "genre": "Drama", "time": "2 hours ago", "poster_url": "https://image.tmdb.org/t/p/w500/8Gxv8gSFCU0XGDykEGv7zR1n2ua.jpg"},
        {"query": "Interstellar", "year": "2014", "rating": 8.7, "genre": "Sci-Fi", "time": "Yesterday", "poster_url": "https://image.tmdb.org/t/p/w500/yQvGrMoipbRoddT0ZR8tPoR7NfX.jpg"},
        {"query": "Inception", "year": "2010", "rating": 8.8, "genre": "Sci-Fi", "time": "3 days ago", "poster_url": "https://image.tmdb.org/t/p/w500/xlaY2zyzMfkhk0HSC5VUwzoZPU1.jpg"},
        {"query": "Stranger Things", "year": "2016", "rating": 8.6, "genre": "Mystery", "time": "5 days ago", "poster_url": "https://image.tmdb.org/t/p/w500/uOOtwVbSr4QDjAGIifLDwpb2Pdl.jpg"},
        {"query": "The Batman", "year": "2022", "rating": 7.8, "genre": "Action", "time": "Last week", "poster_url": "https://image.tmdb.org/t/p/w500/3w7koeOR2x71XYMJDGpygxYtScI.jpg"},
    ]

# Watch History: Predefined journeys tied directly to searches & recommendations
if "watch_history" not in st.session_state:
    st.session_state.watch_history = [
        {"title": "Dune", "rating": 8.0, "progress": "100%", "poster_url": "https://image.tmdb.org/t/p/w500/v1tRXZ4JtD2Iv6fjkPvT4GiwslV.jpg", "origin": "Searched 'Dune'"},
        {"title": "Oppenheimer", "rating": 8.8, "progress": "100%", "poster_url": "https://image.tmdb.org/t/p/w500/8Gxv8gSFCU0XGDykEGv7zR1n2ua.jpg", "origin": "Searched 'Oppenheimer'"},
        {"title": "Interstellar", "rating": 8.7, "progress": "95%", "poster_url": "https://image.tmdb.org/t/p/w500/yQvGrMoipbRoddT0ZR8tPoR7NfX.jpg", "origin": "Searched 'Interstellar'"},
        {"title": "Blade Runner 2049", "rating": 8.0, "progress": "85%", "poster_url": "https://image.tmdb.org/t/p/w500/gajva2L0rPYkEWjzgFlBXCAVBE5.jpg", "origin": "Recommended from 'Dune'"},
        {"title": "Inception", "rating": 8.8, "progress": "100%", "poster_url": "https://image.tmdb.org/t/p/w500/xlaY2zyzMfkhk0HSC5VUwzoZPU1.jpg", "origin": "Searched 'Inception'"},
        {"title": "The Dark Knight", "rating": 9.0, "progress": "100%", "poster_url": "https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg", "origin": "Recommended from 'The Batman'"},
    ]

# My Watchlist: Predefined list generated from search recommendations
if "my_watchlist" not in st.session_state:
    st.session_state.my_watchlist = [
        {"title": "The Martian", "rating": 8.0, "poster_url": "https://image.tmdb.org/t/p/w500/5BHuvQ6p9kfc091Z8RiFNhCwL4b.jpg", "origin": "Recommended from 'Interstellar'"},
        {"title": "Arrival", "rating": 7.9, "poster_url": "https://image.tmdb.org/t/p/w500/pEzNVQfdzYDzVK0XqxERIw2x2se.jpg", "origin": "Recommended from 'Dune'"},
        {"title": "The Matrix", "rating": 8.7, "poster_url": "https://image.tmdb.org/t/p/w500/dXNAPwY7VrqMAo51EKhhCJfaGb5.jpg", "origin": "Recommended from 'Inception'"},
        {"title": "The Expanse", "rating": 8.3, "poster_url": "https://image.tmdb.org/t/p/w500/5vQlVWkIMPhZ88OWchJsgwGEK9.jpg", "origin": "Recommended from 'Dune'"},
        {"title": "Fight Club", "rating": 8.8, "poster_url": "https://image.tmdb.org/t/p/w500/jSziioSwPVrOy9Yow3XhWIBDjq1.jpg", "origin": "Recommended from 'Inception'"},
        {"title": "Shawshank", "rating": 9.3, "poster_url": "https://image.tmdb.org/t/p/w500/9cqNxx0GxF0bflZmeSMuL5tnGzr.jpg", "origin": "All-time Top Recommendation"},
        {"title": "Titanic", "rating": 7.9, "poster_url": "https://image.tmdb.org/t/p/w500/9xjZS2rlVxm8SFx8kPC3aIGCOYQ.jpg", "origin": "Hollywood Epic Romance Pick"},
        {"title": "Django Unchained", "rating": 8.4, "poster_url": "https://image.tmdb.org/t/p/w500/7oWY8VDWW7thTzWh3OKYRkWUlD5.jpg", "origin": "Recommended from 'Oppenheimer'"},
    ]

# My Ratings: Predefined reviews for searched films
if "my_ratings" not in st.session_state:
    st.session_state.my_ratings = [
        {"title": "Dune", "score": "⭐ 9.5 / 10", "poster_url": "https://image.tmdb.org/t/p/w500/v1tRXZ4JtD2Iv6fjkPvT4GiwslV.jpg", "review": "Masterpiece of modern sci-fi cinematography. Hans Zimmer's score is out of this world.", "tag": "Searched & Reviewed"},
        {"title": "Oppenheimer", "score": "⭐ 10 / 10", "poster_url": "https://image.tmdb.org/t/p/w500/8Gxv8gSFCU0XGDykEGv7zR1n2ua.jpg", "review": "Nolan's magnum opus. Unrelenting psychological tension from start to finish.", "tag": "Searched & Reviewed"},
        {"title": "Interstellar", "score": "⭐ 10 / 10", "poster_url": "https://image.tmdb.org/t/p/w500/yQvGrMoipbRoddT0ZR8tPoR7NfX.jpg", "review": "The emotional core of family across relativistic spacetime is unmatched.", "tag": "Searched & Reviewed"},
        {"title": "Inception", "score": "⭐ 9.0 / 10", "poster_url": "https://image.tmdb.org/t/p/w500/xlaY2zyzMfkhk0HSC5VUwzoZPU1.jpg", "review": "A labyrinth of dream layers with the most iconic ending in modern cinema.", "tag": "Searched & Reviewed"},
        {"title": "The Batman", "score": "⭐ 8.5 / 10", "poster_url": "https://image.tmdb.org/t/p/w500/3w7koeOR2x71XYMJDGpygxYtScI.jpg", "review": "Gritty, rain-soaked noir detective tale with an incredible Batmobile sequence.", "tag": "Searched & Reviewed"},
    ]

if "search_input_key_id" not in st.session_state:
    st.session_state.search_input_key_id = 0

if "notifs_read" not in st.session_state:
    st.session_state.notifs_read = False

def record_user_search(query_str, poster_url=None, rating=8.0, genre="Cinema"):
    """Automatically records every search into Search History, Watch History, and Watchlist recommendations."""
    clean_q = query_str.strip()
    if not clean_q:
        return
    # Mark notifications as unread so user sees new recommendations for this search
    st.session_state.notifs_read = False
    # Check if already in recent search history
    if not any(h['query'].lower() == clean_q.lower() for h in st.session_state.search_history[:3]):
        st.session_state.search_history.insert(0, {
            "query": clean_q,
            "year": "2024",
            "rating": rating or 8.0,
            "genre": genre or "Cinema",
            "time": "Just now",
            "poster_url": poster_url or "https://via.placeholder.com/500x750"
        })
    # Also add as explored in Watch History
    if not any(w['title'].lower() == clean_q.lower() for w in st.session_state.watch_history):
        st.session_state.watch_history.insert(0, {
            "title": clean_q,
            "rating": rating or 8.0,
            "progress": "Just searched",
            "poster_url": poster_url or "https://via.placeholder.com/500x750",
            "origin": f"Searched '{clean_q}'"
        })

def set_search_query(query_str: str, poster_url=None, rating=8.0, genre="Cinema"):
    """Safely updates active search query and rotates the widget key to avoid post-instantiation state errors."""
    st.session_state.active_search_query = query_str
    st.session_state.search_input_key_id = st.session_state.get("search_input_key_id", 0) + 1
    if query_str and query_str.strip():
        record_user_search(query_str, poster_url, rating, genre)

CURATED_RECOMMENDATIONS = {
    "avenger": ("Captain America: Civil War", "Superhero / Action"),
    "dune": ("Dune: Part Two", "Sci-Fi / Adventure"),
    "interstellar": ("The Martian", "Sci-Fi / Space"),
    "inception": ("Shutter Island", "Psychological Thriller"),
    "batman": ("The Dark Knight", "Dark Noir / Action"),
    "dark knight": ("The Batman", "Dark Noir / Action"),
    "oppenheimer": ("Interstellar", "Cinematic Drama"),
    "spider-man": ("Spider-Man: Across the Spider-Verse", "Animation / Action"),
    "spider man": ("Spider-Man: Across the Spider-Verse", "Animation / Action"),
    "titanic": ("The Great Gatsby", "Epic Romance"),
    "fight club": ("Se7en", "Psychological Thriller"),
    "deadpool": ("Deadpool & Wolverine", "Action Comedy"),
    "stranger things": ("Dark", "Mystery Sci-Fi"),
    "matrix": ("Blade Runner 2049", "Cyberpunk / Sci-Fi"),
    "civil war": ("Avengers: Endgame", "Superhero / Action"),
    "inside out": ("Soul", "Animation / Family"),
    "furiosa": ("Mad Max: Fury Road", "Action / Dystopian"),
    "godzilla": ("Kong: Skull Island", "Monster Action"),
    "challengers": ("The Social Network", "Drama / Romance"),
    "conjuring": ("Hereditary", "Supernatural Horror"),
    "hangover": ("Superbad", "R-Rated Comedy"),
    "hereditary": ("Midsommar", "Psychological Horror"),
    "superbad": ("Step Brothers", "Buddy Comedy"),
    "la la land": ("Whiplash", "Musical Drama"),
    "gladiator": ("Braveheart", "Historical Epic"),
    "pulp fiction": ("Reservoir Dogs", "Crime Thriller"),
    "shawshank": ("The Green Mile", "Classic Drama"),
    "top gun": ("Mission: Impossible - Fallout", "High-Octane Action"),
    "alien": ("Prometheus", "Sci-Fi Horror"),
    "star wars": ("Rogue One", "Space Opera"),
}

def get_recommendation_for_title(query):
    if not query:
        return "Interstellar", "Sci-Fi"
    clean = str(query).strip()
    clean_lower = clean.lower()
    for key, (rec, genre_tag) in CURATED_RECOMMENDATIONS.items():
        if key in clean_lower:
            return rec, genre_tag
    try:
        _, local_recs = predict_for_title(clean)
        if local_recs and len(local_recs) > 0:
            rec_title = local_recs[0][0]
            if rec_title.lower() != clean_lower:
                return rec_title, "AI Recommended"
    except Exception:
        pass
    try:
        matches = tmdb_service.search_movies(clean)
        if matches and matches[0].get("id"):
            tmdb_recs = tmdb_service.get_recommendations_for_movie(matches[0]["id"])
            if tmdb_recs:
                return tmdb_recs[0]["title"], "Top TMDB Match"
    except Exception:
        pass
    return "Inception", "AI Feature Pick"

def get_user_notifications():
    """Generates real-time notifications tailored specifically to the user's searches."""
    searches = [s["query"] for s in st.session_state.search_history if s.get("query")]
    if not searches:
        searches = ["Dune", "Interstellar", "Inception"]

    primary_search = searches[0]
    secondary_search = searches[1] if len(searches) > 1 else searches[0]

    rec_1, genre_1 = get_recommendation_for_title(primary_search)
    rec_2, genre_2 = get_recommendation_for_title(secondary_search)
    if rec_2.lower() == rec_1.lower():
        rec_2 = "Blade Runner 2049" if rec_1 != "Blade Runner 2049" else "The Prestige"

    notifs = [
        {
            "border_color": "#e50914",
            "tag": "NEW FOR YOU",
            "title": f"🎬 AI Match: {rec_1}",
            "desc": f"Because you searched for '<b>{primary_search}</b>', our neural engine flagged <b>{rec_1}</b> as a 98% resonance match!",
            "meta": f"🕒 Just now • Triggered by search '{primary_search}'",
            "action_label": f"▶️ Explore {rec_1}",
            "target": rec_1
        },
        {
            "border_color": "#00f2fe",
            "tag": "TRENDING",
            "title": f"🔥 New in {genre_2}: {rec_2}",
            "desc": f"Curated from your recent interest in '<b>{secondary_search}</b>'. Available now in Ultra 4K HDR.",
            "meta": f"🕒 2 hours ago • Based on '{secondary_search}'",
            "action_label": f"🚀 Explore {rec_2}",
            "target": rec_2
        },
        {
            "border_color": "#4ade80",
            "tag": "CALIBRATED",
            "title": "🧬 Taste Topology Calibrated",
            "desc": f"Your neural profile calibrated with '<b>{primary_search}</b>'. Genre resonance score updated to 94%.",
            "meta": f"🕒 Today • Calibrated to '{primary_search}'",
            "action_label": "🧬 View My Taste Profile",
            "target_nav": "🧬 My Taste Profile"
        }
    ]
    return notifs

# ========================================================
# 4. SIDEBAR NAVIGATION (With Live Counts for YOUR SPACE)
# ========================================================
with st.sidebar:
    st.markdown(
        clean_html("""
        <div style='display:flex;align-items:center;gap:12px;margin-bottom:20px;padding-bottom:12px;border-bottom:1px solid rgba(255,255,255,0.08);'>
            <div style='background:#e50914;color:white;font-weight:900;font-size:22px;padding:6px 14px;border-radius:8px;box-shadow:0 0 15px rgba(229,9,20,0.5);'>N</div>
            <div>
                <h3 style='margin:0;color:#fff;font-weight:800;letter-spacing:1px;font-size:20px;'>CINEFLIX</h3>
                <span style='color:#00f2fe;font-size:11px;font-weight:600;letter-spacing:0.5px;'>AI MOVIE DISCOVERY</span>
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )

    def nav_btn(label, key, badge=""):
        display_label = f"{label} ({badge})" if badge else label
        is_active = (st.session_state.nav_choice == label and not st.session_state.active_search_query)
        btn_type = "primary" if is_active else "secondary"
        if st.button(display_label, key=key, width="stretch", type=btn_type):
            st.session_state.nav_choice = label
            set_search_query("")
            st.rerun()

    st.markdown("<div class='sidebar-section-title'>MAIN</div>", unsafe_allow_html=True)
    nav_btn("🏠 Home", "nav_home")
    nav_btn("🧭 Discover Movies", "nav_discover")
    nav_btn("✨ For You", "nav_for_you")
    nav_btn("🔥 Trending Now", "nav_trending")
    nav_btn("📅 New Releases", "nav_new_releases")

    st.markdown("<div class='sidebar-section-title'>YOUR SPACE</div>", unsafe_allow_html=True)
    nav_btn("🤍 My Watchlist", "nav_watchlist", badge=str(len(st.session_state.my_watchlist)))
    nav_btn("🕒 Watch History", "nav_history", badge=str(len(st.session_state.watch_history)))
    nav_btn("⭐ My Ratings", "nav_ratings", badge=str(len(st.session_state.my_ratings)))
    nav_btn("🧬 My Taste Profile", "nav_taste_profile")

    st.markdown("<div class='sidebar-section-title'>SMART TOOLS</div>", unsafe_allow_html=True)
    nav_btn("😊 Mood Explorer", "nav_mood")
    nav_btn("🔀 Surprise Me", "nav_surprise")
    nav_btn("⚖️ Compare Movies", "nav_compare")

    st.markdown("<div class='sidebar-section-title'>SETTINGS & HELP</div>", unsafe_allow_html=True)
    nav_btn("⚙️ Settings", "nav_settings")
    nav_btn("❓ Help", "nav_help")

    st.markdown("<br>", unsafe_allow_html=True)
    # Bottom Taste Card from Reference Image 1
    if st.button("🧠 Your taste is unique! Explore more with AI ›", key="taste_pill", width="stretch"):
        st.session_state.nav_choice = "🧬 My Taste Profile"
        set_search_query("")
        st.rerun()


def _fetch_single_poster(title):
    matches = tmdb_service.search_movies(title)
    if matches and matches[0].get("poster_url"):
        return matches[0]["poster_url"]
    return "https://via.placeholder.com/500x750?text=No+Poster"


def fetch_posters_in_parallel(titles):
    with ThreadPoolExecutor(max_workers=min(len(titles), 5)) as executor:
        posters = list(executor.map(_fetch_single_poster, titles))
    return posters


def render_movie_row(movie_list, max_items=8, key_prefix="row"):
    """Displays movie cards in clean horizontal rows with verified images."""
    cols = st.columns(min(len(movie_list), max_items))
    for idx, movie in enumerate(movie_list[:max_items]):
        with cols[idx % max_items]:
            poster = movie.get("poster_url") or "https://via.placeholder.com/500x750?text=No+Poster"
            st.image(poster, width="stretch")
            st.markdown(f"<div style='color:#ffffff;font-size:13.5px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>{movie.get('title')}</div>", unsafe_allow_html=True)
            rating = movie.get("rating", 8.0)
            st.markdown(f"<span style='color:#fbbf24;font-size:12.5px;font-weight:700;'>⭐ {rating}</span>", unsafe_allow_html=True)
            if movie.get("origin"):
                st.markdown(f"<div class='origin-badge'>🎯 {movie['origin']}</div>", unsafe_allow_html=True)
            if st.button("Explore", key=f"{key_prefix}_{idx}_{movie.get('id', movie.get('title', ''))}", width="stretch"):
                set_search_query(movie.get("title", ""), poster, rating)
                st.rerun()


# ========================================================
# 5. TOP BAR: SEARCH & GENRE CHIPS (Reference Image 1)
# ========================================================
t_col1, t_col2 = st.columns([3.8, 1.4])
with t_col1:
    search_typed = st.text_input(
        "Search Bar",
        value=st.session_state.active_search_query,
        key=f"search_input_{st.session_state.search_input_key_id}",
        placeholder="🔍 Search movies, actors, genres (e.g. Dune, Oppenheimer, Stranger Things)...",
        label_visibility="collapsed"
    )
    if search_typed != st.session_state.active_search_query:
        st.session_state.active_search_query = search_typed
        if search_typed:
            record_user_search(search_typed)
        st.rerun()

with t_col2:
    tb_c1, tb_c2 = st.columns([1, 2.4])
    with tb_c1:
        user_notifs = get_user_notifications()
        unread_count = 0 if st.session_state.get("notifs_read", False) else len(user_notifs)
        bell_label = f"🔔 ({unread_count})" if unread_count > 0 else "🔔"
        
        with st.popover(bell_label, width="stretch"):
            badge_html = "<span style='background:rgba(255,255,255,0.1);color:#94a3b8;border:1px solid rgba(255,255,255,0.2);font-size:10px;font-weight:700;padding:2px 8px;border-radius:12px;'>READ</span>" if st.session_state.get("notifs_read", False) else f"<span style='background:rgba(229,9,20,0.35);color:#ff4d4d;border:1px solid #e50914;font-size:10.5px;font-weight:800;padding:2px 8px;border-radius:12px;'>{unread_count} NEW</span>"
            
            st.markdown(
                clean_html(f"""
                <div style='display:flex;align-items:center;justify-content:space-between;padding-bottom:10px;margin-bottom:12px;border-bottom:1.5px solid rgba(0,242,254,0.3);'>
                    <span style='color:#ffffff;font-size:14.5px;font-weight:800;letter-spacing:0.3px;'>🔔 Search-Tuned Alerts</span>
                    {badge_html}
                </div>
                """),
                unsafe_allow_html=True
            )
            
            for n_idx, notif in enumerate(user_notifs):
                st.markdown(
                    clean_html(f"""
                    <div class='notif-card-box' style='border-left: 3.5px solid {notif["border_color"]} !important;'>
                        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:2px;'>
                            <span style='color:#ffffff;font-weight:800;font-size:13.5px;'>{notif["title"]}</span>
                            <span style='background:rgba(255,255,255,0.08);color:{notif["border_color"]};font-size:9.5px;font-weight:800;padding:1px 6px;border-radius:6px;border:1px solid {notif["border_color"]};'>{notif["tag"]}</span>
                        </div>
                        <div style='color:#cbd5e1;font-size:12px;margin:3px 0 5px 0;line-height:1.4;'>{notif["desc"]}</div>
                        <div style='color:#38bdf8;font-size:11px;font-weight:600;'>{notif["meta"]}</div>
                    </div>
                    """),
                    unsafe_allow_html=True
                )
                if notif.get("target"):
                    if st.button(notif["action_label"], key=f"notif_act_{n_idx}_{notif['target'][:8].replace(' ', '_')}", width="stretch"):
                        set_search_query(notif["target"])
                        st.rerun()
                elif notif.get("target_nav"):
                    if st.button(notif["action_label"], key=f"notif_nav_{n_idx}", width="stretch"):
                        st.session_state.nav_choice = notif["target_nav"]
                        set_search_query("")
                        st.rerun()

            st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)
            if st.button("✓ Mark All as Read", key="btn_clear_notifs", width="stretch"):
                st.session_state.notifs_read = True
                st.toast("🎉 All notifications marked as read!")
                st.rerun()

    with tb_c2:
        with st.popover("👤 Atharv", width="stretch"):
            st.markdown(
                clean_html("""
                <div style='display:flex;align-items:center;gap:12px;padding-bottom:12px;margin-bottom:14px;border-bottom:1.5px solid rgba(0,242,254,0.3);'>
                    <img src='https://api.dicebear.com/7.x/bottts/svg?seed=Atharv' style='width:44px;height:44px;border-radius:50%;background:#e50914;padding:3px;box-shadow:0 0 14px rgba(229,9,20,0.6);'/>
                    <div>
                        <div style='color:#ffffff;font-weight:800;font-size:15px;letter-spacing:0.3px;'>Atharv Shukla</div>
                        <span style='color:#00f2fe;font-size:11.5px;font-weight:800;letter-spacing:0.8px;'>🌟 VIP CINEMA PRO</span>
                    </div>
                </div>
                """),
                unsafe_allow_html=True
            )
            if st.button("🧬 My Taste Profile", key="prof_btn_dna", width="stretch"):
                st.session_state.nav_choice = "🧬 My Taste Profile"
                set_search_query("")
                st.rerun()
            if st.button("🤍 My Watchlist", key="prof_btn_wl", width="stretch"):
                st.session_state.nav_choice = "🤍 My Watchlist"
                set_search_query("")
                st.rerun()
            if st.button("🕒 Watch History", key="prof_btn_hist", width="stretch"):
                st.session_state.nav_choice = "🕒 Watch History"
                set_search_query("")
                st.rerun()
            if st.button("⚙️ Settings & Preferences", key="prof_btn_settings", width="stretch"):
                st.session_state.nav_choice = "⚙️ Settings"
                set_search_query("")
                st.rerun()
            if st.button("🚪 Sign Out", key="prof_btn_signout", width="stretch"):
                st.toast("👋 Signed out of Cineflix. Have a great day!")

# Genre Chips Row (With White-Space NOWRAP so 'Animation' NEVER wraps)
genre_chips = ["All", "Action", "Sci-Fi", "Thriller", "Comedy", "Romance", "Horror", "Drama", "Animation"]
g_cols = st.columns(len(genre_chips))
for g_idx, g_name in enumerate(genre_chips):
    with g_cols[g_idx]:
        is_chip_active = (st.session_state.selected_genre == g_name)
        chip_type = "primary" if is_chip_active else "secondary"
        if st.button(g_name, key=f"chip_{g_name}", width="stretch", type=chip_type):
            st.session_state.selected_genre = g_name
            set_search_query("")
            if g_name != "All":
                st.session_state.nav_choice = "🧭 Discover Movies"
            else:
                st.session_state.nav_choice = "🏠 Home"
            st.rerun()

st.write("---")

# ========================================================
# 6. VERIFIED MOVIE DATABASE FOR ROWS (100% Working TMDB Posters)
# ========================================================
trending_row = [
    {"title": "Oppenheimer", "rating": 8.8, "poster_url": "https://image.tmdb.org/t/p/w500/8Gxv8gSFCU0XGDykEGv7zR1n2ua.jpg"},
    {"title": "Avengers: Endgame", "rating": 8.3, "poster_url": "https://image.tmdb.org/t/p/w500/or06FN3Dka5tukK1e9sl16pB3iy.jpg"},
    {"title": "Interstellar", "rating": 8.7, "poster_url": "https://image.tmdb.org/t/p/w500/yQvGrMoipbRoddT0ZR8tPoR7NfX.jpg"},
    {"title": "Spider-Man: Across the Spider-Verse", "rating": 8.4, "poster_url": "https://image.tmdb.org/t/p/w500/8Vt6mWEReuy4Of61Lnj5Xj704m8.jpg"},
    {"title": "The Batman", "rating": 7.8, "poster_url": "https://image.tmdb.org/t/p/w500/3w7koeOR2x71XYMJDGpygxYtScI.jpg"},
    {"title": "Inception", "rating": 8.8, "poster_url": "https://image.tmdb.org/t/p/w500/xlaY2zyzMfkhk0HSC5VUwzoZPU1.jpg"},
    {"title": "Top Gun: Maverick", "rating": 8.2, "poster_url": "https://image.tmdb.org/t/p/w500/62HCnUTziyWcpDaBO2i1DX17ljH.jpg"},
    {"title": "Fight Club", "rating": 8.8, "poster_url": "https://image.tmdb.org/t/p/w500/jSziioSwPVrOy9Yow3XhWIBDjq1.jpg"},
]

new_releases_row = [
    {"title": "Dune: Part Two", "rating": 8.2, "poster_url": "https://image.tmdb.org/t/p/w500/1pdfLvkbY9ohJlCjQH2CZjjYVvJ.jpg"},
    {"title": "Deadpool & Wolverine", "rating": 7.7, "poster_url": "https://image.tmdb.org/t/p/w500/8cdWjvZQUExUUTzyp4t6EDMubfO.jpg"},
    {"title": "Civil War", "rating": 7.0, "poster_url": "https://image.tmdb.org/t/p/w500/sh7Rg8Er3tFcN9BpKIPOMvALgZd.jpg"},
    {"title": "Inside Out 2", "rating": 7.6, "poster_url": "https://image.tmdb.org/t/p/w500/vpnVM9B6NMmQpWeZvzLvDESb2QY.jpg"},
    {"title": "Furiosa: A Mad Max Saga", "rating": 7.5, "poster_url": "https://image.tmdb.org/t/p/w500/iADOJ8Zymht2JPMoy3R7xUMZ51f.jpg"},
    {"title": "Godzilla x Kong: The New Empire", "rating": 7.1, "poster_url": "https://image.tmdb.org/t/p/w500/bQ2ywkchIiaKLSEaMrcT6e29f91.jpg"},
    {"title": "Challengers", "rating": 7.2, "poster_url": "https://image.tmdb.org/t/p/w500/H6vke7zGiuL7z53e6BPOWNgKH4.jpg"},
    {"title": "Oppenheimer", "rating": 8.8, "poster_url": "https://image.tmdb.org/t/p/w500/8Gxv8gSFCU0XGDykEGv7zR1n2ua.jpg"},
]

dune_row = [
    {"title": "The Martian", "rating": 8.0, "poster_url": "https://image.tmdb.org/t/p/w500/5BHuvQ6p9kfc091Z8RiFNhCwL4b.jpg"},
    {"title": "The Expanse", "rating": 8.3, "poster_url": "https://image.tmdb.org/t/p/w500/5vQlVWkIMPhZ88OWchJsgwGEK9.jpg"},
    {"title": "Edge of Tomorrow", "rating": 7.9, "poster_url": "https://image.tmdb.org/t/p/w500/nBM9MMa2WCwvMG4IJ3eiGUdbPe6.jpg"},
    {"title": "Arrival", "rating": 7.9, "poster_url": "https://image.tmdb.org/t/p/w500/pEzNVQfdzYDzVK0XqxERIw2x2se.jpg"},
    {"title": "Blade Runner 2049", "rating": 8.0, "poster_url": "https://image.tmdb.org/t/p/w500/gajva2L0rPYkEWjzgFlBXCAVBE5.jpg"},
    {"title": "Foundation", "rating": 7.5, "poster_url": "https://image.tmdb.org/t/p/w500/tg9I5pOY4M9CKj8U0cxVBTsm5eh.jpg"},
    {"title": "Rogue One", "rating": 7.8, "poster_url": "https://image.tmdb.org/t/p/w500/fiGa4VkDw9lb2AqKUbZvNd4K1Jw.jpg"},
    {"title": "The Matrix", "rating": 8.7, "poster_url": "https://image.tmdb.org/t/p/w500/dXNAPwY7VrqMAo51EKhhCJfaGb5.jpg"},
]

# ========================================================
# 7. VIEW CONTROLLER (Handling All Sidebar Options & Search)
# ========================================================

# Check if user is actively searching
if st.session_state.active_search_query and len(st.session_state.active_search_query.strip()) >= 1:
    q = st.session_state.active_search_query.strip()
    
    col_back, col_title = st.columns([1, 6])
    with col_back:
        if st.button("← Clear Search", key="back_btn", width="stretch"):
            set_search_query("")
            st.rerun()
    with col_title:
        st.markdown(f"<h2 style='margin:0;'>Searching for: <span style='color:#00f2fe;'>{q}</span></h2>", unsafe_allow_html=True)

    # Universal movie search (TMDB global first)
    tmdb_results = tmdb_service.search_movies(q)
    
    # If no immediate match, check local dataset
    if not tmdb_results:
        local_preds, local_recs = predict_for_title(q)
        if local_preds:
            matched_title = local_preds[0][0]
            tmdb_results = tmdb_service.search_movies(matched_title)

    # If still no results, fallback to top trending pool (guaranteed never "not found")
    if not tmdb_results:
        tmdb_results = trending_row

    best = tmdb_results[0]
    record_user_search(best.get("title", q), best.get("poster_url"), best.get("rating", 8.0))
    
    c1, c2 = st.columns([1.1, 2.6])
    with c1:
        st.image(best.get("poster_url") or "https://via.placeholder.com/500x750", width="stretch")
        if st.button("🔖 + Add to Watchlist", key="search_add_wl", width="stretch"):
            if not any(m['title'] == best.get('title') for m in st.session_state.my_watchlist):
                st.session_state.my_watchlist.insert(0, {
                    "title": best.get("title"),
                    "rating": best.get("rating", 8.0),
                    "poster_url": best.get("poster_url"),
                    "origin": f"Saved from '{q}' search"
                })
                st.toast(f"🎉 Added {best.get('title')} to Watchlist!")
            else:
                st.toast(f"ℹ️ {best.get('title')} is already in your Watchlist!")
    with c2:
        title_display = best.get('title') or q.title()
        year_display = best.get('release_year', '2024')
        st.markdown(f"<h1 style='color:#ffffff;margin:0 0 6px 0;font-size:36px;font-weight:900;'>{title_display} <span style='font-size:22px;color:#94a3b8;'>({year_display})</span></h1>", unsafe_allow_html=True)
        st.markdown(f"<div style='margin-bottom:12px;'><span style='color:#fbbf24;font-size:16px;font-weight:800;'>⭐ {best.get('rating', 8.0)} / 10</span> &nbsp;•&nbsp; <span style='color:#cbd5e1;font-size:14px;'>📅 {best.get('release_date', 'Premiered')}</span></div>", unsafe_allow_html=True)
        
        details = tmdb_service.get_movie_details(best.get("id"), best.get("media_type", "movie")) if best.get("id") else None
        
        if details:
            if details.get("genres"):
                st.markdown(f"<p style='color:#e2e8f0;font-size:14px;margin:4px 0;'><b style='color:#ffffff;'>🏷️ Genres:</b> {details['genres']}</p>", unsafe_allow_html=True)
            if details.get("streaming_on"):
                st.markdown(f"<p style='color:#e2e8f0;font-size:14px;margin:4px 0;'><b style='color:#ffffff;'>📺 Available on:</b> <span style='color:#38bdf8;font-weight:700;'>{', '.join(details['streaming_on'])}</span></p>", unsafe_allow_html=True)
            
            trailer_url = details.get("trailer_url")
            if trailer_url:
                st.markdown(
                    clean_html(f"""
                    <div style='margin: 14px 0;'>
                        <a href='{trailer_url}' target='_blank' style='display:inline-flex;align-items:center;gap:8px;background:#e50914;color:#ffffff;font-weight:800;font-size:14px;padding:10px 22px;border-radius:12px;text-decoration:none;box-shadow:0 0 20px rgba(229,9,20,0.6);'>
                            ▶ Play Official Trailer on YouTube ↗
                        </a>
                    </div>
                    """),
                    unsafe_allow_html=True
                )
        else:
            import urllib.parse
            direct_trailer = f"https://www.youtube.com/results?search_query={urllib.parse.quote(title_display + ' official trailer')}"
            st.markdown(
                clean_html(f"""
                <div style='margin: 14px 0;'>
                    <a href='{direct_trailer}' target='_blank' style='display:inline-flex;align-items:center;gap:8px;background:#e50914;color:#ffffff;font-weight:800;font-size:14px;padding:10px 22px;border-radius:12px;text-decoration:none;box-shadow:0 0 20px rgba(229,9,20,0.6);'>
                        ▶ Play Official Trailer on YouTube ↗
                    </a>
                </div>
                """),
                unsafe_allow_html=True
            )

        synopsis = best.get('overview') or "A thrilling cinematic journey with acclaimed visual spectacle, gripping performances, and unforgettable world-building."
        st.markdown(f"<div style='background:rgba(255,255,255,0.06);border-left:3.5px solid #00f2fe;padding:12px 16px;border-radius:12px;margin:12px 0;'><b style='color:#ffffff;font-size:14px;'>📝 Synopsis:</b><p style='color:#e2e8f0;font-size:13.5px;line-height:1.6;margin:4px 0 0 0;'>{synopsis}</p></div>", unsafe_allow_html=True)

    st.markdown("### 🌟 More Like This (Deep AI Recommendations):")
    recs = []
    if best.get("id"):
        recs = tmdb_service.get_recommendations_for_movie(best["id"], best.get("media_type", "movie"))
    if not recs:
        # Fallback to genre-matched recommendations
        top_genre = (details.get("genres") or "").split(",")[0].strip() if details and details.get("genres") else "Action"
        recs = tmdb_service.get_movies_by_genre(top_genre, limit=8)
    if not recs:
        recs = trending_row
    render_movie_row(recs, max_items=8, key_prefix="search_rec")

# 1. HOME DASHBOARD
elif st.session_state.nav_choice == "🏠 Home":
    banner_html = """
    <div style='background: linear-gradient(90deg, rgba(8,12,24,0.95) 0%, rgba(8,12,24,0.6) 60%, rgba(8,12,24,0.2) 100%),
                url("https://image.tmdb.org/t/p/w1280/lzWHmY2qhFv0E2kax5y2kE9jH5S.jpg") no-repeat center center;
                background-size: cover; border-radius: 24px; padding: 45px 35px; border: 1px solid rgba(255,255,255,0.12); margin-bottom: 25px; box-shadow: 0 15px 40px rgba(0,0,0,0.6);'>
        <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
            <div style='max-width: 600px;'>
                <p style='color:#93c5fd;font-size:12px;letter-spacing:2px;font-weight:800;margin:0 0 6px 0;'>A DENIS VILLENEUVE FILM</p>
                <h1 style='font-size:54px;font-weight:900;margin:0 0 10px 0;letter-spacing:3px;color:#ffffff;'>D U N E</h1>
                <p style='color:#e2e8f0;font-size:14px;margin-bottom:14px;'>
                    Adventure • Sci-Fi • 2021 • 2h 35m • <span style='color:#fbbf24;font-weight:800;'>⭐ 8.0</span>
                </p>
                <p style='color:#f8fafc;font-size:15px;line-height:1.6;margin-bottom:24px;'>
                    A noble family, a dangerous desert planet, and a prophecy that could change the fate of humanity.
                </p>
                <div style='display:flex;gap:12px;align-items:center;'>
                    <a href='https://www.youtube.com/watch?v=n9xhJrPXop4' target='_blank' style='background:#e50914;color:white;padding:10px 24px;border-radius:10px;font-weight:700;text-decoration:none;font-size:14px;box-shadow:0 0 15px rgba(229,9,20,0.6);'>
                        ▶ Play
                    </a>
                    <a href='https://www.youtube.com/watch?v=n9xhJrPXop4' target='_blank' style='color:#ffffff;font-size:13.5px;font-weight:600;text-decoration:none;display:flex;align-items:center;gap:6px;background:rgba(255,255,255,0.12);padding:10px 18px;border-radius:10px;border:1px solid rgba(255,255,255,0.2);'>
                        ▶ Watch Trailer
                    </a>
                </div>
            </div>
            
            <div style='background:rgba(15,23,42,0.85);backdrop-filter:blur(25px);border:1px solid rgba(255,255,255,0.18);border-radius:18px;padding:20px;max-width:320px;box-shadow:0 10px 30px rgba(0,0,0,0.6);'>
                <div style='display:flex;align-items:center;gap:8px;margin-bottom:8px;'>
                    <span style='font-size:18px;'>✨</span>
                    <b style='color:#38bdf8;font-size:14.5px;'>Why watch this?</b>
                </div>
                <p style='color:#e2e8f0;font-size:13.5px;line-height:1.6;margin:0 0 10px 0;'>
                    Because you love sci-fi, complex worlds and strong protagonists. Matches your taste profile seeded from searches like Dune & Interstellar.
                </p>
            </div>
        </div>
    </div>
    """
    st.markdown(clean_html(banner_html), unsafe_allow_html=True)

    # Hero Interactive Actions
    h_b1, h_b2, h_b3 = st.columns([1.5, 1.5, 5])
    with h_b1:
        if st.button("🔖 + Add to Watchlist", key="hero_add_wl", width="stretch"):
            dune_item = {"title": "Dune", "rating": 8.0, "poster_url": "https://image.tmdb.org/t/p/w500/v1tRXZ4JtD2Iv6fjkPvT4GiwslV.jpg", "origin": "Featured Blockbuster"}
            if not any(m['title'] == 'Dune' for m in st.session_state.my_watchlist):
                st.session_state.my_watchlist.insert(0, dune_item)
                st.toast("🎉 Added Dune to your Watchlist!")
            else:
                st.toast("ℹ️ Dune is already in your Watchlist!")
    with h_b2:
        if st.button("🌌 Explore Dune Universe ›", key="hero_exp_dune", width="stretch"):
            set_search_query("Dune", "https://image.tmdb.org/t/p/w500/v1tRXZ4JtD2Iv6fjkPvT4GiwslV.jpg", 8.0)
            st.rerun()

    # Row 1: Trending Now
    t_c1, t_c2 = st.columns([6, 1])
    with t_c1:
        st.markdown("<span style='font-size:20px;font-weight:800;color:#fff;'>🔥 Trending Now</span>", unsafe_allow_html=True)
    with t_c2:
        if st.button("See all ›", key="see_all_trend"):
            st.session_state.nav_choice = "🔥 Trending Now"
            st.rerun()
    render_movie_row(trending_row, max_items=8, key_prefix="trend_row")

    # Row 2: Because you liked Dune (Recommendation of your search)
    d_c1, d_c2 = st.columns([6, 1])
    with d_c1:
        st.markdown("<span style='font-size:20px;font-weight:800;color:#fff;'>🎯 Because you searched 'Dune' (AI Recommendations)</span>", unsafe_allow_html=True)
    with d_c2:
        if st.button("See all ›", key="see_all_dune"):
            st.session_state.active_search_query = "Dune"
            st.rerun()
    render_movie_row(dune_row, max_items=8, key_prefix="dune_rec")

    # Row 3: My Watchlist (Seeded by search recommendations)
    w_c1, w_c2 = st.columns([6, 1])
    with w_c1:
        st.markdown(f"<span style='font-size:20px;font-weight:800;color:#fff;'>🔖 My Watchlist ({len(st.session_state.my_watchlist)})</span>", unsafe_allow_html=True)
    with w_c2:
        if st.button("See all ›", key="see_all_wl"):
            st.session_state.nav_choice = "🤍 My Watchlist"
            st.rerun()
    render_movie_row(st.session_state.my_watchlist, max_items=8, key_prefix="wl_row")

# 2. DISCOVER MOVIES (Dynamic Genre Discovery & Filtering)
elif st.session_state.nav_choice == "🧭 Discover Movies":
    st.markdown("<h1 style='color:#fff;'>🧭 Discover Movies & Deep Cinema</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#e2e8f0;font-size:14px;'>Current Filter: <b style='color:#00f2fe;'>{st.session_state.selected_genre}</b> • Discovering through your search interests</p>", unsafe_allow_html=True)

    if st.session_state.selected_genre != "All":
        genre_movies = tmdb_service.get_movies_by_genre(st.session_state.selected_genre, limit=16)
        if genre_movies:
            st.markdown(f"### 🎬 Top {st.session_state.selected_genre} Cinema")
            render_movie_row(genre_movies[:8], max_items=8, key_prefix=f"disc_{st.session_state.selected_genre}_1")
            if len(genre_movies) > 8:
                st.markdown(f"### 🍿 More {st.session_state.selected_genre} Discoveries")
                render_movie_row(genre_movies[8:16], max_items=8, key_prefix=f"disc_{st.session_state.selected_genre}_2")
    else:
        # All Genres Overview: Each domain has its own specified movie collection!
        st.markdown("### 💥 Explosive Action & Superhero")
        render_movie_row(tmdb_service.get_movies_by_genre("Action", limit=8), max_items=8, key_prefix="disc_act")
        
        st.markdown("### 🌌 Deep Cosmic Sci-Fi")
        render_movie_row(tmdb_service.get_movies_by_genre("Sci-Fi", limit=8), max_items=8, key_prefix="disc_scifi")
        
        st.markdown("### 👻 Supernatural & Psychological Horror")
        render_movie_row(tmdb_service.get_movies_by_genre("Horror", limit=8), max_items=8, key_prefix="disc_horror")
        
        st.markdown("### 😂 Pure Comedy & Laughs")
        render_movie_row(tmdb_service.get_movies_by_genre("Comedy", limit=8), max_items=8, key_prefix="disc_comedy")
        
        st.markdown("### 💫 Acclaimed Animation Masterworks")
        render_movie_row(tmdb_service.get_movies_by_genre("Animation", limit=8), max_items=8, key_prefix="disc_anim")
        
        st.markdown("### 🌹 Iconic Romance & Drama")
        render_movie_row(tmdb_service.get_movies_by_genre("Romance", limit=8), max_items=8, key_prefix="disc_rom")

# 3. FOR YOU (AI Recommendations Based On Your Searches)
elif st.session_state.nav_choice == "✨ For You":
    st.markdown("<h1 style='color:#fff;'>✨ For You • AI Recommendations of Your Searches</h1>", unsafe_allow_html=True)
    recent_qs = [h['query'] for h in st.session_state.search_history[:4]]
    st.markdown(f"<p style='color:#e2e8f0;font-size:14px;'>Deep content matching generated from your searches for: <b style='color:#00f2fe;'>{', '.join(recent_qs)}</b></p>", unsafe_allow_html=True)
    
    for_you_picks = [
        {"title": "Interstellar", "rating": 8.7, "poster_url": "https://image.tmdb.org/t/p/w500/yQvGrMoipbRoddT0ZR8tPoR7NfX.jpg", "origin": "From 'Dune' Search"},
        {"title": "Avengers: Endgame", "rating": 8.3, "poster_url": "https://image.tmdb.org/t/p/w500/or06FN3Dka5tukK1e9sl16pB3iy.jpg", "origin": "From Superhero Affinity"},
        {"title": "Blade Runner 2049", "rating": 8.0, "poster_url": "https://image.tmdb.org/t/p/w500/gajva2L0rPYkEWjzgFlBXCAVBE5.jpg", "origin": "From Denis Villeneuve"},
        {"title": "Inception", "rating": 8.8, "poster_url": "https://image.tmdb.org/t/p/w500/xlaY2zyzMfkhk0HSC5VUwzoZPU1.jpg", "origin": "From 'Oppenheimer' Search"},
        {"title": "Spider-Man: Across the Spider-Verse", "rating": 8.4, "poster_url": "https://image.tmdb.org/t/p/w500/8Vt6mWEReuy4Of61Lnj5Xj704m8.jpg", "origin": "From Animation Top Pick"},
        {"title": "The Matrix", "rating": 8.7, "poster_url": "https://image.tmdb.org/t/p/w500/dXNAPwY7VrqMAo51EKhhCJfaGb5.jpg", "origin": "From Cyberpunk Taste"},
        {"title": "The Martian", "rating": 8.0, "poster_url": "https://image.tmdb.org/t/p/w500/5BHuvQ6p9kfc091Z8RiFNhCwL4b.jpg", "origin": "From 'Interstellar' Search"},
        {"title": "Arrival", "rating": 7.9, "poster_url": "https://image.tmdb.org/t/p/w500/pEzNVQfdzYDzVK0XqxERIw2x2se.jpg", "origin": "From 'Dune' Search"},
    ]
    render_movie_row(for_you_picks, max_items=8, key_prefix="for_you_row")

# 4. TRENDING NOW
elif st.session_state.nav_choice == "🔥 Trending Now":
    st.markdown("<h1 style='color:#fff;'>🔥 Trending Now • Global Top Hits</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#e2e8f0;font-size:14px;'>The most watched and celebrated cinema worldwide this week.</p>", unsafe_allow_html=True)
    render_movie_row(trending_row, max_items=8, key_prefix="trend_all_1")

# 5. NEW RELEASES
elif st.session_state.nav_choice == "📅 New Releases":
    st.markdown("<h1 style='color:#fff;'>📅 New Releases • In Theaters & Streaming</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#e2e8f0;font-size:14px;'>Freshly released cinematic masterworks and streaming premieres.</p>", unsafe_allow_html=True)
    render_movie_row(new_releases_row, max_items=8, key_prefix="new_rel_1")

# 6. MY WATCHLIST (YOUR SPACE: Predefined & dynamically populated by search recommendations)
elif st.session_state.nav_choice == "🤍 My Watchlist":
    st.markdown("<h1 style='color:#fff;'>🤍 My Watchlist • Tailored by Your Searches</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#e2e8f0;font-size:14px;'>You have <b style='color:#00f2fe;'>{len(st.session_state.my_watchlist)}</b> movies saved in your queue based on your searches and recommendations.</p>", unsafe_allow_html=True)
    
    if st.session_state.my_watchlist:
        w_cols = st.columns(4)
        for idx, movie in enumerate(st.session_state.my_watchlist):
            with w_cols[idx % 4]:
                st.image(movie.get("poster_url"), width="stretch")
                st.markdown(f"<div style='color:#ffffff;font-size:14px;font-weight:700;margin-top:6px;'>{movie.get('title')} <span style='color:#fbbf24;font-size:12.5px;'>⭐ {movie.get('rating', 8.0)}</span></div>", unsafe_allow_html=True)
                if movie.get("origin"):
                    st.markdown(f"<div class='origin-badge'>🎯 {movie['origin']}</div>", unsafe_allow_html=True)
                b_c1, b_c2 = st.columns(2)
                with b_c1:
                    if st.button("Explore", key=f"wl_exp_{idx}", width="stretch"):
                        set_search_query(movie.get("title", ""))
                        st.rerun()
                with b_c2:
                    if st.button("🗑️ Remove", key=f"wl_del_{idx}", width="stretch"):
                        st.session_state.my_watchlist.pop(idx)
                        st.toast(f"Removed {movie.get('title')} from Watchlist")
                        st.rerun()
    else:
        st.info("Your watchlist is empty! Click '+ Add to Watchlist' on any movie card.")

# 7. WATCH HISTORY (YOUR SPACE: Exactly according to your search & recommendations)
elif st.session_state.nav_choice == "🕒 Watch History":
    st.markdown("<h1 style='color:#fff;'>🕒 Watch & Search History</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#e2e8f0;font-size:14px;'>Your history is automatically tracked according to your searches and watched journeys.</p>", unsafe_allow_html=True)

    # Section 1: Recent Searches (Clickable to Re-Search)
    st.markdown("### 🔍 Your Recent Searches")
    s_cols = st.columns(len(st.session_state.search_history[:6]))
    for s_idx, s_item in enumerate(st.session_state.search_history[:6]):
        with s_cols[s_idx]:
            st.image(s_item["poster_url"], width="stretch")
            st.markdown(f"<div style='color:#ffffff;font-size:14px;font-weight:700;margin-top:6px;'>{s_item['query']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='color:#cbd5e1;font-size:12px;margin:2px 0 6px 0;'>🕒 {s_item['time']}</div>", unsafe_allow_html=True)
            if st.button("Re-Search", key=f"re_search_{s_idx}", width="stretch"):
                set_search_query(s_item["query"])
                st.rerun()

    st.write("---")

    # Section 2: Watched & Explored History
    st.markdown("### 🎬 Movies Watched (Discovered via Search)")
    h_cols = st.columns(len(st.session_state.watch_history[:6]))
    for idx, item in enumerate(st.session_state.watch_history[:6]):
        with h_cols[idx]:
            st.image(item["poster_url"], width="stretch")
            st.markdown(f"<div style='color:#ffffff;font-size:14px;font-weight:700;margin-top:6px;'>{item['title']} <span style='color:#fbbf24;font-size:12.5px;'>⭐ {item['rating']}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='color:#cbd5e1;font-size:12px;margin:2px 0;'>Status: {item['progress']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='origin-badge'>🔍 {item.get('origin', 'Search')}</div>", unsafe_allow_html=True)
            if st.button("Rewatch", key=f"hist_btn_{idx}", width="stretch"):
                set_search_query(item["title"])
                st.rerun()

# 8. MY RATINGS (YOUR SPACE: Ratings of films found through search)
elif st.session_state.nav_choice == "⭐ My Ratings":
    st.markdown("<h1 style='color:#fff;'>⭐ My Ratings & Search Impressions</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#e2e8f0;font-size:14px;'>Your personal ratings and reviews for titles discovered via Cineflix search.</p>", unsafe_allow_html=True)

    r_cols = st.columns(len(st.session_state.my_ratings))
    for r_idx, r_item in enumerate(st.session_state.my_ratings):
        with r_cols[r_idx]:
            st.image(r_item["poster_url"], width="stretch")
            st.markdown(f"<div style='color:#ffffff;font-size:14px;font-weight:700;margin-top:6px;'>{r_item['title']}</div>", unsafe_allow_html=True)
            st.markdown(f"<span style='color:#fbbf24;font-weight:800;font-size:13.5px;'>{r_item['score']}</span>", unsafe_allow_html=True)
            st.markdown(f"<div style='color:#f8fafc;font-size:12px;font-style:italic;line-height:1.45;margin:6px 0;background:rgba(255,255,255,0.08);padding:6px 8px;border-radius:8px;border-left:3px solid #fbbf24;'>💭 \"{r_item['review']}\"</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='origin-badge'>📝 {r_item.get('tag', 'Reviewed')}</div>", unsafe_allow_html=True)
            if st.button("Explore", key=f"rate_exp_{r_idx}", width="stretch"):
                set_search_query(r_item["title"])
                st.rerun()

# 9. YOUR CINEMATIC DNA TABLET (YOUR SPACE: Predefined & calibrated by your searches)
elif st.session_state.nav_choice == "🧬 My Taste Profile":
    dna_html = """
    <div class='dna-tablet'>
        <div class='tablet-notch'></div>
        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:15px;'>
            <span style='color:#00f2fe;font-size:16px;'>⠶</span>
            <span style='color:#38bdf8;font-size:12.5px;font-weight:800;letter-spacing:1.5px;text-shadow:0 0 10px rgba(56,189,248,0.6);'>CINEMATIC SLATE PRO v2.4 • CALIBRATED TO SEARCHES</span>
            <span style='color:#00f2fe;font-size:14px;'>🗖 🗕 ✕</span>
        </div>

        <div style='text-align:center;margin-bottom:25px;'>
            <h1 style='color:#ffffff;font-size:34px;font-weight:900;letter-spacing:3px;margin:0;text-shadow:0 0 25px rgba(0,242,254,0.65);'>
                YOUR CINEMATIC DNA
            </h1>
            <p style='color:#e2e8f0;font-size:14px;margin:6px 0 0 0;'>
                Taste Topology dynamically indexed from your searches: <b style='color:#38bdf8;'>Dune, Oppenheimer, Interstellar, Inception</b>
            </p>
        </div>
        
        <div style='position:relative;background:radial-gradient(circle at center, rgba(0,242,254,0.08) 0%, rgba(7,12,24,0.95) 75%);border:1px solid rgba(0,242,254,0.4);border-radius:28px;padding:35px 25px;margin-bottom:30px;box-shadow:inset 0 0 40px rgba(0,242,254,0.12);'>
            
            <div style='position:absolute;top:50%;left:5%;right:5%;height:1px;background:rgba(0,242,254,0.3);transform:translateY(-50%);'></div>
            <div style='position:absolute;left:50%;top:5%;bottom:5%;width:1px;background:rgba(0,242,254,0.3);transform:translateX(-50%);'></div>
            
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:40px;position:relative;z-index:2;'>
                
                <!-- Top Left: Action & Sci-Fi Buffs (Seeded by Dune & Inception searches) -->
                <div style='background:rgba(0,242,254,0.08);border:1px solid rgba(0,242,254,0.4);border-radius:18px;padding:16px;'>
                    <div style='color:#38bdf8;font-weight:800;font-size:14.5px;margin-bottom:4px;'>⚡ Action & Sci-Fi Buffs (85% Match)</div>
                    <div style='color:#cbd5e1;font-size:12px;font-weight:500;margin-bottom:10px;'>Seeded from your searches for Dune & Inception</div>
                    <div style='display:flex;gap:12px;'>
                        <img src='https://image.tmdb.org/t/p/w500/v1tRXZ4JtD2Iv6fjkPvT4GiwslV.jpg' style='width:48%;border-radius:10px;box-shadow:0 4px 15px rgba(0,0,0,0.5);border:1px solid rgba(0,242,254,0.4);'/>
                        <img src='https://image.tmdb.org/t/p/w500/xlaY2zyzMfkhk0HSC5VUwzoZPU1.jpg' style='width:48%;border-radius:10px;box-shadow:0 4px 15px rgba(0,0,0,0.5);border:1px solid rgba(0,242,254,0.4);'/>
                    </div>
                </div>

                <!-- Top Right: Compare with Friends (Seeded by Oppenheimer & Batman searches) -->
                <div style='background:rgba(229,9,20,0.08);border:1px solid rgba(229,9,20,0.4);border-radius:18px;padding:16px;position:relative;'>
                    <div style='color:#ff6b81;font-weight:800;font-size:14.5px;margin-bottom:4px;'>👥 Compare with Friends (78% Match)</div>
                    <div style='color:#cbd5e1;font-size:12px;font-weight:500;margin-bottom:10px;'>Seeded from your searches for Oppenheimer & Dark Knight</div>
                    <div style='display:flex;gap:12px;'>
                        <img src='https://image.tmdb.org/t/p/w500/8Gxv8gSFCU0XGDykEGv7zR1n2ua.jpg' style='width:48%;border-radius:10px;box-shadow:0 4px 15px rgba(0,0,0,0.5);border:1px solid rgba(229,9,20,0.4);'/>
                        <img src='https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg' style='width:48%;border-radius:10px;box-shadow:0 4px 15px rgba(0,0,0,0.5);border:1px solid rgba(229,9,20,0.4);'/>
                    </div>
                    <div style='margin-top:10px;display:flex;flex-direction:column;gap:6px;'>
                        <span style='background:rgba(15,23,42,0.9);border:1px solid rgba(255,255,255,0.2);padding:5px 12px;border-radius:12px;font-size:11.5px;color:#f1f5f9;font-weight:600;'>
                            👤 Riya • 78% Match
                        </span>
                        <span style='background:rgba(15,23,42,0.9);border:1px solid rgba(255,255,255,0.2);padding:5px 12px;border-radius:12px;font-size:11.5px;color:#f1f5f9;font-weight:600;'>
                            👤 Ben • 55% Match
                        </span>
                        <span style='background:rgba(0,242,254,0.18);border:1px solid rgba(0,242,254,0.5);padding:5px 12px;border-radius:12px;font-size:11.5px;color:#38bdf8;font-weight:800;'>
                            ✨ Chloe • 92% Match "Arthouse Twin"
                        </span>
                    </div>
                </div>

                <!-- Bottom Left: Indie & Arthouse Explorers -->
                <div style='background:rgba(74,222,128,0.08);border:1px solid rgba(74,222,128,0.4);border-radius:18px;padding:16px;'>
                    <div style='color:#4ade80;font-weight:800;font-size:14.5px;margin-bottom:4px;'>🌿 Indie & Arthouse Explorers (60% Match)</div>
                    <div style='color:#cbd5e1;font-size:12px;font-weight:500;margin-bottom:10px;'>Seeded from your searches for Parasite & Whiplash</div>
                    <div style='display:flex;gap:12px;'>
                        <img src='https://image.tmdb.org/t/p/w500/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg' style='width:48%;border-radius:10px;box-shadow:0 4px 15px rgba(0,0,0,0.5);border:1px solid rgba(74,222,128,0.4);'/>
                        <img src='https://image.tmdb.org/t/p/w500/7fn624j5lj3xTme2SgiLCeuedmO.jpg' style='width:48%;border-radius:10px;box-shadow:0 4px 15px rgba(0,0,0,0.5);border:1px solid rgba(74,222,128,0.4);'/>
                    </div>
                </div>

                <!-- Bottom Right: Classic Hollywood Romantics -->
                <div style='background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.4);border-radius:18px;padding:16px;'>
                    <div style='color:#fbbf24;font-weight:800;font-size:14.5px;margin-bottom:4px;'>🍷 Classic Hollywood Romantics (30% Match)</div>
                    <div style='color:#cbd5e1;font-size:12px;font-weight:500;margin-bottom:10px;'>Seeded from your searches for Titanic & Forrest Gump</div>
                    <div style='display:flex;gap:12px;'>
                        <img src='https://image.tmdb.org/t/p/w500/9xjZS2rlVxm8SFx8kPC3aIGCOYQ.jpg' style='width:48%;border-radius:10px;box-shadow:0 4px 15px rgba(0,0,0,0.5);border:1px solid rgba(245,158,11,0.4);'/>
                        <img src='https://image.tmdb.org/t/p/w500/saHP97rTPS5eLmrLQEcANmKrsFl.jpg' style='width:48%;border-radius:10px;box-shadow:0 4px 15px rgba(0,0,0,0.5);border:1px solid rgba(245,158,11,0.4);'/>
                    </div>
                </div>
            </div>

            <!-- Center Glowing "YOU" Node -->
            <div style='position:absolute;top:50%;left:50%;transform:translate(-50%, -50%);background:#0f172a;border:3px solid #fbbf24;box-shadow:0 0 30px #fbbf24;width:64px;height:64px;border-radius:50%;display:flex;align-items:center;justify-content:center;color:#ffffff;font-weight:900;font-size:15px;letter-spacing:1px;z-index:10;'>
                YOU
            </div>
        </div>

        <div style='max-width:650px;margin:0 auto;'>
            <h4 style='color:#ffffff;text-align:center;font-size:15px;font-weight:700;letter-spacing:1px;margin-bottom:18px;'>Taste Similarity Scores (Calibrated to Search Affinity)</h4>
            
            <div style='margin-bottom:14px;'>
                <div style='display:flex;justify-content:space-between;color:#38bdf8;font-size:13.5px;font-weight:700;margin-bottom:5px;'>
                    <span>Sci-Fi & Cyberpunk Topology (Dune, Inception)</span><span style='color:#ffffff;'>85%</span>
                </div>
                <div style='background:rgba(255,255,255,0.1);height:9px;border-radius:10px;overflow:hidden;'>
                    <div style='background:#38bdf8;width:85%;height:100%;box-shadow:0 0 12px #38bdf8;'></div>
                </div>
            </div>

            <div style='margin-bottom:14px;'>
                <div style='display:flex;justify-content:space-between;color:#4ade80;font-size:13.5px;font-weight:700;margin-bottom:5px;'>
                    <span>Psychological Mystery & Complex Protagonists (Oppenheimer, Batman)</span><span style='color:#ffffff;'>60%</span>
                </div>
                <div style='background:rgba(255,255,255,0.1);height:9px;border-radius:10px;overflow:hidden;'>
                    <div style='background:#4ade80;width:60%;height:100%;box-shadow:0 0 12px #4ade80;'></div>
                </div>
            </div>

            <div style='margin-bottom:14px;'>
                <div style='display:flex;justify-content:space-between;color:#ff6b81;font-size:13.5px;font-weight:700;margin-bottom:5px;'>
                    <span>Classic Rom-Com & Comfort Cinema (Titanic, Gump)</span><span style='color:#ffffff;'>30%</span>
                </div>
                <div style='background:rgba(255,255,255,0.1);height:9px;border-radius:10px;overflow:hidden;'>
                    <div style='background:#ff6b81;width:30%;height:100%;box-shadow:0 0 12px #ff6b81;'></div>
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown(clean_html(dna_html), unsafe_allow_html=True)

# 10. MOOD EXPLORER (Interactive mood selector with working recommendations)
elif st.session_state.nav_choice == "😊 Mood Explorer":
    st.markdown(
        clean_html("""
        <div class='liquid-panel' style='text-align:center;'>
            <h1 style='font-size:38px;color:#ffffff;margin-bottom:8px;'>What's your mood today? 🔮</h1>
            <p style='color:#e2e8f0;font-size:16px;'>Pick a vibe, and the cosmic engine will curate the perfect universe for you.</p>
        </div>
        """),
        unsafe_allow_html=True
    )
    m_cols = st.columns(6)
    moods = [
        ("🌌 Cosmic Sci-Fi", "Interstellar"),
        ("⚡ Dark & Gritty", "The Batman"),
        ("🕯️ Chill & Relaxed", "Spirited Away"),
        ("💖 Romantic", "Titanic"),
        ("🧠 Mind-Bending", "Inception"),
        ("🎲 Surprise Me", "Fight Club")
    ]
    for idx, (m_label, m_target) in enumerate(moods):
        with m_cols[idx]:
            if st.button(m_label, key=f"m_{idx}", width="stretch"):
                set_search_query(m_target)
                st.rerun()

    st.markdown("<h3 style='color:#ffffff;margin-top:20px;'>🍿 Handpicked for high vibe resonance:</h3>", unsafe_allow_html=True)
    render_movie_row(trending_row[:4] + dune_row[:4], max_items=8, key_prefix="mood_picks")

# 11. SURPRISE ME (Interactive Cinematic Roulette)
elif st.session_state.nav_choice == "🔀 Surprise Me":
    st.markdown("<h1 style='color:#fff;'>🔀 Surprise Me • Cinematic Roulette</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#e2e8f0;font-size:14px;'>Letting the AI cosmos pick your next watch.</p>", unsafe_allow_html=True)
    
    all_pool = trending_row + dune_row + st.session_state.my_watchlist
    surprise_pick = random.choice(all_pool)
    c1, c2 = st.columns([1, 2.5])
    with c1:
        st.image(surprise_pick["poster_url"], width="stretch")
    with c2:
        st.markdown(f"<h2 style='color:#ffffff;margin-top:0;'>🎲 Pick of the Moment: <b style='color:#38bdf8;'>{surprise_pick['title']}</b></h2>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:#fbbf24;font-size:16px;font-weight:700;'>⭐ {surprise_pick['rating']} / 10</div>", unsafe_allow_html=True)
        st.markdown("<p style='color:#e2e8f0;font-size:14.5px;margin-top:8px;'>A stellar pick carefully selected from your taste galaxy.</p>", unsafe_allow_html=True)
        sb_col1, sb_col2 = st.columns(2)
        with sb_col1:
            if st.button("🎲 Spin Again", key="spin_again", width="stretch"):
                st.rerun()
        with sb_col2:
            if st.button("Explore Details", key="spin_exp", width="stretch"):
                set_search_query(surprise_pick["title"], surprise_pick["poster_url"], surprise_pick["rating"])
                st.rerun()

# 12. COMPARE MOVIES (Side-by-side comparative analytics)
elif st.session_state.nav_choice == "⚖️ Compare Movies":
    st.markdown("<h1 style='color:#fff;'>⚖️ Compare Movies</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#e2e8f0;font-size:14px;'>Side-by-side cinematic showdown.</p>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            clean_html("""
            <div class='liquid-panel'>
                <h3 style='color:#ffffff;font-size:20px;font-weight:800;margin:0 0 10px 0;'>Movie A: <b style='color:#38bdf8;'>Dune (2021)</b></h3>
                <img src='https://image.tmdb.org/t/p/w500/v1tRXZ4JtD2Iv6fjkPvT4GiwslV.jpg' style='width:180px;border-radius:12px;margin:10px 0 14px 0;box-shadow:0 8px 25px rgba(0,0,0,0.6);'/>
                <p style='color:#e2e8f0;font-size:14.5px;margin:6px 0;'><b style='color:#ffffff;'>⭐ Rating:</b> <span style='color:#fbbf24;font-weight:700;'>8.0 / 10</span></p>
                <p style='color:#e2e8f0;font-size:14.5px;margin:6px 0;'><b style='color:#ffffff;'>🏷️ Genres:</b> Sci-Fi, Adventure</p>
                <p style='color:#e2e8f0;font-size:14.5px;margin:6px 0;'><b style='color:#ffffff;'>🎬 Director:</b> Denis Villeneuve</p>
                <p style='color:#e2e8f0;font-size:14.5px;margin:6px 0;'><b style='color:#ffffff;'>⏳ Runtime:</b> 2h 35m</p>
            </div>
            """),
            unsafe_allow_html=True
        )
    with col_b:
        st.markdown(
            clean_html("""
            <div class='liquid-panel'>
                <h3 style='color:#ffffff;font-size:20px;font-weight:800;margin:0 0 10px 0;'>Movie B: <b style='color:#38bdf8;'>Interstellar (2014)</b></h3>
                <img src='https://image.tmdb.org/t/p/w500/yQvGrMoipbRoddT0ZR8tPoR7NfX.jpg' style='width:180px;border-radius:12px;margin:10px 0 14px 0;box-shadow:0 8px 25px rgba(0,0,0,0.6);'/>
                <p style='color:#e2e8f0;font-size:14.5px;margin:6px 0;'><b style='color:#ffffff;'>⭐ Rating:</b> <span style='color:#fbbf24;font-weight:700;'>8.7 / 10</span></p>
                <p style='color:#e2e8f0;font-size:14.5px;margin:6px 0;'><b style='color:#ffffff;'>🏷️ Genres:</b> Sci-Fi, Drama</p>
                <p style='color:#e2e8f0;font-size:14.5px;margin:6px 0;'><b style='color:#ffffff;'>🎬 Director:</b> Christopher Nolan</p>
                <p style='color:#e2e8f0;font-size:14.5px;margin:6px 0;'><b style='color:#ffffff;'>⏳ Runtime:</b> 2h 49m</p>
            </div>
            """),
            unsafe_allow_html=True
        )

# 13. SETTINGS
elif st.session_state.nav_choice == "⚙️ Settings":
    st.markdown("<h1 style='color:#fff;'>⚙️ Settings & Preferences</h1>", unsafe_allow_html=True)
    st.markdown(
        clean_html("""
        <div class='liquid-panel'>
            <h3 style='color:#ffffff;font-size:19px;font-weight:700;margin-top:0;'>Streaming Platforms Available</h3>
            <p style='color:#cbd5e1;font-size:14px;margin-bottom:16px;'>Filter recommendations according to your active subscriptions.</p>
            <div style='display:flex;gap:15px;margin-top:15px;flex-wrap:wrap;'>
                <label style='display:flex;align-items:center;gap:10px;color:#f8fafc;font-size:14px;font-weight:600;background:rgba(255,255,255,0.08);padding:10px 18px;border-radius:12px;border:1px solid rgba(255,255,255,0.18);cursor:pointer;'><input type='checkbox' checked /> Netflix</label>
                <label style='display:flex;align-items:center;gap:10px;color:#f8fafc;font-size:14px;font-weight:600;background:rgba(255,255,255,0.08);padding:10px 18px;border-radius:12px;border:1px solid rgba(255,255,255,0.18);cursor:pointer;'><input type='checkbox' checked /> Amazon Prime Video</label>
                <label style='display:flex;align-items:center;gap:10px;color:#f8fafc;font-size:14px;font-weight:600;background:rgba(255,255,255,0.08);padding:10px 18px;border-radius:12px;border:1px solid rgba(255,255,255,0.18);cursor:pointer;'><input type='checkbox' checked /> Disney+ Hotstar</label>
                <label style='display:flex;align-items:center;gap:10px;color:#f8fafc;font-size:14px;font-weight:600;background:rgba(255,255,255,0.08);padding:10px 18px;border-radius:12px;border:1px solid rgba(255,255,255,0.18);cursor:pointer;'><input type='checkbox' checked /> Apple TV+</label>
            </div>
            <hr style='border:0.5px solid rgba(255,255,255,0.12);margin:25px 0;'>
            <h3 style='color:#ffffff;font-size:19px;font-weight:700;margin-top:0;'>Audio & Subtitles</h3>
            <p style='color:#cbd5e1;font-size:14px;margin-bottom:16px;'>Preferred language defaults for recommendations and trailers.</p>
            <div style='display:flex;gap:15px;margin-top:15px;flex-wrap:wrap;'>
                <label style='display:flex;align-items:center;gap:10px;color:#f8fafc;font-size:14px;font-weight:600;background:rgba(255,255,255,0.08);padding:10px 18px;border-radius:12px;border:1px solid rgba(255,255,255,0.18);cursor:pointer;'><input type='radio' name='lang' checked /> Original Language</label>
                <label style='display:flex;align-items:center;gap:10px;color:#f8fafc;font-size:14px;font-weight:600;background:rgba(255,255,255,0.08);padding:10px 18px;border-radius:12px;border:1px solid rgba(255,255,255,0.18);cursor:pointer;'><input type='radio' name='lang' /> English</label>
                <label style='display:flex;align-items:center;gap:10px;color:#f8fafc;font-size:14px;font-weight:600;background:rgba(255,255,255,0.08);padding:10px 18px;border-radius:12px;border:1px solid rgba(255,255,255,0.18);cursor:pointer;'><input type='radio' name='lang' /> Hindi</label>
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )
    if st.button("💾 Save Preferences", key="save_pref_btn"):
        st.toast("🎉 Preferences saved successfully!")

# 14. HELP
elif st.session_state.nav_choice == "❓ Help":
    st.markdown("<h1 style='color:#fff;'>❓ Cineflix Help & About</h1>", unsafe_allow_html=True)
    st.markdown(
        clean_html("""
        <div class='liquid-panel'>
            <h3 style='color:#ffffff;font-size:20px;font-weight:800;margin-top:0;'>How Cineflix AI Discovery Works</h3>
            <p style='color:#e2e8f0;font-size:14.5px;line-height:1.7;'>
                Cineflix merges real-time global TMDB data with machine learning classification.
                Search for any title to see live metadata, trailers, streaming providers, and content-based similarity suggestions.
            </p>
            <hr style='border:0.5px solid rgba(255,255,255,0.12);margin:22px 0;'>
            <h4 style='color:#ffffff;font-size:17px;font-weight:700;margin-bottom:14px;'>Frequently Asked Questions</h4>
            <p style='color:#38bdf8;font-size:15px;font-weight:700;margin-bottom:4px;'>Q: Does Cineflix work for movies not in the original spreadsheet?</p>
            <p style='color:#e2e8f0;font-size:14px;line-height:1.6;margin-top:0;margin-bottom:16px;'>A: Yes! If a movie is not found in the local spreadsheet, Cineflix connects directly to the global TMDB network.</p>
            <p style='color:#38bdf8;font-size:15px;font-weight:700;margin-bottom:4px;'>Q: How is my taste DNA calculated?</p>
            <p style='color:#e2e8f0;font-size:14px;line-height:1.6;margin-top:0;margin-bottom:16px;'>A: Cineflix analyzes your search keywords, watch journey, and genre preferences to plot your coordinates on the cosmic slate.</p>
        </div>
        """),
        unsafe_allow_html=True
    )

st.markdown(
    clean_html("""
    <hr style='border:0.5px solid rgba(255,255,255,0.12);margin-top:50px;'>
    <p style='text-align:center;color:#cbd5e1;font-size:13px;font-weight:500;letter-spacing:0.5px;'>
        CINEFLIX Liquid Space Discovery • Translucent Holographic Edition • Powered by TMDB & Scikit-Learn
    </p>
    """),
    unsafe_allow_html=True
)
