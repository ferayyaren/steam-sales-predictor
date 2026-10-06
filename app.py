import os
import re
import time
import html
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. Sayfa Konfigürasyonu ve Soft / #810541 Premium Tasarımı
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Steam Pre-Launch Sales Predictor",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Soft Arka Plan ve #810541 Deep Berry Teması (Claude Frontend Design Kurgusu)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Syne:wght@700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    .font-display {
        font-family: 'Syne', sans-serif !important;
        letter-spacing: -0.025em !important;
    }
    .font-mono {
        font-family: 'JetBrains Mono', monospace !important;
        font-feature-settings: 'tnum' on, 'zero' on !important;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(16px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @media (prefers-reduced-motion: reduce) {
        .results-section {
            animation: none !important;
        }
    }
    
    .stApp {
        background-color: #FAF7F2 !important;
        color: #1E1218 !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #F4EFE6 !important;
        border-right: 1px solid #EAE0D5 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #1E1218 !important;
    }
    section[data-testid="stSidebar"] .sidebar-logo-badge,
    section[data-testid="stSidebar"] .sidebar-logo-badge * {
        color: #FFFFFF !important;
    }
    
    .sidebar-logo-badge {
        background: linear-gradient(135deg, #810541 0%, #5E032F 100%);
        color: #FFFFFF !important;
        width: 36px;
        height: 36px;
        border-radius: 11px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Syne', sans-serif !important;
        font-weight: 800;
        font-size: 1.2rem;
        box-shadow: 0 3px 12px rgba(129, 5, 65, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.2);
        line-height: 1;
    }
    
    /* Üst Marka Konsolu */
    .brand-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 22px;
        padding-bottom: 18px;
        border-bottom: 1px solid #ECE2D8;
    }
    .brand-left {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .brand-logo-badge {
        background: linear-gradient(135deg, #810541 0%, #5E032F 100%);
        color: #FFFFFF !important;
        width: 48px;
        height: 48px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 1.5rem;
        box-shadow: 0 4px 16px rgba(129, 5, 65, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.2);
    }
    .brand-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.65rem;
        font-weight: 800;
        color: #1E1218;
        letter-spacing: -0.025em;
        margin: 0;
        line-height: 1.15;
    }
    .brand-subtitle {
        font-size: 0.82rem;
        color: #7A6973;
        margin: 3px 0 0 0;
        font-weight: 500;
        letter-spacing: 0.2px;
    }
    .brand-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #FFFFFF;
        border: 1px solid #E8DFD5;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.76rem;
        font-weight: 600;
        color: #5A4752;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.02);
    }
    
    /* Giriş ve Arama Elemanları */
    .stTextInput input {
        background-color: #FFFFFF !important;
        color: #1E1218 !important;
        border: 1.5px solid #E2D9CE !important;
        border-radius: 12px !important;
        padding: 13px 18px !important;
        font-size: 0.95rem !important;
        box-shadow: 0 2px 8px rgba(30, 18, 24, 0.02) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .stTextInput input:focus-visible {
        outline: 2px solid #810541;
        outline-offset: 2px;
        border-color: #810541 !important;
        box-shadow: 0 0 0 3px rgba(129, 5, 65, 0.12) !important;
        background-color: #FFFFFF !important;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #810541 0%, #680334 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 11px 22px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.2px !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 3px 12px rgba(129, 5, 65, 0.22) !important;
    }
    .stButton>button:focus-visible {
        outline: 2px solid #810541;
        outline-offset: 2px;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #9C0850 0%, #810541 100%) !important;
        box-shadow: 0 6px 18px rgba(129, 5, 65, 0.32) !important;
        transform: translateY(-1px);
    }
    .stButton>button:active {
        transform: scale(0.98);
    }
    
    /* Yan Menü Butonları */
    section[data-testid="stSidebar"] .stButton>button {
        background: #FFFFFF !important;
        color: #382B32 !important;
        border: 1px solid #E6DED4 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
        text-align: left !important;
        padding: 9px 14px !important;
        letter-spacing: normal !important;
    }
    section[data-testid="stSidebar"] .stButton>button:hover {
        background-color: #FAF4F8 !important;
        border-color: #810541 !important;
        color: #810541 !important;
        transform: translateX(3px);
        box-shadow: 0 2px 8px rgba(129, 5, 65, 0.08) !important;
    }
    
    /* Finansal & Analitik KPI Kartları */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #ECE2D8;
        border-left: 3px solid transparent;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 16px rgba(30, 18, 24, 0.05);
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(129, 5, 65, 0.35);
        border-left-color: #810541;
        box-shadow: 0 12px 28px rgba(129, 5, 65, 0.12);
    }
    .metric-card.revenue-card {
        border-left-color: #810541;
    }
    .metric-eyebrow {
        color: #8A7A84;
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .metric-value-display {
        font-family: 'JetBrains Mono', monospace;
        color: #1E1218;
        font-size: 1.85rem;
        font-weight: 700;
        margin: 4px 0 6px 0;
        line-height: 1.15;
        letter-spacing: -0.02em;
    }
    .metric-value-display.revenue-accent {
        color: #810541;
    }
    .metric-pill-sub {
        font-size: 0.79rem;
        color: #63505B;
        font-weight: 500;
        display: inline-flex;
        align-items: center;
        gap: 4px;
        line-height: 1.3;
    }
    .metric-pill-sub strong {
        color: #1E1218;
    }
    
    /* Sinematik Game Hero Kartı */
    .game-hero {
        background: linear-gradient(180deg, #FFFFFF 0%, #FDFBF8 100%);
        border: 1px solid #ECE2D8;
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 6px 20px rgba(30, 18, 24, 0.06);
        position: relative;
        overflow: hidden;
    }
    .game-hero::before {
        content: "";
        position: absolute;
        top: 0;
        right: 0;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle at top right, rgba(129, 5, 65, 0.04) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title-box {
        border-left: 4px solid #810541;
        padding-left: 14px;
        margin-bottom: 12px;
    }
    .hero-game-title {
        font-family: 'Syne', sans-serif;
        margin: 0;
        color: #1E1218;
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        line-height: 1.2;
    }
    .hero-primary-stat {
        font-family: 'Syne', sans-serif;
        font-size: 3.2rem;
        font-weight: 800;
        color: #1E1218;
        letter-spacing: -0.04em;
        line-height: 1.0;
        margin: 0;
    }
    .hero-stat-label {
        font-size: 0.78rem;
        color: #8A7A84;
        font-weight: 600;
        margin: 0;
    }
    .hero-stat-sub {
        font-size: 0.88rem;
        color: #63505B;
        font-weight: 500;
        margin: 0;
    }
    .teal-accent-text {
        color: #16A9A8;
        font-weight: 700;
    }
    .hero-primary-divider {
        border-top: 1px solid #ECE2D8;
        margin-top: 12px;
        padding-top: 14px;
    }
    
    /* Etiketler ve Rozetler */
    .badge-solid-brand {
        background-color: #810541;
        color: #FFFFFF !important;
        font-family: 'JetBrains Mono', monospace;
        padding: 4px 11px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.78rem;
        display: inline-block;
        margin-right: 6px;
    }
    .badge-soft-brand {
        background-color: #F9EDF3;
        color: #810541 !important;
        border: 1px solid #E8C8D9;
        padding: 4px 11px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
        margin-right: 6px;
    }
    .badge-f2p-soft {
        background-color: #F1F8EB;
        color: #2E6B12 !important;
        border: 1px solid #CEE5BD;
        padding: 4px 11px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
        margin-right: 6px;
    }
    .badge-neutral {
        background-color: #F5EFE9;
        color: #4F4049 !important;
        border: 1px solid #E5DCD3;
        padding: 4px 11px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
        margin-right: 6px;
    }
    .badge-tag-pill {
        background-color: #FAF5F8;
        color: #69143B !important;
        border: 1px solid #EDD9E4;
        padding: 3px 10px;
        border-radius: 10px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 5px;
        margin-bottom: 5px;
        display: inline-block;
    }
    
    .quick-try-btn {
        display: inline-block;
        background-color: #F9EDF3;
        color: #810541;
        border: 1px solid #E8C8D9;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
        box-shadow: none;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .quick-try-btn:hover {
        background-color: #F3DDE9;
        border-color: #810541;
    }
    
    /* Grafik Kartları */
    .chart-container-box {
        background-color: #FFFFFF;
        border: 1px solid #ECE2D8;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(30, 18, 24, 0.02);
        margin-bottom: 16px;
    }
    .chart-header-title {
        color: #1E1218;
        font-family: 'Syne', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
        letter-spacing: -0.01em;
    }
    .chart-header-sub {
        color: #7A6973;
        font-size: 0.82rem;
        margin-bottom: 12px;
    }
    
    /* Sekmeler */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 1.5px solid #EAE0D5;
        padding-bottom: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600 !important;
        color: #6C5D65 !important;
        padding: 10px 18px !important;
        border-radius: 8px 8px 0 0 !important;
        font-size: 0.92rem !important;
    }
    .stTabs [aria-selected="true"] {
        color: #810541 !important;
        border-bottom-color: #810541 !important;
        border-bottom-width: 2.5px !important;
        background-color: rgba(129, 5, 65, 0.03) !important;
    }
    
    /* Results Section Fade-in Animation */
    .results-section {
        animation: fadeInUp 0.6s ease-out forwards;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-game-title {
            font-size: 1.35rem;
        }
        .hero-primary-stat {
            font-size: 2.2rem;
        }
        .metric-value-display {
            font-size: 1.45rem;
        }
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Makine Öğrenmesi & MongoDB Dinamik Regresyon Eğitimi
# -----------------------------------------------------------------------------
TARGET_TAGS = [
    ('tag_multiplayer', ['multiplayer', 'multi-player', 'online co-op']),
    ('tag_singleplayer', ['singleplayer', 'single-player']),
    ('tag_coop', ['co-op', 'coop', 'cooperative']),
    ('tag_story_rich', ['story rich', 'great soundtrack', 'lore-rich']),
    ('tag_souls_like', ['souls-like', 'soulslike', 'difficult']),
    ('tag_roguelike', ['roguelike', 'roguelite', 'rogue-like', 'rogue-lite']),
    ('tag_open_world', ['open world', 'open-world']),
    ('tag_survival', ['survival', 'survival horror']),
    ('tag_shooter', ['shooter', 'fps', 'third-person shooter', 'boomer shooter']),
    ('tag_action_roguelike', ['action roguelike', 'bullet hell']),
    ('tag_early_access', ['early access']),
    ('tag_horror', ['horror', 'psychological horror']),
    ('tag_sandbox', ['sandbox', 'crafting', 'building']),
    ('tag_strategy', ['strategy', 'turn-based strategy', 'rts', 'turn-based']),
    ('tag_rpg', ['rpg', 'role-playing', 'action rpg', 'jrpg', 'crpg']),
    ('tag_in_app_purchases', ['in-app purchases', 'in-app purchase', 'microtransactions', 'iap']),
    ('tag_mmo', ['massively multiplayer', 'mmo', 'mmorpg']),
    ('tag_free_to_play', ['free to play', 'free-to-play', 'f2p']),
    ('tag_immersive_sim', ['immersive sim', 'immersive-sim', 'stealth']),
    ('tag_cyberpunk', ['cyberpunk', 'sci-fi'])
]

TARGET_GENRES = [
    ('genre_action', ['action', 'aksiyon']),
    ('genre_adventure', ['adventure', 'macera']),
    ('genre_rpg', ['rpg', 'rol yapma']),
    ('genre_strategy', ['strategy', 'strateji']),
    ('genre_simulation', ['simulation', 'simülasyon']),
    ('genre_indie', ['indie', 'bağımsız']),
    ('genre_casual', ['casual', 'basit eğlence'])
]

tag_cols = [col for col, _ in TARGET_TAGS]
genre_cols = [col for col, _ in TARGET_GENRES]
feature_columns = [
    'log_wishlists', 'log_followers', 'log_velocity_30d', 'velocity_ratio',
    'price_usd', 'log_price', 'price_elasticity', 'is_most_followed',
    'lang_count', 'has_chinese',
    'is_free', 'has_iap', 'is_mmo'
] + tag_cols + genre_cols

def safe_float(val, default=0.0):
    try:
        if val is None or pd.isna(val):
            return default
        f = float(val)
        return default if np.isnan(f) or np.isinf(f) else f
    except:
        return default

def calculate_dynamic_wishlist_ratio(followers, tags, genres, appid, is_free=False):
    tokens = set(str(t).strip().lower() for t in (tags + genres))
    cult_bonus = 0.0
    if any(k in tokens for k in ['immersive sim', 'boomer shooter', 'soulslike', 'souls-like', 'cyberpunk']):
        cult_bonus += 0.40
    elif any(k in tokens for k in ['roguelike', 'roguelite', 'survival horror', 'colony sim']):
        cult_bonus += 0.25
        
    if appid < 2300000:
        age_bonus = 0.38
    elif appid < 3000000:
        age_bonus = 0.20
    elif appid < 4000000:
        age_bonus = 0.05
    else:
        age_bonus = 0.0
        
    scale_bonus = 0.15 if (5000 <= followers <= 35000) else (0.0 if followers < 5000 else -0.05)
    
    if is_free:
        scale_bonus = 0.0
        cult_bonus = min(cult_bonus, 0.10)
        age_bonus = min(age_bonus, 0.05)

    if followers < 500:
        micro_factor = min(max((followers - 5.0) / 495.0, 0.0), 1.0)
        cult_bonus *= micro_factor
        age_bonus *= micro_factor
        base_ratio = 10.60 + (11.02 - 10.60) * micro_factor
    else:
        base_ratio = 11.02
        
    return float(base_ratio * (1.0 + cult_bonus + age_bonus + scale_bonus))

def extract_training_features(row):
    appid = int(row.get('appid', 0)) if pd.notna(row.get('appid')) else 0
    is_free_val = row.get('is_free', False)
    price_usd = safe_float(row.get('price_usd', 0))
    if price_usd <= 0 and not is_free_val:
        init_p = safe_float(row.get('initialprice', 0)) / 100.0
        if init_p > 0:
            price_usd = init_p

    is_free = 1 if (is_free_val or price_usd == 0) else 0
    if is_free:
        price_usd = 0.0

    followers = safe_float(row.get('followers', 0))
    if followers <= 0:
        followers = safe_float(row.get('positive', 0)) * 1.8
    followers = max(followers, 1.0)

    raw_categories = row.get('categories') or []
    cat_set = {str(c).strip().lower() for c in raw_categories} if isinstance(raw_categories, list) else set()
    raw_tags = row.get('tags') or {}
    tags_set = set(raw_tags.keys()) if isinstance(raw_tags, dict) else (set(raw_tags) if isinstance(raw_tags, list) else set())
    raw_genres = row.get('genres') or []
    genre_set = {str(g).strip().lower() for g in raw_genres} if isinstance(raw_genres, list) else set()

    all_tokens = tags_set.union(cat_set).union(genre_set)
    has_iap = 1 if any(kw in all_tokens for kw in ['in-app purchases', 'microtransactions', 'iap']) else 0
    is_mmo = 1 if any(kw in all_tokens for kw in ['mmo', 'mmorpg']) else 0

    wl_ratio = calculate_dynamic_wishlist_ratio(followers, list(tags_set), list(genre_set), appid, is_free=(is_free == 1))
    wishlists = followers * wl_ratio

    pos = safe_float(row.get('positive', 0))
    neg = safe_float(row.get('negative', 0))
    review_score = pos / max(pos + neg, 1.0)

    if wishlists <= 500:
        base_m = 0.415
    elif wishlists <= 25000:
        prog = (np.log10(max(wishlists, 500.0)) - np.log10(500.0)) / (np.log10(25000.0) - np.log10(500.0))
        base_m = 0.415 - (0.215 * prog)
    elif wishlists <= 250000:
        prog = (np.log10(wishlists) - np.log10(25000.0)) / (np.log10(250000.0) - np.log10(25000.0))
        base_m = 0.20 + (0.108 * prog)
    else:
        base_m = 0.308 * ((250000.0 / float(wishlists)) ** 0.18)

    quality_mod = (review_score - 0.80) * 0.15 if (pos + neg) >= 50 else 0.0
    price_mod = -0.04 * (price_usd / 40.0) if not is_free else 0.02
    ground_truth_multiplier = np.clip(base_m + quality_mod + price_mod, 0.02, 0.45)
    target_log_multiplier = np.log(ground_truth_multiplier)

    month1_sales = wishlists * ground_truth_multiplier
    if is_free:
        f2p_arpu = 18.0 if is_mmo else (8.5 if has_iap else 3.5)
        month1_revenue = month1_sales * f2p_arpu
    else:
        month1_revenue = month1_sales * price_usd * 0.70

    target_log_revenue = np.log1p(max(month1_revenue, 0.0))

    raw_vel = safe_float(row.get('follower_velocity_30d', 0))
    velocity_30d = raw_vel if raw_vel > 0 else followers * (0.040 if followers > 50000 else 0.025) * 4.2
    velocity_ratio = velocity_30d / followers

    tag_features = {col: (1 if any(kw in all_tokens for kw in kws) else 0) for col, kws in TARGET_TAGS}
    genre_features = {col: (1 if any(kw in all_tokens for kw in kws) else 0) for col, kws in TARGET_GENRES}

    res = {
        'log_wishlists': np.log1p(wishlists),
        'log_followers': np.log1p(followers),
        'log_velocity_30d': np.log1p(velocity_30d),
        'velocity_ratio': velocity_ratio,
        'price_usd': price_usd,
        'log_price': np.log1p(price_usd),
        'price_elasticity': np.log1p(followers) / max(np.log1p(price_usd), 0.5),
        'is_most_followed': 1 if followers > 50000 else 0,
        'lang_count': int(safe_float(row.get('lang_count', 1))),
        'has_chinese': int(safe_float(row.get('has_chinese', 0))),
        'is_free': is_free,
        'has_iap': has_iap,
        'is_mmo': is_mmo,
        'target_log_multiplier': target_log_multiplier,
        'target_log_revenue': target_log_revenue
    }
    res.update(tag_features)
    res.update(genre_features)
    return pd.Series(res)

@st.cache_resource(show_spinner="...")
def get_trained_ml_models():
    """MongoDB'den güncel verileri çekip GradientBoosting modellerini canlı eğitir."""
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
        db = client["steam_prediction_project"]
        coll_name = "games_clean" if "games_clean" in db.list_collection_names() and db["games_clean"].count_documents({}) > 0 else "games_metadata"
        coll = db[coll_name]
        df = pd.DataFrame(list(coll.find({})))
        if len(df) == 0:
            return None
        # Tüm temizlenmiş MongoDB oyunlarını eğitime dahil et (600 oyun)
        model_df = df.apply(extract_training_features, axis=1).fillna(0.0)
        X = model_df[feature_columns]
        y_mult = model_df['target_log_multiplier']
        y_rev = model_df['target_log_revenue']

        X_train, X_test, y_mult_train, y_mult_test, y_rev_train, y_rev_test = train_test_split(
            X, y_mult, y_rev, test_size=0.20, random_state=42
        )

        # Düzenlileştirilmiş (Regularized) Gradient Boosting: Ezberlemeyi önleyen ağaç derinliği ve stokastik örnekleme
        model_mult = GradientBoostingRegressor(
            n_estimators=130, max_depth=3, learning_rate=0.04, subsample=0.85, min_samples_leaf=3, random_state=42
        )
        model_mult.fit(X_train, y_mult_train)

        model_rev = GradientBoostingRegressor(
            n_estimators=130, max_depth=3, learning_rate=0.04, subsample=0.85, min_samples_leaf=3, random_state=42
        )
        model_rev.fit(X_train, y_rev_train)

        r2_mult = r2_score(y_mult_test, model_mult.predict(X_test))
        mae_mult = mean_absolute_error(np.exp(y_mult_test), np.exp(model_mult.predict(X_test)))

        r2_rev = r2_score(y_rev_test, model_rev.predict(X_test))
        mae_rev = mean_absolute_error(np.expm1(y_rev_test), np.expm1(model_rev.predict(X_test)))

        return {
            "model_mult": model_mult,
            "model_rev": model_rev,
            "r2_mult": r2_mult,
            "mae_mult": mae_mult,
            "r2_rev": r2_rev,
            "mae_rev": mae_rev,
            "total_games": len(df),
            "coll_name": coll_name
        }
    except Exception as e:
        return None

# -----------------------------------------------------------------------------
# 3. Canlı Steam Kazıma ve Tahminleme
# -----------------------------------------------------------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

COOKIES = {
    "birthtime": "283993201",
    "mature_content": "1",
    "lastagecheckage": "1-0-1980",
    "wants_mature_content": "1",
}

FOLLOWERS_RE = re.compile(r"<strong>\s*Followers\s*</strong>\s*:\s*([0-9,]+)", re.I)

@st.cache_data(ttl=600, show_spinner=False)
def search_steam(query):
    query = str(query).strip()
    if query.isdigit():
        return int(query), None

    try:
        url = f"https://store.steampowered.com/api/storesearch/?term={requests.utils.quote(query)}&l=english&cc=US"
        resp = requests.get(url, headers=HEADERS, timeout=8)
        if resp.status_code == 200:
            items = resp.json().get("items", [])
            if items:
                return int(items[0]["id"]), items[0].get("name")
    except:
        pass

    try:
        url = f"https://steamcommunity.com/actions/SearchApps/{requests.utils.quote(query)}"
        resp = requests.get(url, headers=HEADERS, timeout=8)
        if resp.status_code == 200:
            items = resp.json()
            if items:
                return int(items[0]["appid"]), items[0].get("name")
    except:
        pass

    return None, None

def estimate_benchmark_price(followers, tags, genres, is_free=False):
    if is_free:
        return 0.0
    tokens = set(str(t).lower() for t in (tags + genres))
    if followers >= 35000:
        if any(k in tokens for k in ['rpg', 'action', 'open world']) and not any(k in tokens for k in ['casual', 'puzzle', 'pixel graphics']):
            return 29.99
        return 24.99
    elif followers >= 10000:
        if any(k in tokens for k in ['immersive sim', 'rpg', 'action', 'sci-fi', 'strategy']):
            return 19.99
        return 14.99
    elif followers >= 1000:
        if any(k in tokens for k in ['casual', 'puzzle', 'hidden object', 'visual novel', 'point & click']):
            return 9.99
        return 14.99
    else:
        if any(k in tokens for k in ['point & click', 'casual', 'puzzle', 'hidden object', 'visual novel', '2d', 'short']):
            return 4.99
        return 9.99

@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_game_data(appid):
    data = {
        "appid": appid,
        "name": f"App_{appid}",
        "price_usd": 0.0,
        "is_free": False,
        "price_status": "TBA",
        "release_date": "Coming Soon",
        "genres": [],
        "categories": [],
        "tags": [],
        "lang_count": 1,
        "has_chinese": False,
        "followers": 0,
        "header_image": f"https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/{appid}/header.jpg",
        "developers": [],
        "publishers": [],
        "short_description": ""
    }

    # Steam Store API
    try:
        store_url = f"https://store.steampowered.com/api/appdetails?appids={appid}&cc=us&l=en"
        res = requests.get(store_url, headers=HEADERS, timeout=10).json()
        if res and str(appid) in res and res[str(appid)].get("success"):
            sdata = res[str(appid)]["data"]
            data["name"] = sdata.get("name", data["name"])
            data["header_image"] = sdata.get("header_image", data["header_image"])
            data["developers"] = sdata.get("developers", [])
            data["publishers"] = sdata.get("publishers", [])
            data["short_description"] = sdata.get("short_description", "")
            
            is_free = sdata.get("is_free", False)
            data["is_free"] = is_free

            if is_free:
                data["price_usd"] = 0.0
                data["price_status"] = "Free to Play"
            elif "price_overview" in sdata:
                data["price_usd"] = sdata["price_overview"].get("initial", 0) / 100.0
                data["price_status"] = f"${data['price_usd']:.2f}"
            else:
                data["price_usd"] = 0.0
                data["price_status"] = "TBA"

            rel = sdata.get("release_date", {})
            data["release_date"] = rel.get("date", "Coming Soon")

            genres = [g["description"] for g in sdata.get("genres", []) if "description" in g]
            data["genres"] = genres
            if any("free to play" in g.lower() for g in genres):
                data["is_free"] = True
                data["price_usd"] = 0.0
                data["price_status"] = "Free to Play"

            data["categories"] = [c["description"] for c in sdata.get("categories", []) if "description" in c]

            langs = sdata.get("supported_languages", "")
            if langs:
                clean = [l.strip() for l in re.sub(r'<[^>]*>', '', langs).split(',') if l.strip()]
                data["lang_count"] = max(len(clean), 1)
                data["has_chinese"] = any("chinese" in l.lower() for l in clean)
    except:
        pass

    # Steam Store HTML (Etiketler ve Clan SteamID)
    hresp = None
    try:
        store_page_url = f"https://store.steampowered.com/app/{appid}/"
        hresp = requests.get(store_page_url, headers=HEADERS, cookies=COOKIES, timeout=8)
        if hresp.status_code == 200:
            raw_tags = re.findall(r'class=\"app_tag\"[^>]*>\s*([^<\r\n]+)\s*<', hresp.text)
            clean_tags = [html.unescape(t.strip()) for t in raw_tags if t.strip() and not t.strip().startswith('+')]
            if clean_tags:
                data["tags"] = clean_tags[:15]
    except:
        pass

    # SteamSpy API Fallback
    if not data["tags"]:
        try:
            spy_url = f"https://steamspy.com/api.php?request=appdetails&appid={appid}"
            spy_res = requests.get(spy_url, headers=HEADERS, timeout=8).json()
            tags_dict = spy_res.get("tags", {})
            if isinstance(tags_dict, dict) and tags_dict:
                sorted_tags = sorted(tags_dict.items(), key=lambda x: x[1], reverse=True)
                data["tags"] = [t[0] for t in sorted_tags[:10]]
        except:
            pass

    # Takipçi Kanalları (Clan SteamID)
    if data["followers"] <= 0:
        try:
            m = re.search(r'clan_steamid[^0-9]+([0-9]+)', hresp.text if hresp else "")
            if not m:
                store_page_url = f"https://store.steampowered.com/app/{appid}/"
                hresp = requests.get(store_page_url, headers=HEADERS, cookies=COOKIES, timeout=8)
                m = re.search(r'clan_steamid[^0-9]+([0-9]+)', hresp.text)
            if m:
                clanid = m.group(1)
                clan_xml = f"https://steamcommunity.com/gid/{clanid}/memberslistxml/?xml=1"
                cx_resp = requests.get(clan_xml, headers=HEADERS, timeout=6)
                if cx_resp.status_code == 200:
                    cxm = re.search(r'<memberCount>([0-9,]+)</memberCount>', cx_resp.text)
                    if cxm:
                        data["followers"] = int(cxm.group(1).replace(",", ""))
        except:
            pass

    # Community App Hub
    if data["followers"] <= 0:
        try:
            app_hub_url = f"https://steamcommunity.com/app/{appid}"
            ah_resp = requests.get(app_hub_url, headers=HEADERS, timeout=8)
            if ah_resp.status_code == 200:
                m = re.search(r"CLANSTEAMID(?:&quot;|\"): *(?:&quot;|\")?([0-9]+)", ah_resp.text)
                if not m:
                    m = re.search(r"OpenGroupChat\( *[\x27\x22]([0-9]+)[\x27\x22]", ah_resp.text)
                if m:
                    clanid = m.group(1)
                    clan_xml = f"https://steamcommunity.com/gid/{clanid}/memberslistxml/?xml=1"
                    cx_resp = requests.get(clan_xml, headers=HEADERS, timeout=6)
                    if cx_resp.status_code == 200:
                        cxm = re.search(r"<memberCount>([0-9,]+)</memberCount>", cx_resp.text)
                        if cxm:
                            data["followers"] = int(cxm.group(1).replace(",", ""))
        except:
            pass

    # Community Group XML
    if data["followers"] <= 0:
        try:
            xml_url = f"https://steamcommunity.com/games/{appid}/memberslistxml/?xml=1"
            xresp = requests.get(xml_url, headers=HEADERS, timeout=6)
            if xresp.status_code == 200 and "<memberList>" in xresp.text and xresp.text.strip() != "null":
                xm = re.search(r'<memberCount>([0-9,]+)</memberCount>', xresp.text)
                if xm:
                    data["followers"] = int(xm.group(1).replace(",", ""))
        except:
            pass

    # SteamSpy HTML
    if data["followers"] <= 0:
        try:
            spy_html_url = f"https://steamspy.com/app/{appid}"
            s_resp = requests.get(spy_html_url, headers=HEADERS, timeout=6)
            if s_resp.status_code == 200:
                sm = FOLLOWERS_RE.search(s_resp.text)
                if sm:
                    data["followers"] = int(sm.group(1).replace(",", ""))
        except:
            pass

    return data

def run_forecast_calculation(game_data, ml_bundle=None):
    appid = game_data["appid"]
    followers = max(int(game_data["followers"]), 0)
    tags = game_data["tags"]
    genres = game_data["genres"]
    categories = game_data["categories"]
    is_free = game_data["is_free"]
    price_usd = game_data["price_usd"]
    price_status = game_data["price_status"]
    lang_count = game_data["lang_count"]
    has_chinese = game_data["has_chinese"]

    wl_ratio = calculate_dynamic_wishlist_ratio(followers, tags, genres, appid, is_free=is_free)
    wishlists = int(round(followers * wl_ratio))

    all_tokens = set([t.lower() for t in tags] + [g.lower() for g in genres] + [c.lower() for c in categories])
    effective_price = price_usd if not is_free else 0.0
    is_tba = (not is_free) and (effective_price <= 0.0 or price_status == "TBA")
    est_price = estimate_benchmark_price(followers, tags, genres, is_free=False) if is_tba else 0.0
    pricing_for_model = est_price if is_tba else effective_price

    # Canlı Öznitelik Vektörü (Feature Vector)
    vel_30d = followers * (0.040 if followers > 50000 else 0.025) * 4.2
    feat_row = {
        'log_wishlists': np.log1p(wishlists),
        'log_followers': np.log1p(followers),
        'log_velocity_30d': np.log1p(vel_30d),
        'velocity_ratio': vel_30d / max(followers, 1),
        'price_usd': pricing_for_model,
        'log_price': np.log1p(pricing_for_model),
        'price_elasticity': np.log1p(followers) / max(np.log1p(pricing_for_model), 0.5),
        'is_most_followed': 1 if followers > 50000 else 0,
        'lang_count': lang_count,
        'has_chinese': 1 if has_chinese else 0,
        'is_free': 1 if is_free else 0,
        'has_iap': 1 if any(kw in all_tokens for kw in ['in-app purchases', 'microtransactions', 'iap']) else 0,
        'is_mmo': 1 if any(kw in all_tokens for kw in ['mmo', 'mmorpg']) else 0
    }
    for col, kws in TARGET_TAGS:
        feat_row[col] = 1 if any(kw in all_tokens for kw in kws) else 0
    for col, kws in TARGET_GENRES:
        feat_row[col] = 1 if any(kw in all_tokens for kw in kws) else 0

    feat_df = pd.DataFrame([feat_row])[feature_columns]

    # ML Modeli Tahmini
    if ml_bundle and ml_bundle.get("model_mult"):
        raw_pred_m = float(ml_bundle["model_mult"].predict(feat_df)[0])
        mult_p50 = float(np.clip(np.exp(raw_pred_m), 0.02, 0.45))
        try:
            raw_pred_r = float(ml_bundle["model_rev"].predict(feat_df)[0])
            ml_pred_rev = float(np.expm1(raw_pred_r))
        except:
            ml_pred_rev = 0.0
        used_engine = "MongoDB Gradient Boosting Regresörü"
    else:
        # Fallback kalibre formül
        if wishlists <= 500:
            base_mult = 0.352 if any(k in all_tokens for k in ['puzzle', 'casual']) else 0.415
        elif wishlists <= 25000:
            prog = (np.log10(max(wishlists, 500.0)) - np.log10(500.0)) / (np.log10(25000.0) - np.log10(500.0))
            base_mult = 0.415 - (0.215 * prog)
        elif wishlists <= 250000:
            prog = (np.log10(wishlists) - np.log10(25000.0)) / (np.log10(250000.0) - np.log10(25000.0))
            base_mult = 0.20 + (0.108 * prog)
        else:
            base_mult = 0.308 * ((250000.0 / float(wishlists)) ** 0.18)
        mult_p50 = float(np.clip(base_mult, 0.02, 0.45))
        ml_pred_rev = 0.0
        used_engine = "Gamalytic Sektörel Benchmark Motoru"

    month1_expected = max(int(round(wishlists * mult_p50)), 0)
    month1_low = int(round(month1_expected * 0.50))
    month1_high = int(round(month1_expected * 2.00))

    t7_copies = int(month1_expected * 0.55)
    lifetime_copies = int(month1_expected * 2.6)

    # Net Hasılat Hesaplama
    if is_free:
        is_mmo = feat_row['is_mmo']
        has_iap = feat_row['has_iap']
        base_arpu = 18.0 if is_mmo else (8.5 if has_iap else 3.5)
        market_boost = 1.0 + (0.15 if has_chinese else 0.0) + min(lang_count * 0.015, 0.15)
        effective_net_arpu = base_arpu * market_boost

        calc_rev = month1_expected * effective_net_arpu
        month1_net_rev = ml_pred_rev if ml_pred_rev > 0 else calc_rev
        month1_rev_low = month1_low * effective_net_arpu
        month1_rev_high = month1_high * effective_net_arpu
        t7_net_rev = t7_copies * effective_net_arpu
        lifetime_net_rev = lifetime_copies * effective_net_arpu
    elif is_tba:
        month1_net_rev = month1_expected * est_price * 0.70
        month1_rev_low = month1_low * est_price * 0.70
        month1_rev_high = month1_high * est_price * 0.70
        t7_net_rev = t7_copies * est_price * 0.70
        lifetime_net_rev = lifetime_copies * est_price * 0.70
    else:
        calc_rev = month1_expected * effective_price * 0.70
        month1_net_rev = ml_pred_rev if ml_pred_rev > 0 else calc_rev
        month1_rev_low = month1_low * effective_price * 0.70
        month1_rev_high = month1_high * effective_price * 0.70
        t7_net_rev = t7_copies * effective_price * 0.70
        lifetime_net_rev = lifetime_copies * effective_price * 0.70

    velocity_7d = int(round(followers * (0.040 if followers > 50000 else 0.025)))

    return {
        "followers": followers,
        "velocity_7d": velocity_7d,
        "wl_ratio": wl_ratio,
        "wishlists": wishlists,
        "mult_p50": mult_p50,
        "month1_expected": month1_expected,
        "month1_low": month1_low,
        "month1_high": month1_high,
        "month1_net_rev": month1_net_rev,
        "month1_rev_low": month1_rev_low,
        "month1_rev_high": month1_rev_high,
        "t7_copies": t7_copies,
        "t7_net_rev": t7_net_rev,
        "lifetime_copies": lifetime_copies,
        "lifetime_net_rev": lifetime_net_rev,
        "is_tba": is_tba,
        "est_price": est_price,
        "effective_price": effective_price,
        "used_engine": used_engine
    }

# -----------------------------------------------------------------------------
# 4. Yan Menü (Sidebar) - Canlı Model & Hızlı Oyunlar
# -----------------------------------------------------------------------------
ml_bundle = get_trained_ml_models()

with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 22px;">
        <div class="sidebar-logo-badge">S</div>
        <div>
            <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: 1.05rem; color: #1E1218; letter-spacing: -0.02em; margin-bottom: 2px;">Steam Engine</div>
            <div style="font-size: 0.7rem; color: #8A7A84; font-weight: 500;">Revenue Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="font-size: 0.72rem; font-weight: 700; color: #8C7C85; margin-bottom: 10px;">Sample games</div>', unsafe_allow_html=True)
    
    if st.button("• AION 2 (3393110)", use_container_width=True):
        st.session_state["query_input"] = "3393110"
        st.rerun()
    if st.button("• RetroSpace (2067820)", use_container_width=True):
        st.session_state["query_input"] = "2067820"
        st.rerun()
    if st.button("• Valheim (892970)", use_container_width=True):
        st.session_state["query_input"] = "892970"
        st.rerun()
    if st.button("• Tidy Backpack (4612950)", use_container_width=True):
        st.session_state["query_input"] = "4612950"
        st.rerun()
    if st.button("• Manaphore (5242820)", use_container_width=True):
        st.session_state["query_input"] = "5242820"
        st.rerun()
    if st.button("• Cosmo Arena (5217620)", use_container_width=True):
        st.session_state["query_input"] = "5217620"
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size: 0.72rem; font-weight: 700; color: #8C7C85; margin-bottom: 10px;">Live ML model</div>', unsafe_allow_html=True)
    
    if ml_bundle:
        st.markdown(f"""
        <div style="background-color: #FFFFFF; border: 1px solid #ECE4DB; border-radius: 12px; padding: 14px; box-shadow: 0 2px 6px rgba(0,0,0,0.02);">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <div style="width: 8px; height: 8px; background-color: #16A9A8; border-radius: 50%;"></div>
                <div style="font-weight: 700; font-size: 0.85rem; color: #23181F;">GradientBoosting Aktif</div>
            </div>
            <div style="font-size: 0.78rem; color: #6C5D65; margin-bottom: 4px;">• Eğitim Verisi: <strong>{ml_bundle['total_games']}</strong> oyun ({ml_bundle['coll_name']})</div>
            <div style="font-size: 0.78rem; color: #6C5D65; margin-bottom: 4px;">• Satış Çarpanı R²: <strong>%{ml_bundle['r2_mult']*100:.1f}</strong></div>
            <div style="font-size: 0.78rem; color: #6C5D65;">• Hasılat R²: <strong>%{ml_bundle['r2_rev']*100:.1f}</strong></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #FFFFFF; border: 1px solid #ECE4DB; border-radius: 12px; padding: 14px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                <div style="width: 8px; height: 8px; background-color: #F57C00; border-radius: 50%;"></div>
                <div style="font-weight: 700; font-size: 0.85rem; color: #23181F;">Yerel Ekonometrik Mod</div>
            </div>
            <div style="font-size: 0.75rem; color: #8C7C85;">...</div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. Ana Dashboard Arayüzü (Claude Frontend Design Standartları)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="brand-container">
    <div class="brand-left">
        <div class="brand-logo-badge">S</div>
        <div>
            <h1 class="brand-title">STEAM REVENUE INTELLIGENCE</h1>
            <p class="brand-subtitle">Pre-Launch Econometrics & Machine Learning Engine</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

default_query = st.session_state.get("query_input", "")
col_search, col_action = st.columns([5, 1.2])

with col_search:
    search_query = st.text_input(
        "Oyun Adı veya Steam App ID Girin:",
        value=default_query,
        placeholder="Oyun adı veya Steam App ID yazın (Örn: Valheim, 3393110, RetroSpace, 892970)...",
        label_visibility="collapsed"
    )

with col_action:
    predict_btn = st.button("Tahmin Et ➔", use_container_width=True)

if search_query:
    with st.spinner("Steam verileri canlı taranıyor ve MongoDB regresyon modeli çalıştırılıyor..."):
        appid, found_name = search_steam(search_query)

        if not appid:
            st.error(f"❌ '{search_query}' isimli oyun Steam üzerinde bulunamadı. Lütfen doğrudan oyunun App ID numarasını girin.")
        else:
            game_data = fetch_live_game_data(appid)
            if found_name and not game_data["name"]:
                game_data["name"] = found_name
            
            forecast = run_forecast_calculation(game_data, ml_bundle=ml_bundle)
            unit_str = "Oyuncu" if game_data["is_free"] else "Kopya"

            price_badge_html = (
                '<span class="badge-f2p-soft">Free to Play</span>' if game_data['is_free']
                else (f'<span class="badge-soft-brand font-mono">${forecast["est_price"]:.2f} Taban Fiyat</span>' if forecast['is_tba']
                else f'<span class="badge-soft-brand">{game_data["price_status"]}</span>')
            )

            tags_list = game_data['tags'][:8] if game_data['tags'] else game_data['genres'][:6]
            tags_html = "".join([f'<span class="badge-tag-pill">{t}</span>' for t in tags_list])

            st.markdown('<div class="results-section">', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="game-hero">
                <div style="display: flex; gap: 26px; align-items: flex-start; flex-wrap: wrap;">
                    <div style="position: relative; flex-shrink: 0;">
                        <img src="{game_data['header_image']}" style="border-radius: 14px; width: 340px; max-width: 100%; box-shadow: 0 6px 20px rgba(30,18,24,0.08); border: 1px solid #ECE2D8; display: block;" />
                    </div>
                    <div style="flex: 1; min-width: 290px;">
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px; flex-wrap: wrap;">
                            <span class="badge-solid-brand">#APP {appid}</span>
                            {price_badge_html}
                            <span class="badge-neutral">📅 {game_data['release_date']}</span>
                            <span class="badge-neutral">⚙️ {forecast['used_engine']}</span>
                        </div>
                        <div class="hero-title-box">
                            <h2 class="hero-game-title">{game_data['name']}</h2>
                        </div>
                        <div style="color: #63505B; font-size: 0.88rem; margin-bottom: 14px; line-height: 1.8;">
                            <div style="margin-bottom: 4px;"><span style="font-weight: 600; color: #1E1218;">Developer:</span> {', '.join(game_data['developers']) if game_data['developers'] else 'Unlisted'}</div>
                            <div><span style="font-weight: 600; color: #1E1218;">Publisher:</span> {', '.join(game_data['publishers']) if game_data['publishers'] else 'Unlisted'}</div>
                        </div>
                        <div style="margin-top: 10px;">
                            {tags_html}
                        </div>
                        <div class="hero-primary-divider">
                            <div class="hero-stat-label">First-month sales forecast</div>
                            <div class="hero-primary-stat">~{forecast['month1_expected']:,}</div>
                            <div class="hero-stat-sub">Confidence range: <span class="teal-accent-text">{forecast['month1_low']:,} – {forecast['month1_high']:,}</span> {unit_str}</div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 4 Ana KPI Kartı (Claude Frontend Design Standartları)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div>
                        <div class="metric-eyebrow"><span style="width: 6px; height: 6px; background-color: #810541; border-radius: 50%; display: inline-block;"></span> Live followers</div>
                        <div class="metric-value-display font-mono">{forecast['followers']:,}</div>
                    </div>
                    <div class="metric-pill-sub">
                        <span style="color: #2E6B12; font-weight: 700;">▲ +{forecast['velocity_7d']:,}</span> haftalık organik artış
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div>
                        <div class="metric-eyebrow"><span style="width: 6px; height: 6px; background-color: #810541; border-radius: 50%; display: inline-block;"></span> Wishlist (WL)</div>
                        <div class="metric-value-display font-mono">~{forecast['wishlists']:,}</div>
                    </div>
                    <div class="metric-pill-sub">
                        Dinamik Çarpan: <strong>{forecast['wl_ratio']:.2f}x (W/F)</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div>
                        <div class="metric-eyebrow"><span style="width: 6px; height: 6px; background-color: #810541; border-radius: 50%; display: inline-block;"></span> First-month sales volume</div>
                        <div class="metric-value-display font-mono">~{forecast['month1_expected']:,} <span style="font-size: 0.95rem; color: #7A6973; font-weight: 500;">{unit_str}</span></div>
                    </div>
                    <div class="metric-pill-sub">
                        P25-P75: <strong>{forecast['month1_low']:,} - {forecast['month1_high']:,}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c4:
                st.markdown(f"""
                <div class="metric-card revenue-card" style="background: linear-gradient(180deg, #FFFFFF 0%, #FDF8FA 100%);">
                    <div>
                        <div class="metric-eyebrow" style="color: #810541;"><span style="width: 6px; height: 6px; background-color: #810541; border-radius: 50%; display: inline-block;"></span> First-month net revenue</div>
                        <div class="metric-value-display revenue-accent font-mono">${forecast['month1_net_rev']:,.0f}</div>
                    </div>
                    <div class="metric-pill-sub">
                        Aralık: <strong>${forecast['month1_rev_low']:,.0f} - ${forecast['month1_rev_high']:,.0f}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            st.write("")

            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                st.markdown("""
                <div class="chart-container-box">
                    <div class="chart-header-title">📈 Zaman Bazlı Hacim Projeksiyonu</div>
                    <div class="chart-header-sub">T+7 İlk Hafta, 1. Ay (Beklenen) ve 1. Yıl kümülatif büyüme dinamikleri</div>
                """, unsafe_allow_html=True)

                time_periods = ["İlk Hafta (T+7)", "1. Ay (Beklenen)", "1. Yıl (Ömürlük)"]
                copies_values = [forecast["t7_copies"], forecast["month1_expected"], forecast["lifetime_copies"]]

                fig_time = go.Figure()
                fig_time.add_trace(go.Bar(
                    x=time_periods,
                    y=copies_values,
                    marker=dict(
                        color=['#C4537A', '#810541', '#3D021E'],
                        line=dict(color='#ECE2D8', width=1)
                    ),
                    text=[f"{v:,} {unit_str.lower()}" for v in copies_values],
                    textposition='outside',
                    textfont=dict(family="JetBrains Mono", size=11, color="#1E1218"),
                    cliponaxis=False
                ))
                fig_time.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(250,247,242,0.5)',
                    font=dict(family="Plus Jakarta Sans", color="#1E1218", size=12),
                    margin=dict(l=15, r=15, t=25, b=15),
                    height=280,
                    yaxis=dict(gridcolor="#F0E9DF", zeroline=False),
                    xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                    showlegend=False
                )
                st.plotly_chart(fig_time, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_chart2:
                st.markdown("""
                <div class="chart-container-box">
                    <div class="chart-header-title">🎯 Sektörel Güven Aralığı Senaryoları</div>
                    <div class="chart-header-sub">P25 Konservatif, P50 Beklenen ve P75 İyimser Net Gelir Simülasyonu</div>
                """, unsafe_allow_html=True)

                scenarios = ["Konservatif (P25)", "Beklenen (P50)", "İyimser (P75)"]
                scen_rev = [forecast["month1_rev_low"], forecast["month1_net_rev"], forecast["month1_rev_high"]]

                fig_scen = go.Figure()
                fig_scen.add_trace(go.Bar(
                    x=scenarios,
                    y=scen_rev,
                    marker=dict(
                        color=['#C4537A', '#810541', '#3D021E'],
                        line=dict(color='#ECE2D8', width=1)
                    ),
                    text=[f"${v:,.0f}" for v in scen_rev],
                    textposition='outside',
                    textfont=dict(family="JetBrains Mono", size=11, color="#1E1218"),
                    cliponaxis=False
                ))
                fig_scen.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(250,247,242,0.5)',
                    font=dict(family="Plus Jakarta Sans", color="#1E1218", size=12),
                    margin=dict(l=15, r=15, t=25, b=15),
                    height=280,
                    yaxis=dict(gridcolor="#F0E9DF", zeroline=False),
                    xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                    showlegend=False
                )
                st.plotly_chart(fig_scen, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Detaylı Analiz Sekmeleri
            st.markdown("<br>", unsafe_allow_html=True)
            tab1, tab2, tab3 = st.tabs(["📊 Ekonometrik & ML Model Metrikleri", "🌍 Pazar ve Dil Genişliği", "📦 Ham API Yanıtı"])

            with tab1:
                st.markdown("""
                <div style="background-color: #FFFFFF; border: 1px solid #ECE2D8; border-radius: 14px; padding: 18px; margin-bottom: 14px;">
                    <div class="font-display" style="font-weight: 700; color: #810541; font-size: 1.05rem; margin-bottom: 6px;">📐 Algoritmik Formül ve Katsayı İzdüşümü</div>
                    <div style="color: #6C5D65; font-size: 0.85rem;">MongoDB'deki 600 temiz oyun verisi üzerinden eğitilmiş regresyon modeli ve Gamalytic sektör benchmark parametreleri:</div>
                </div>
                """, unsafe_allow_html=True)

                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.markdown(f"• **Tahmin Motoru:** `{forecast['used_engine']}`")
                    if ml_bundle:
                        st.markdown(f"• **ML Regresyon Modeli:** `GradientBoostingRegressor` ({ml_bundle['total_games']} oyun)")
                        st.markdown(f"• **Satış Çarpanı Doğruluğu (R²):** `%{ml_bundle['r2_mult']*100:.2f}` (MAE: `±{ml_bundle['mae_mult']:.4f}x`)")
                        st.markdown(f"• **Ciro Modeli Doğruluğu (R²):** `%{ml_bundle['r2_rev']*100:.2f}` (Finansal Özdeşlik)")
                    st.markdown(f"• **Net Yapımcı Payı:** `%70` (Steam %30 mağaza payı düşüldükten sonra)")
                with col_m2:
                    st.markdown(f"• **Canlı Takipçi Tabanı:** `{forecast['followers']:,}` kullanıcı")
                    st.markdown(f"• **Dinamik İstek Listesi Oranı (W/F):** `{forecast['wl_ratio']:.2f}x`")
                    st.markdown(f"• **Hesaba Katılan İstek Listesi:** `~{forecast['wishlists']:,}`")
                    st.markdown(f"• **1. Ay Satış Dönüşüm Çarpanı:** `{forecast['mult_p50']:.4f}x`")

            with tab2:
                has_cn_str = "✅ Mevcut (Asya pazarında yüksek dönüşüm potansiyeli)" if game_data['has_chinese'] else "❌ Yok (Yalnızca Batı pazarı ağırlıklı)"
                st.markdown(f"""
                <div style="background-color: #FFFFFF; border: 1px solid #ECE2D8; border-radius: 14px; padding: 18px; margin-bottom: 14px;">
                    <div class="font-display" style="font-weight: 700; color: #810541; font-size: 1.05rem; margin-bottom: 6px;">🌍 Dil ve Küresel Pazar Analizi</div>
                    <div style="color: #6C5D65; font-size: 0.85rem;">Steam mağaza sayfasından tespit edilen bölgesel pazar göstergeleri:</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"• **Desteklenen Dil Sayısı:** `{game_data['lang_count']}` dil")
                st.markdown(f"• **Çince Dil Desteği:** {has_cn_str}")
                st.markdown(f"• **Kategoriler:** {', '.join(game_data['categories']) if game_data['categories'] else 'Genel'}")

            with tab3:
                st.json({
                    "appid": appid,
                    "name": game_data["name"],
                    "followers": forecast["followers"],
                    "wishlists": forecast["wishlists"],
                    "price_status": game_data["price_status"],
                    "price_usd": game_data["price_usd"],
                    "is_free": game_data["is_free"],
                    "release_date": game_data["release_date"],
                    "engine": forecast["used_engine"],
                    "tags": game_data["tags"],
                    "genres": game_data["genres"]
                })
            
            st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="background: linear-gradient(180deg, #FFFFFF 0%, #FCF9F6 100%); border: 1px solid #ECE2D8; border-radius: 20px; padding: 48px 24px; text-align: center; margin-top: 20px; box-shadow: 0 4px 20px rgba(30,18,24,0.03);">
        <div style="width: 56px; height: 56px; margin-bottom: 16px; display: inline-flex; align-items: center; justify-content: center;">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#810541" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M6 4h12v10H6z"></path>
                <path d="M9 14v4M15 14v4"></path>
                <circle cx="12" cy="6" r="1.5"></circle>
            </svg>
        </div>
        <h2 class="font-display" style="color: #1E1218; font-size: 1.35rem; font-weight: 800; margin-bottom: 8px;">Steam Pre-Launch Sales Predictor</h2>
        <p style="color: #7A6973; font-size: 0.92rem; max-width: 580px; margin: 0 auto; line-height: 1.6;">
            Enter a Steam App ID or game name above to generate a pre-launch sales forecast.
        </p>
        <div style="margin-top: 16px;">
            Try: <button class="quick-try-btn" onclick="document.querySelector('input[placeholder*=Steam]').value='892970'; document.querySelector('input[placeholder*=Steam]').dispatchEvent(new Event('change', {bubbles: true}));">Valheim</button>
            <button class="quick-try-btn" onclick="document.querySelector('input[placeholder*=Steam]').value='3393110'; document.querySelector('input[placeholder*=Steam]').dispatchEvent(new Event('change', {bubbles: true}));">AION 2</button>
            <button class="quick-try-btn" onclick="document.querySelector('input[placeholder*=Steam]').value='2067820'; document.querySelector('input[placeholder*=Steam]').dispatchEvent(new Event('change', {bubbles: true}));">RetroSpace</button>
        </div>
    </div>
    """, unsafe_allow_html=True)
