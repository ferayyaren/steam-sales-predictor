import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

print("====================================================================")
print("      PRE-LAUNCH TAHMİNLEME MOTORU   ")
print("====================================================================")

# 1. MongoDB Bağlantısı ve Veri Çekme
client = MongoClient("localhost", 27017)
db = client["steam_prediction_project"]
collection = db["games_metadata"]

cursor = collection.find({})
df = pd.DataFrame(list(cursor))
print(f"[*] MongoDB 'games_metadata' koleksiyonundan çekilen kayıt: {len(df)}")

if len(df) == 0:
    print("[!] Hata: 'games_metadata' koleksiyonunda veri bulunamadı!")
    exit()

# 2. Etiket ve Tür Tanımları (One-Hot)
TARGET_TAGS = [
    ('tag_multiplayer', ['multiplayer', 'multi-player', 'online co-op']),
    ('tag_singleplayer', ['singleplayer', 'single-player']),
    ('tag_coop', ['co-op', 'coop', 'cooperative']),
    ('tag_story_rich', ['story rich', 'great soundtrack', 'lore-rich']),
    ('tag_souls_like', ['souls-like', 'soulslike', 'difficult']),
    ('tag_roguelike', ['roguelike', 'roguelite', 'rogue-like', 'rogue-lite']),
    ('tag_open_world', ['open world', 'open-world']),
    ('tag_survival', ['survival', 'survival horror']),
    ('tag_shooter', ['shooter', 'fps', 'third-person shooter']),
    ('tag_action_roguelike', ['action roguelike', 'bullet hell']),
    ('tag_early_access', ['early access']),
    ('tag_horror', ['horror', 'psychological horror']),
    ('tag_sandbox', ['sandbox', 'crafting', 'building']),
    ('tag_strategy', ['strategy', 'turn-based strategy', 'rts']),
    ('tag_rpg', ['rpg', 'role-playing', 'action rpg', 'jrpg'])
]

TARGET_GENRES = [
    ('genre_action', ['action', 'aksiyon']),
    ('genre_adventure', ['adventure', 'macera']),
    ('genre_rpg', ['rpg', 'rol yapma']),
    ('genre_strategy', ['strategy', 'strateji']),
    ('genre_simulation', ['simulation', 'simülasyon']),
    ('genre_indie', ['indie', 'bağımsız']),
    ('genre_casual', ['casual', 'basit eğlence']),
    ('genre_sports_racing', ['sports', 'racing', 'spor', 'yarış'])
]

def safe_float(val, default=0.0):
    try:
        if val is None or pd.isna(val):
            return default
        f = float(val)
        return default if np.isnan(f) or np.isinf(f) else f
    except:
        return default

def extract_tags_and_genre(row):
    raw_tags = row.get('tags')
    tags_set = set()
    if isinstance(raw_tags, dict):
        tags_set = {str(k).strip().lower() for k in raw_tags.keys()}
    elif isinstance(raw_tags, list):
        tags_set = {str(item).strip().lower() for item in raw_tags}
    elif isinstance(raw_tags, str) and raw_tags:
        tags_set = {t.strip().lower() for t in raw_tags.split(',')}

    raw_genre = row.get('genre') or row.get('genres') or row.get('main_genre') or ''
    genre_set = set()
    if isinstance(raw_genre, list):
        for g in raw_genre:
            if isinstance(g, dict):
                genre_set.add(str(g.get('description', '')).strip().lower())
            else:
                genre_set.add(str(g).strip().lower())
    elif isinstance(raw_genre, str) and raw_genre:
        genre_set = {g.strip().lower() for g in raw_genre.split(',')}

    genre_set.update(tags_set)
    return tags_set, genre_set

# 3. Öznitelik Çıkarımı (Data Leakage Yok)
def extract_pre_launch_features(row):
    appid = int(row.get('appid', 0)) if pd.notna(row.get('appid')) else 0
    name = row.get('name', f"AppID_{appid}")
    estimated_sales = safe_float(row.get('estimated_sales', 0))

    initial_price = safe_float(row.get('initialprice', 0)) / 100.0
    reg_prices = row.get('regional_prices') if isinstance(row.get('regional_prices'), dict) else {}

    price_us = safe_float(reg_prices.get('price_US', 0))
    if price_us <= 0:
        price_us = initial_price if initial_price > 0 else 19.99

    price_cn = safe_float(reg_prices.get('price_CN', 0))
    if price_cn <= 0:
        price_cn = price_us * 0.60

    price_tr = safe_float(reg_prices.get('price_TR', 0))
    if price_tr <= 0:
        price_tr = price_us * 0.50

    # Bölgesel Ağırlıklı Fiyat: US %50, CN %30, TR %20
    weighted_price = (price_us * 0.50) + (price_cn * 0.30) + (price_tr * 0.20)

    # Followers / Wishlist Değeri (Sızıntıyı Önleyen Bağımsız Değişken Mantığı)
    raw_followers = safe_float(row.get('followers', 0))
    if raw_followers > 0:
        followers = raw_followers
    else:
        # Hedef değişkenden türetmek yerine bağımsız log-normal piyasa simülasyonu
        followers = float(np.random.lognormal(mean=7.5, sigma=1.2))

    wishlists = followers * 11.0

    raw_vel = safe_float(row.get('follower_velocity_30d', 0))
    velocity_30d = raw_vel if raw_vel > 0 else followers * 0.18
    velocity_ratio = velocity_30d / max(followers, 1.0)

    tags_set, genre_set = extract_tags_and_genre(row)

    tag_features = {}
    for tag_col, keywords in TARGET_TAGS:
        tag_features[tag_col] = 1 if any(kw in tags_set for kw in keywords) else 0

    genre_features = {}
    for genre_col, keywords in TARGET_GENRES:
        genre_features[genre_col] = 1 if any(kw in genre_set for kw in keywords) else 0

    raw_langs = row.get('supported_languages')
    langs = str(raw_langs) if pd.notna(raw_langs) else ''
    has_chinese = 1 if ('Simplified Chinese' in langs or 'Traditional Chinese' in langs) else 0
    lang_count = len([l for l in langs.split(',') if l.strip()]) if langs else 1

    price_elasticity = np.log1p(followers) * np.log1p(weighted_price)
    net_revenue = estimated_sales * weighted_price * 0.70

    feat_dict = {
        'appid': appid,
        'name': name,
        'log_followers': np.log1p(followers),
        'log_wishlists': np.log1p(wishlists),
        'log_velocity_30d': np.log1p(velocity_30d),
        'velocity_ratio': velocity_ratio,
        'weighted_price': weighted_price,
        'log_price': np.log1p(weighted_price),
        'price_elasticity': price_elasticity,
        'has_chinese': has_chinese,
        'lang_count': lang_count,
        'followers': followers,
        'wishlists': wishlists,
        'velocity_30d': velocity_30d,
        'estimated_sales': estimated_sales,
        'net_revenue': net_revenue
    }
    feat_dict.update(tag_features)
    feat_dict.update(genre_features)
    return pd.Series(feat_dict)

print("[*] Pre-launch öznitelikleri çıkarılıyor...")
model_df = df.apply(extract_pre_launch_features, axis=1)
model_df = model_df[model_df['estimated_sales'] > 0].copy()
model_df = model_df.fillna(0.0)

print(f"[*] Modele dahil edilen oyun sayısı: {len(model_df)}")

tag_columns = [col for col, _ in TARGET_TAGS]
genre_columns = [col for col, _ in TARGET_GENRES]

feature_columns = [
    'log_followers', 'log_wishlists', 'log_velocity_30d', 'velocity_ratio',
    'weighted_price', 'log_price', 'price_elasticity', 'has_chinese', 'lang_count'
] + tag_columns + genre_columns

X = model_df[feature_columns]
y_units = np.log1p(model_df['estimated_sales'])
y_revenue = np.log1p(model_df['net_revenue'])

# 4. Model Eğitimi (Gradient Boosting Regressor)
X_train, X_test, y_train_u, y_test_u, y_train_r, y_test_r = train_test_split(
    X, y_units, y_revenue, test_size=0.20, random_state=42
)

print(f"[*] Eğitim Seti: {len(X_train)} oyun | Test Seti: {len(X_test)} oyun")

print("\n[+] 1. Kopya Satış Modeli eğitiliyor...")
model_units = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.06,
    max_depth=5,
    subsample=0.85,
    min_samples_split=6,
    random_state=42
)
model_units.fit(X_train, y_train_u)

print("[+] 2. Net Gelir Modeli eğitiliyor...")
model_revenue = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.06,
    max_depth=5,
    subsample=0.85,
    min_samples_split=6,
    random_state=42
)
model_revenue.fit(X_train, y_train_r)

# 5. Metrik Raporu
def evaluate(name, y_test_log, preds_log):
    actual = np.expm1(y_test_log)
    preds = np.expm1(preds_log)

    r2_log = r2_score(y_test_log, preds_log)
    rmse = np.sqrt(mean_squared_error(actual, preds))
    mae = mean_absolute_error(actual, preds)
    denom = (np.abs(actual) + np.abs(preds)) / 2.0
    smape = np.mean(np.where(denom == 0, 0, np.abs(preds - actual) / denom)) * 100

    print(f"\n========================================================")
    print(f"  PERFORMANS RAPORU: {name.upper()}")
    print(f"========================================================")
    print(f"  * R² Skoru (Log Uzayı):      %{r2_log * 100:.2f}")
    print(f"  * SMAPE (Bağıl Yüzde Hata):  %{smape:.2f}")
    print(f"  * MAE (Ortalama Mutlak Hata): {mae:,.2f}")
    print(f"  * RMSE (Kök Hata):            {rmse:,.2f}")
    return r2_log

evaluate("Copies Sold (Satış Adedi)", y_test_u, model_units.predict(X_test))
evaluate("Net Revenue (Net Gelir - USD)", y_test_r, model_revenue.predict(X_test))

# 6. Tahmin ve Simülasyon
def predict_pre_launch(game_name, followers, velocity_30d, price_usd, tags=[], genre='Action', has_chinese=False, lang_count=1):
    wishlists = followers * 11.0
    vel_ratio = velocity_30d / max(followers, 1.0)

    price_us = price_usd
    price_cn = price_us * 0.60
    price_tr = price_us * 0.50
    weighted_price = (price_us * 0.50) + (price_cn * 0.30) + (price_tr * 0.20)
    price_elasticity = np.log1p(followers) * np.log1p(weighted_price)

    tags_lower = [t.lower() for t in tags]
    genre_lower = genre.lower()

    row_data = {
        'log_followers': np.log1p(followers),
        'log_wishlists': np.log1p(wishlists),
        'log_velocity_30d': np.log1p(velocity_30d),
        'velocity_ratio': vel_ratio,
        'weighted_price': weighted_price,
        'log_price': np.log1p(weighted_price),
        'price_elasticity': price_elasticity,
        'has_chinese': 1 if has_chinese else 0,
        'lang_count': lang_count
    }

    for tag_col, keywords in TARGET_TAGS:
        row_data[tag_col] = 1 if any(kw in tags_lower for kw in keywords) else 0

    for genre_col, keywords in TARGET_GENRES:
        row_data[genre_col] = 1 if any(kw in genre_lower for kw in keywords) else 0

    input_df = pd.DataFrame([row_data])[feature_columns]

    pred_copies = float(np.expm1(model_units.predict(input_df)[0]))
    pred_revenue = float(np.expm1(model_revenue.predict(input_df)[0]))

    first_week_copies = int(wishlists * 0.18)
    first_week_revenue = first_week_copies * weighted_price * 0.70

    t90_copies = int(pred_copies * 0.45)
    t90_revenue = pred_revenue * 0.45

    return {
        'game_name': game_name,
        'followers': followers,
        'wishlists': int(wishlists),
        'first_week_copies': first_week_copies,
        'first_week_revenue': first_week_revenue,
        't90_copies': t90_copies,
        't90_revenue': t90_revenue,
        'lifetime_copies': int(pred_copies),
        'lifetime_revenue': pred_revenue
    }

print("\n=========================================================================================================")
print("                   PRE-LAUNCH OYUN SİMÜLASYONU VE TAHMİN                                           ")
print("=========================================================================================================")

sample = predict_pre_launch(
    game_name='Project Darkblade (Indie Souls-like)',
    followers=12000,
    velocity_30d=2800,
    price_usd=24.99,
    tags=['Souls-like', 'Dark Fantasy', 'Action', 'Singleplayer'],
    genre='Action',
    has_chinese=True,
    lang_count=6
)

print(f"Oyun Adı         : {sample['game_name']}")
print(f"Tahmini Wishlist : {sample['wishlists']:,}")
print(f"İlk Hafta Satış  : {sample['first_week_copies']:,} kopya (${sample['first_week_revenue']:,.0f})")
print(f"T+90 Gün Gelir   : ${sample['t90_revenue']:,.0f} ({sample['t90_copies']:,} kopya)")
print(f"Ömürlük Net Gelir: ${sample['lifetime_revenue']:,.0f} ({sample['lifetime_copies']:,} kopya)")
print("=========================================================================================================\n")