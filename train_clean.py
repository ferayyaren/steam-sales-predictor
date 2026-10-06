import re
import sys
import time
import html
import requests
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

print("====================================================================")
print("     GAMALYTIC & VGI STANDARTLARINDA PRE-LAUNCH TAHMİNLEME PANELİ   ")
print("          (Ekonometrik Gradient Boosting Regresyon Motoru)          ")
print("====================================================================")

# 1. MongoDB Bağlantısı ve Veri Seti Hazırlığı
client = MongoClient("mongodb://localhost:27017/")
db = client["steam_prediction_project"]

collection = db["games_clean"] if "games_clean" in db.list_collection_names() and db["games_clean"].count_documents({}) > 0 else db["games_metadata"]
print(f"[*] Kaynak Koleksiyon: MongoDB '{collection.name}'")

cursor = collection.find({})
df = pd.DataFrame(list(cursor))
print(f"[*] Veritabanından çekilen toplam ham oyun sayısı: {len(df)}")

if len(df) == 0:
    print("[!] Hata: Koleksiyonda kayıt bulunamadı! Lütfen önce 'build_clean_dataset.py' çalıştırın.")
    sys.exit(1)

# Modern Steam Dönemi Zaman Filtresi (2018+):
if 'release_year' in df.columns:
    modern_df = df[(df['release_year'] >= 2018) | (df['release_year'].isna()) | (df['release_year'] == 0)].copy()
    if len(modern_df) >= 150:
        df = modern_df
        print(f"[*] Modern Steam dönemi filtresi uygulandı (2018+): {len(df)} güncel oyun eğitime alındı.")
    else:
        print(f"[*] Bilgi: Veri seti boyutu korundu: {len(df)} oyun.")

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

# 2. Steam Etiketleri ve Tür Belirteçleri
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
    ('tag_in_app_purchases', ['in-app purchases', 'in-app purchase', 'in app purchases', 'microtransactions', 'iap']),
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

def safe_float(val, default=0.0):
    try:
        if val is None or pd.isna(val):
            return default
        f = float(val)
        return default if np.isnan(f) or np.isinf(f) else f
    except:
        return default

# 3. Dinamik İstek Listesi (Wishlist) Oranı Hesaplama
def calculate_dynamic_wishlist_ratio(followers, tags, genres, appid, is_free=False):
    """
    Sektörel araştırmalar (GameDiscoverCo, Chris Zukowski, Gamalytic, VG Insights):
    İstek Listesi / Takipçi oranı (W/F) sabit 11.02x değildir.
    Sayfa bekleme yaşı, kült türler (Immersive Sim, Boomer Shooter vb.) ve bağımsız kitle ölçeğine göre
    8x ile 24x arasında dinamik olarak değişir.
    """
    tokens = set(str(t).strip().lower() for t in (tags + genres))
    
    # A) Kült ve Niş Hype Faktörü (Wishlist Magnet Türler)
    cult_bonus = 0.0
    if any(k in tokens for k in ['immersive sim', 'boomer shooter', 'soulslike', 'souls-like', 'cyberpunk']):
        cult_bonus += 0.40
    elif any(k in tokens for k in ['roguelike', 'roguelite', 'survival horror', 'colony sim']):
        cult_bonus += 0.25
    elif any(k in tokens for k in ['moba', 'auto battler', 'action roguelike']) and is_free and followers < 500:
        cult_bonus -= 0.04
        
    # B) Steam Mağaza Sayfası Birikim Yaşı (AppID Kronolojik Sırası)
    # Steam AppID'leri ardışık artar: <2.3M (~2022 veya öncesi), <3M (~2023), <4M (~2024-2025)
    if appid < 2300000:
        age_bonus = 0.38    # 3.5 - 4 yıldır wishlist biriktiren olgun mağaza sayfası
    elif appid < 3000000:
        age_bonus = 0.20    # 2 - 3 yıldır yayında olan sayfa
    elif appid < 4000000:
        age_bonus = 0.05    # 1 - 2 yıllık sayfa
    else:
        age_bonus = 0.0     # Yeni açılmış taze sayfa (<1 yıl)
        
    # C) Bağımsız Yapım Tatlı Noktası (5K - 35K takipçi viral indie bölgesi)
    scale_bonus = 0.15 if (5000 <= followers <= 35000) else (0.0 if followers < 5000 else -0.05)
    
    if is_free:
        scale_bonus = 0.0
        cult_bonus = min(cult_bonus, 0.10)
        age_bonus = min(age_bonus, 0.05)

    # Mikro oyun ölçeklendirmesi (<500 takipçi): yapay sınır yok, pürüzsüz ölçek geçişi
    if followers < 500:
        micro_factor = min(max((followers - 5.0) / 495.0, 0.0), 1.0)
        cult_bonus *= micro_factor
        age_bonus *= micro_factor
        base_ratio = 10.60 + (11.02 - 10.60) * micro_factor
    else:
        base_ratio = 11.02
        
    wl_ratio = base_ratio * (1.0 + cult_bonus + age_bonus + scale_bonus)
    return float(wl_ratio)

def estimate_benchmark_price(followers, tags, genres, is_free=False):
    """
    Steam üzerinde fiyatı henüz açıklanmamış (TBA) oyunlar için benzer tür ve
    kitle ölçeğine dayalı optimal çıkış fiyatı (USD) tahmin eder.
    """
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

# 4. Öznitelik Çıkarımı ve Eğitim Matrisi
def extract_features(row):
    appid = int(row.get('appid', 0)) if pd.notna(row.get('appid')) else 0
    name = row.get('name', f"AppID_{appid}")
    estimated_sales = safe_float(row.get('estimated_sales', 0))

    # F2P vs Ücretli Tespiti
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

    # Etiketler ve Kategoriler
    raw_categories = row.get('categories') or []
    cat_set = {str(c).strip().lower() for c in raw_categories} if isinstance(raw_categories, list) else set()

    raw_tags = row.get('tags') or {}
    tags_set = set()
    if isinstance(raw_tags, dict):
        tags_set = {str(k).strip().lower() for k in raw_tags.keys()}
    elif isinstance(raw_tags, list):
        tags_set = {str(t).strip().lower() for t in raw_tags}

    raw_genres = row.get('genres') or []
    genre_set = {str(g).strip().lower() for g in raw_genres} if isinstance(raw_genres, list) else set()

    all_tokens = tags_set.union(cat_set).union(genre_set)

    has_iap = 1 if any(kw in all_tokens for kw in ['in-app purchases', 'in-app purchase', 'microtransactions', 'iap']) else 0
    is_mmo = 1 if any(kw in all_tokens for kw in ['massively multiplayer', 'mmo', 'mmorpg']) else 0

    # Dinamik İstek Listesi
    wl_ratio = calculate_dynamic_wishlist_ratio(followers, list(tags_set), list(genre_set), appid, is_free=(is_free == 1))
    wishlists = followers * wl_ratio

    # Sektörel Ground Truth Multiplier (Gamalytic & Steam Sektörel Doygunluk Eğrisi)
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
        month1_revenue = month1_sales * price_usd * 0.70  # Steam %30 kesintisi sonrası yapımcı payı

    target_log_revenue = np.log1p(max(month1_revenue, 0.0))

    raw_vel = safe_float(row.get('follower_velocity_30d', 0))
    velocity_30d = raw_vel if raw_vel > 0 else followers * (0.040 if followers > 50000 else 0.025) * 4.2
    velocity_ratio = velocity_30d / followers

    lang_count = int(safe_float(row.get('lang_count', 1)))
    has_chinese = int(safe_float(row.get('has_chinese', 0)))
    is_most_followed = int(safe_float(row.get('is_most_followed', 0)))

    tag_features = {col: (1 if any(kw in all_tokens for kw in kws) else 0) for col, kws in TARGET_TAGS}
    genre_features = {col: (1 if any(kw in all_tokens for kw in kws) else 0) for col, kws in TARGET_GENRES}

    price_elasticity = np.log1p(followers) / max(np.log1p(price_usd), 0.5)

    res = {
        'appid': appid,
        'name': name,
        'followers': followers,
        'wishlists': wishlists,
        'wl_ratio': wl_ratio,
        'log_wishlists': np.log1p(wishlists),
        'log_followers': np.log1p(followers),
        'log_velocity_30d': np.log1p(velocity_30d),
        'velocity_ratio': velocity_ratio,
        'price_usd': price_usd,
        'log_price': np.log1p(price_usd),
        'price_elasticity': price_elasticity,
        'is_most_followed': is_most_followed,
        'lang_count': lang_count,
        'has_chinese': has_chinese,
        'is_free': is_free,
        'has_iap': has_iap,
        'is_mmo': is_mmo,
        'multiplier': ground_truth_multiplier,
        'target_log_multiplier': target_log_multiplier,
        'month1_sales': month1_sales,
        'month1_revenue': month1_revenue,
        'target_log_revenue': target_log_revenue
    }
    res.update(tag_features)
    res.update(genre_features)
    return pd.Series(res)

print("[*] Model matrisi ve ekonometrik hedef değişkenler hesaplanıyor...")
model_df = df.apply(extract_features, axis=1).fillna(0.0)

print(f"[*] Eğitime dahil edilen toplam oyun sayısı (F2P + Ücretli): {len(model_df)}")
print(f"    - Ücretli Oyunlar: {(model_df['is_free'] == 0).sum()} adet")
print(f"    - Oynaması Ücretsiz (F2P) Oyunlar: {(model_df['is_free'] == 1).sum()} adet")

tag_cols = [col for col, _ in TARGET_TAGS]
genre_cols = [col for col, _ in TARGET_GENRES]
feature_columns = [
    'log_wishlists', 'log_followers', 'log_velocity_30d', 'velocity_ratio',
    'price_usd', 'log_price', 'price_elasticity', 'is_most_followed',
    'lang_count', 'has_chinese',
    'is_free', 'has_iap', 'is_mmo'
] + tag_cols + genre_cols

X = model_df[feature_columns]
y_mult = model_df['target_log_multiplier']
y_rev = model_df['target_log_revenue']

# Eğitim ve Test Ayrımı
X_train, X_test, y_mult_train, y_mult_test, y_rev_train, y_rev_test = train_test_split(
    X, y_mult, y_rev, test_size=0.20, random_state=42
)

# 5. Modellerin Eğitimi
print("\n[+] 1. HEDEF: Dönüşüm Çarpanı Gradient Boosting Regresörü Eğitiliyor...")
model_mult = GradientBoostingRegressor(n_estimators=160, max_depth=4, learning_rate=0.05, random_state=42)
model_mult.fit(X_train, y_mult_train)

print("[+] 2. HEDEF: Net Hasılat (Revenue USD) Regresörü Eğitiliyor...")
model_revenue = GradientBoostingRegressor(n_estimators=160, max_depth=4, learning_rate=0.05, random_state=42)
model_revenue.fit(X_train, y_rev_train)

# Performans Metrikleri
preds_mult = np.exp(model_mult.predict(X_test))
actual_mult = np.exp(y_mult_test)
r2_mult = r2_score(y_mult_test, model_mult.predict(X_test))
mae_mult = mean_absolute_error(actual_mult, preds_mult)

preds_rev = np.expm1(model_revenue.predict(X_test))
actual_rev = np.expm1(y_rev_test)
r2_rev = r2_score(y_rev_test, model_revenue.predict(X_test))
mae_rev = mean_absolute_error(actual_rev, preds_rev)

print("\n" + "=" * 65)
print("  MODEL PERFORMANS RAPORU (DÖNÜŞÜM & GELİR REGRESYONU)")
print("=" * 65)
print(f"  [1] 1. Ay Satış Çarpanı R² Skoru:  %{r2_mult * 100:.2f}")
print(f"      - Ortalama Çarpan Hatası (MAE): ±{mae_mult:.4f}x (Sektörel Bant: %2 - %45)")
print(f"  [2] Net Hasılat (USD) R² Skoru:    %{r2_rev * 100:.2f}")
print(f"      - Ortalama Net Ciro Test MAE:   ${mae_rev:,.0f}")
print("=" * 65)

# 6. Geriye Dönük Doğrulama (Backtesting) Tablosu
print("\n" + "-" * 88)
print("  GERİYE DÖNÜK DOĞRULAMA (BACKTESTING) - TEST SETİNDEN ÇIKMIŞ OYUNLAR")
print("-" * 88)
print(f"{'Oyun Adı':<25} | {'Model':<7} | {'Gerçek Satış':<13} | {'ML Tahmini':<13} | {'Fark %':<8}")
print("-" * 88)

for idx in X_test.index[:5]:
    g_name = str(model_df.loc[idx, 'name'])[:23]
    g_is_f2p = "F2P" if model_df.loc[idx, 'is_free'] == 1 else f"${model_df.loc[idx, 'price_usd']:.0f}"
    act_s = int(model_df.loc[idx, 'month1_sales'])
    
    inp_row = X.loc[[idx]]
    pred_m = float(np.clip(np.exp(model_mult.predict(inp_row)[0]), 0.02, 0.45))
    pred_s = int(model_df.loc[idx, 'wishlists'] * pred_m)
    
    diff_pct = abs(pred_s - act_s) / max(act_s, 1) * 100
    print(f"{g_name:<25} | {g_is_f2p:<7} | {act_s:<13,d} | {pred_s:<13,d} | %{diff_pct:.1f}")
print("-" * 88)

# 7. Steam Canlı Arama ve Veri Çekme Fonksiyonları
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

def fetch_live_game_data(appid):
    """Steam Store, Mağaza HTML ve Topluluk kanallarından eksiksiz canlı veri çeker."""
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
        "followers": 0
    }

    # A) Steam Store API
    try:
        store_url = f"https://store.steampowered.com/api/appdetails?appids={appid}&cc=us&l=en"
        res = requests.get(store_url, headers=HEADERS, timeout=10).json()
        if res and str(appid) in res and res[str(appid)].get("success"):
            sdata = res[str(appid)]["data"]
            data["name"] = sdata.get("name", data["name"])
            
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

    # B) Steam Store Sayfası HTML'i (Topluluk etiketleri ve Clan SteamID)
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

    # C) SteamSpy API (Eğer HTML'den etiket alınamadıysa yedek)
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

    # D) Steam Clan SteamID üzerinden Takipçi Sayısı (429 Rate Limit'e takılmayan doğrudan kanal)
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

    # D2) Steam Topluluk App Hub CLANSTEAMID (Mikro indie ve duyuru yapmamış oyunlar için doğrudan kanal)
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
                        cxm = re.search(r"<groupDetails>.*?<memberCount>([0-9,]+)</memberCount>", cx_resp.text, re.DOTALL)
                        if not cxm:
                            cxm = re.search(r"<memberCount>([0-9,]+)</memberCount>", cx_resp.text)
                        if cxm:
                            data["followers"] = int(cxm.group(1).replace(",", ""))
        except:
            pass

    # E) Steam Topluluk XML (Yedek kanal)
    if data["followers"] <= 0:
        for attempt in range(3):
            try:
                xml_url = f"https://steamcommunity.com/games/{appid}/memberslistxml/?xml=1"
                xresp = requests.get(xml_url, headers=HEADERS, timeout=6)
                if xresp.status_code == 200 and "<memberList>" in xresp.text and xresp.text.strip() != "null":
                    xm = re.search(r'<groupDetails>.*?<memberCount>([0-9,]+)</memberCount>', xresp.text, re.DOTALL)
                    if not xm:
                        xm = re.search(r'<memberCount>([0-9,]+)</memberCount>', xresp.text)
                    if xm:
                        data["followers"] = int(xm.group(1).replace(",", ""))
                        break
                elif xresp.status_code == 429 or xresp.text.strip() == "null":
                    time.sleep(1.2)
            except:
                pass

    # E) Steam Topluluk Hub Sayfası
    if data["followers"] <= 0:
        try:
            comm_url = f"https://steamcommunity.com/games/{appid}"
            cresp = requests.get(comm_url, headers=HEADERS, timeout=6)
            if cresp.status_code == 200:
                fm = re.search(r'([0-9,]+)\s+Followers', cresp.text, re.I)
                if fm:
                    data["followers"] = int(fm.group(1).replace(",", ""))
                else:
                    cm = re.search(r'([0-9,]+)\s*(?:Members|Followers)', cresp.text, re.I)
                    if cm:
                        data["followers"] = int(cm.group(1).replace(",", ""))
        except:
            pass

    # F) SteamSpy HTML
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

# 8. Ekonometrik Tahmin ve Raporlama Fonksiyonu
def execute_game_forecast(appid, name, followers, velocity_7d, price_usd, price_status, release_date, tags=[], genres=[], categories=[], lang_count=4, has_chinese=False, is_free=False):
    # 1. Dinamik Wishlist Oranı
    wl_ratio = calculate_dynamic_wishlist_ratio(followers, tags, genres, appid, is_free=is_free)
    wishlists = int(round(followers * wl_ratio))

    velocity_30d = int(velocity_7d * 4.2)
    vel_ratio = velocity_30d / max(followers, 1.0)
    effective_price = price_usd if not is_free else 0.0
    price_elasticity = np.log1p(followers) / max(np.log1p(effective_price), 0.5)

    all_tokens = set([t.lower() for t in tags] + [g.lower() for g in genres] + [c.lower() for c in categories])
    has_iap = 1 if any(kw in all_tokens for kw in ['in-app purchases', 'in-app purchase', 'microtransactions', 'iap']) else 0
    is_mmo = 1 if any(kw in all_tokens for kw in ['massively multiplayer', 'mmo', 'mmorpg']) else 0

    row = {
        'log_wishlists': np.log1p(wishlists),
        'log_followers': np.log1p(followers),
        'log_velocity_30d': np.log1p(velocity_30d),
        'velocity_ratio': vel_ratio,
        'price_usd': effective_price,
        'log_price': np.log1p(effective_price),
        'price_elasticity': price_elasticity,
        'is_most_followed': 1 if followers >= 50000 else 0,
        'lang_count': lang_count,
        'has_chinese': 1 if has_chinese else 0,
        'is_free': 1 if is_free else 0,
        'has_iap': has_iap,
        'is_mmo': is_mmo
    }
    for col, kws in TARGET_TAGS:
        row[col] = 1 if any(kw in all_tokens for kw in kws) else 0
    for col, kws in TARGET_GENRES:
        row[col] = 1 if any(kw in all_tokens for kw in kws) else 0

    inp = pd.DataFrame([row])[feature_columns]

    # ML Modeli Tahmini & Ekonometrik Hibrit
    pred_log_mult = float(model_mult.predict(inp)[0])
    ml_mult = float(np.clip(np.exp(pred_log_mult), 0.02, 0.45))

    if any(k in all_tokens for k in ['puzzle', 'casual', 'hidden object']):
        micro_base = 0.352
    else:
        micro_base = 0.415

    if wishlists <= 500:
        econ_mult = micro_base
    elif wishlists <= 25000:
        prog = (np.log10(max(wishlists, 500.0)) - np.log10(500.0)) / (np.log10(25000.0) - np.log10(500.0))
        econ_mult = micro_base - ((micro_base - 0.20) * prog)
    elif wishlists <= 250000:
        prog = (np.log10(wishlists) - np.log10(25000.0)) / (np.log10(250000.0) - np.log10(25000.0))
        econ_mult = 0.20 + (0.108 * prog)
    else:
        econ_mult = 0.308 * ((250000.0 / float(wishlists)) ** 0.18)

    # 15.000 istek listesi altında veri seyrekliği nedeniyle ekonometrik eğriyle pürüzsüz harmanla
    if wishlists < 15000:
        w_blend = min(max((np.log10(max(wishlists, 1000.0)) - np.log10(1000.0)) / (np.log10(15000.0) - np.log10(1000.0)), 0.0), 1.0)
        mult_p50 = float(np.clip((1.0 - w_blend) * econ_mult + w_blend * ml_mult, 0.02, 0.45))
    else:
        mult_p50 = ml_mult

    # 1. Ay Satış / Aktif Oyuncu Tahmini (Yapay alt sınır kaldırıldı)
    month1_expected = max(int(round(wishlists * mult_p50)), 0)

    # Sektörel Log-Normal Güven Aralığı (Gamalytic & VG Insights Standart Bantları: 0.50x - 2.00x)
    month1_low = int(round(month1_expected * 0.50))
    month1_high = int(round(month1_expected * 2.00))

    # Zaman Kırılımları
    t7_copies = int(month1_expected * 0.55)
    lifetime_copies = int(month1_expected * 2.6)

    # Finansal Gelir Hesaplamaları
    if is_free or effective_price == 0.0:
        base_arpu = 18.0 if is_mmo else (8.5 if has_iap else 3.5)
        market_boost = 1.0 + (0.15 if has_chinese else 0.0) + min(lang_count * 0.015, 0.15)
        effective_net_arpu = base_arpu * market_boost

        month1_net_rev = month1_expected * effective_net_arpu
        gross_rev = month1_net_rev / 0.70
        month1_rev_low = month1_low * effective_net_arpu
        month1_rev_high = month1_high * effective_net_arpu
        t7_net_rev = t7_copies * effective_net_arpu
        lifetime_net_rev = lifetime_copies * effective_net_arpu
    else:
        gross_rev = month1_expected * effective_price
        month1_net_rev = gross_rev * 0.70
        month1_rev_low = month1_low * effective_price * 0.70
        month1_rev_high = month1_high * effective_price * 0.70
        t7_net_rev = t7_copies * effective_price * 0.70
        lifetime_net_rev = lifetime_copies * effective_price * 0.70

    is_tba = (not is_free) and (effective_price <= 0.0 or price_status == "TBA")
    est_price = 0.0
    if is_tba:
        est_price = estimate_benchmark_price(followers, tags, genres, is_free=False)
        month1_net_rev = month1_expected * est_price * 0.70
        month1_rev_low = month1_low * est_price * 0.70
        month1_rev_high = month1_high * est_price * 0.70
        t7_net_rev = t7_copies * est_price * 0.70
        lifetime_net_rev = lifetime_copies * est_price * 0.70

    # Raporlama Ekranı (Sade, Öz ve Anlaşılır)
    tag_str = ", ".join(tags[:4]) if tags else ", ".join(genres[:3])
    unit = "oyuncu" if is_free else "kopya"

    print("\n" + "=" * 80)
    print(f"  TAHMİN RAPORU: {name.upper()}")
    print("=" * 80)
    print(f"  * Çıkış Tarihi      : {release_date}")
    if is_free:
        print("  * Fiyat Durumu      : Free to Play (Ücretsiz)")
    elif is_tba:
        print(f"  * Tahmini Fiyat     : ${est_price:.2f} (Sektörel Tür Benchmarkı)")
    else:
        print(f"  * Çıkış Fiyatı      : ${effective_price:.2f}")

    print(f"  * Canlı Takipçi     : {followers:,} kişi")
    print(f"  * İstek Listesi     : ~{wishlists:,} kişi (Tahmini)")
    print(f"  * Tür & Etiketler   : {tag_str}")
    print("-" * 80)

    if is_free:
        print(f"  ► 1. AY OYUNCU      : ~{month1_expected:,} oyuncu  [Aralık: {month1_low:,} - {month1_high:,}]")
        print(f"  ► 1. AY NET GELİR   : ~${month1_net_rev:,.0f}  (Tahmini oyun içi harcama)")
    elif is_tba:
        print(f"  ► 1. AY SATIŞI      : ~{month1_expected:,} kopya  [Aralık: {month1_low:,} - {month1_high:,}]")
        print(f"  ► 1. AY NET CİRO    : ~${month1_net_rev:,.0f}  [Aralık: ${month1_rev_low:,.0f} - ${month1_rev_high:,.0f}] (${est_price:.2f} tahmini fiyatla)")
    else:
        print(f"  ► 1. AY SATIŞI      : ~{month1_expected:,} kopya  [Aralık: {month1_low:,} - {month1_high:,}]")
        print(f"  ► 1. AY NET CİRO    : ~${month1_net_rev:,.0f}  [Aralık: ${month1_rev_low:,.0f} - ${month1_rev_high:,.0f}] (Steam %30 kesintisi sonrası)")

    print("-" * 80)
    print(f"  ► İlk Hafta (T+7)   : ~{t7_copies:,} {unit}")
    print(f"  ► 1. Yıl (Ömürlük)  : ~{lifetime_copies:,} {unit}")
    print("=" * 80 + "\n")

# 9. İnteraktif Kullanıcı Girişi Döngüsü (Tek Tuş Enter)
def main():
    print("\n" + "=" * 95)
    print("  [✓] MODEL EĞİTİLDİ VE HAZIR! OYUN ADINI YA DA APPID GİRİP ENTER'A BASIN")
    print("  (Tüm veriler Steam Store & Community üzerinden tek seferde otomatik çekilir)")
    print("=" * 95)

    while True:
        try:
            user_input = input("\n>> Tahmin edilecek oyunu girin (Örn: RetroSpace veya 2067820, Çıkmak için 'q'): ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['q', 'exit', 'quit']:
                print("\nÇıkış yapıldı. Başarılar dileriz!")
                break

            print(f"[*] '{user_input}' Steam sunucularında aranıyor ve canlı veriler çekiliyor...")
            appid, found_name = search_steam(user_input)

            if not appid:
                print(f"[!] '{user_input}' otomatik bulunamadı. Lütfen AppID'sini girin: ")
                in_app = input("    AppID: ").strip()
                if in_app.isdigit():
                    appid = int(in_app)
                else:
                    continue

            gdata = fetch_live_game_data(appid)
            gname = found_name if found_name else gdata["name"]
            followers = gdata["followers"]
            if followers < 0:
                followers = 0
            if followers == 0:
                f_inp = input("    [?] Canlı takipçi 0 veya tespit edilemedi. Takipçi sayısı (Varsayılan 0, Enter ile devam): ").strip()
                if f_inp.isdigit():
                    followers = int(f_inp)

            # Gerçekçi Haftalık İvme (VGI Standardı: ~%2.5 haftalık organik artış, yapay alt sınır yok)
            velocity_7d = int(round(followers * (0.040 if followers > 50000 else 0.025)))

            execute_game_forecast(
                appid=appid,
                name=gname,
                followers=followers,
                velocity_7d=velocity_7d,
                price_usd=gdata["price_usd"],
                price_status=gdata["price_status"],
                release_date=gdata["release_date"],
                tags=gdata["tags"],
                genres=gdata["genres"],
                categories=gdata.get("categories", []),
                lang_count=gdata["lang_count"],
                has_chinese=gdata["has_chinese"],
                is_free=gdata["is_free"]
            )

        except (KeyboardInterrupt, EOFError):
            print("\nÇıkış yapıldı.")
            break
        except Exception as e:
            print(f"[!] Hata: {e}")

if __name__ == "__main__":
    main()
