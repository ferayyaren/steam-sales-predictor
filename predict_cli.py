import re
import sys
import time
import html
import requests
import numpy as np
import pandas as pd
from pymongo import MongoClient
import warnings
warnings.filterwarnings('ignore')

print("====================================================================")
print("     STEAM CANLI PRE-LAUNCH TAHMİNLEME PANELİ (İNTERAKTİF CLI)      ")
print("          (Gamalytic & VG Insights Sektörel Benchmark Modeli)       ")
print("====================================================================")

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

def search_steam(query):
    """Kullanıcının girdiği oyun adını veya AppID'sini Steam'de arar."""
    query = str(query).strip()
    
    if query.isdigit():
        return int(query), None

    try:
        url = f"https://store.steampowered.com/api/storesearch/?term={requests.utils.quote(query)}&l=english&cc=US"
        resp = requests.get(url, headers=HEADERS, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("items", [])
            if items:
                best = items[0]
                return int(best["id"]), best.get("name")
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

def calculate_dynamic_wishlist_ratio(followers, tags, genres, appid, is_free=False):
    """
    Sektörel araştırmalar (GameDiscoverCo, Chris Zukowski, Gamalytic, VG Insights):
    İstek Listesi / Takipçi oranı (W/F) sabit 11.02x değildir.
    Sayfa bekleme yaşı, kült türler (Immersive Sim, Boomer Shooter vb.) ve bağımsız kitle ölçeğine göre
    8x ile 24x arasında dinamik olarak değişir. Mikro oyunlarda (<500 takipçi)
    topluluk etkisi henüz oluşmadığından Gamalytic 10.6x tabanı uygulanır.
    """
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

    # B) Steam Store Sayfası HTML'i (Topluluk Etiketleri ve Clan SteamID)
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

    # C) SteamSpy API
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

def calculate_forecast(appid, name, followers, velocity_7d, price_usd, price_status, release_date, tags, genres, categories, lang_count, has_chinese, is_free=False):
    """Gamalytic & VGI Hibrit Ekonometrik Algoritmasıyla Tahmin Yapar."""
    
    # 1. Dinamik İstek Listesi
    wl_ratio = calculate_dynamic_wishlist_ratio(followers, tags, genres, appid, is_free=is_free)
    wishlists = int(round(followers * wl_ratio))

    # 2. Gamalytic & Steam Sektörel Doygunluk Eğrisi (Yapay alt sınır kaldırıldı)
    all_tokens = set([t.lower() for t in tags] + [g.lower() for g in genres] + [c.lower() for c in categories])
    if any(k in all_tokens for k in ['puzzle', 'casual', 'hidden object']):
        micro_base = 0.352
    else:
        micro_base = 0.415

    if wishlists <= 500:
        base_mult = micro_base
    elif wishlists <= 25000:
        prog = (np.log10(max(wishlists, 500.0)) - np.log10(500.0)) / (np.log10(25000.0) - np.log10(500.0))
        base_mult = micro_base - ((micro_base - 0.20) * prog)
    elif wishlists <= 250000:
        prog = (np.log10(wishlists) - np.log10(25000.0)) / (np.log10(250000.0) - np.log10(25000.0))
        base_mult = 0.20 + (0.108 * prog)
    else:
        base_mult = 0.308 * ((250000.0 / float(wishlists)) ** 0.18)

    mult_p50 = float(np.clip(base_mult, 0.02, 0.45))

    # 3. 1. Ay Satış / Oyuncu Tahmini ve Sektörel Güven Aralığı
    month1_expected = max(int(round(wishlists * mult_p50)), 0)
    month1_low = int(round(month1_expected * 0.50))
    month1_high = int(round(month1_expected * 2.00))

    # 4. Zaman Kırılımları ve Hasılat
    t7_copies = int(month1_expected * 0.55)
    lifetime_copies = int(month1_expected * 2.6)

    effective_price = price_usd if not is_free else 0.0

    if is_free or effective_price == 0.0:
        all_tokens = set([t.lower() for t in tags] + [g.lower() for g in genres] + [c.lower() for c in categories])
        is_mmo = 1 if any(k in all_tokens for k in ['massively multiplayer', 'mmo', 'mmorpg']) else 0
        has_iap = 1 if any(kw in all_tokens for kw in ['in-app purchases', 'in-app purchase', 'microtransactions', 'iap']) else 0

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

def main():
    print("[✓] Tahmin Motoru Hazır! Çıkacak herhangi bir oyunun adını veya AppID'sini yazıp Enter'a basabilirsiniz.\n")
    
    while True:
        try:
            user_input = input(">> Tahmin edilecek oyunu girin (Çıkmak için 'q'): ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['q', 'exit', 'quit']:
                print("\nÇıkış yapılıyor. İyi çalışmalar!")
                break

            appid, found_name = search_steam(user_input)
            
            if not appid:
                print(f"[!] '{user_input}' Steam üzerinde otomatik bulunamadı.")
                appid_input = input("    Lütfen oyunun Steam AppID'sini girin (veya Enter ile iptal): ").strip()
                if appid_input.isdigit():
                    appid = int(appid_input)
                else:
                    continue

            game_data = fetch_live_game_data(appid)
            game_name = found_name if found_name else game_data["name"]
            followers = game_data["followers"]
            if followers < 0:
                followers = 0
            if followers == 0:
                f_inp = input("    [?] Canlı takipçi 0 veya tespit edilemedi. Takipçi sayısı (Varsayılan 0, Enter ile devam): ").strip()
                if f_inp.isdigit():
                    followers = int(f_inp)

            velocity_7d = int(round(followers * (0.040 if followers > 50000 else 0.025)))

            calculate_forecast(
                appid=appid,
                name=game_name,
                followers=followers,
                velocity_7d=velocity_7d,
                price_usd=game_data["price_usd"],
                price_status=game_data["price_status"],
                release_date=game_data["release_date"],
                tags=game_data["tags"],
                genres=game_data["genres"],
                categories=game_data.get("categories", []),
                lang_count=game_data["lang_count"],
                has_chinese=game_data["has_chinese"],
                is_free=game_data["is_free"]
            )

        except (KeyboardInterrupt, EOFError):
            print("\nÇıkış yapıldı.")
            break
        except Exception as e:
            print(f"[!] Bir hata oluştu: {e}\n")

if __name__ == "__main__":
    main()
