import requests
import time
from pymongo import MongoClient

# 1. MongoDB Bağlantısı
client = MongoClient("mongodb://localhost:27017/")
db = client["steam_prediction_project"]
collection = db["games_metadata"]

# Veritabanındaki oyunları çekiyoruz
games = list(collection.find({}))
print(f"Toplam {len(games)} oyun veritabanında mevcut.")

max_games = 1000
target_games = games[0:max_games]

print(f"İşlem başlıyor: İlk {len(target_games)} oyun için Steam'den bölgesel fiyatlar çekilecek...")

for index, game in enumerate(target_games):
    app_id = str(game['appid'])
    print(f"[{index+1}/{len(target_games)}] İşleniyor: {game['name']} (AppID: {app_id})")
    
    # Hedef bölgelerimiz: ABD (Global), Çin (En büyük pazar), Türkiye (MENA/LATAM proxy'si)
    regions = {'US': 'us', 'CN': 'cn', 'TR': 'tr'}
    regional_prices = {}
    supported_languages = ""
    
    # 2. Her bölge için Steam API'ye istek atıyoruz
    for region_name, country_code in regions.items():
        url = f"https://store.steampowered.com/api/appdetails?appids={app_id}&cc={country_code}"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                
                # Eğer oyun Steam'de bulunamadıysa (kaldırıldıysa) pas geçiyoruz
                if data and data[app_id].get('success'):
                    game_data = data[app_id]['data']
                    
                    # Fiyat Bilgisi (Cent cinsinden geldiği için 100'e bölüyoruz)
                    if 'price_overview' in game_data:
                        price = game_data['price_overview']['initial'] / 100.0
                        regional_prices[f"price_{region_name}"] = price
                    else:
                        regional_prices[f"price_{region_name}"] = 0.0
                        
                    # Dil desteği (Sadece ilk istekte almamız yeterli)
                    if region_name == 'US':
                        supported_languages = game_data.get('supported_languages', '')
                        
            # API'yi yormamak için her bölge isteğinde yarım saniye bekle
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Hata oluştu: {e}")
            
    # 3. Elde edilen bölgesel fiyatları ve dil verisini MongoDB'deki o oyuna (Document) Update ediyoruz
    update_data = {
        "$set": {
            "regional_prices": regional_prices,
            "supported_languages": supported_languages
        }
    }
    
    collection.update_one({"_id": game["_id"]}, update_data)
    print(f"--> {game['name']} güncellendi: {regional_prices}")
    
    # Steam API rate limitlerine takılmamak için oyunlar arası 1 saniye bekleme
    time.sleep(1)

print("Bölgesel veri zenginleştirme işlemi başarıyla tamamlandı!")