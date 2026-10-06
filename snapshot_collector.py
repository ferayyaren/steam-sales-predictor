import requests
import time
from datetime import datetime
from pymongo import MongoClient

# 1. MongoDB Bağlantısı ve Koleksiyonlar
client = MongoClient("mongodb://localhost:27017/")
db = client["steam_prediction_project"]
metadata_collection = db["games_metadata"]
snapshot_collection = db["game_snapshots"]

# Ana tablodaki oyunları çekiyoruz
games = list(metadata_collection.find({}, {"appid": 1, "name": 1}))
print(f"Toplam {len(games)} oyun için günlük snapshot (anlık durum) kaydı başlatılıyor...")

bugunun_tarihi = datetime.now().strftime("%Y-%m-%d")
basarili_sayisi = 0

for index, game in enumerate(games):
    app_id = str(game['appid'])
    
    # Steam Store API'den o anki güncel verileri çekiyoruz
    url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            
            if data and data[app_id].get('success'):
                game_data = data[app_id]['data']
                
                # Fiyat ve İndirim Kontrolü
                price_current = 0.0
                is_discounted = False
                if 'price_overview' in game_data:
                    price_info = game_data['price_overview']
                    price_current = price_info['final'] / 100.0
                    is_discounted = price_info.get('discount_percent', 0) > 0
                
                # Snapshot Dokümanını Oluşturma
                snapshot_doc = {
                    "appid": int(app_id),
                    "date": bugunun_tarihi,
                    "price_current": price_current,
                    "is_discounted": is_discounted,
                    "recommendations": game_data.get('recommendations', {}).get('total', 0),
                    "captured_at": datetime.now()
                }
                
                # game_snapshots koleksiyonuna kaydediyoruz
                snapshot_collection.insert_one(snapshot_doc)
                basarili_sayisi += 1
                print(f"[{index+1}/{len(games)}] Snapshot alındı: {game['name']} (Fiyat: {price_current}$, İndirimde mi: {is_discounted})")
                
        # API Rate Limit Koruması
        time.sleep(0.4)
        
    except Exception as e:
        print(f"Hata oluştu ({game['name']}): {e}")

print(f"\nSnapshot işlemi tamamlandı! Toplam {basarili_sayisi} oyunun bugünkü durumu 'game_snapshots' tablosuna işlendi.")