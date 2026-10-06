import requests
import time
from pymongo import MongoClient

# 1. MongoDB Bağlantısı ve Koleksiyon Ayarları
client = MongoClient("mongodb://localhost:27017/")
db = client["steam_prediction_project"]
collection = db["games_metadata"]

# Test aşamasında eski verilerin üstüne yazmamak için tabloyu her seferinde temizliyoruz.
# Proje canlıya çıkarken bu satırı kaldıracağız.
collection.drop()

print("SteamSpy API'den Zemin Verisi (Ground Truth) çekimi başlıyor...")

sayfa = 0
# İlk etapta mimariyi test etmek için 5 sayfa (yaklaşık 5000 oyun) çekiyoruz. 
# Başarılı olunca bu limiti kaldırıp tüm veritabanını indirebilirsin.
max_sayfa = 5 
toplam_kayit = 0

while sayfa < max_sayfa:
    # SteamSpy API pagination (sayfalama) uç noktası
    url = f"https://steamspy.com/api.php?request=all&page={sayfa}"
    
    try:
        response = requests.get(url)
        
        # Sınır aşımı durumunda bekleme (Exponential Backoff mantığının temeli)
        if response.status_code == 429:
            print(f"Rate limit (429) uyarısı! 10 saniye bekleniyor...")
            time.sleep(10)
            continue
            
        if response.status_code != 200:
            print(f"Hata Kodu: {response.status_code}. Sayfa: {sayfa}")
            break
            
        data = response.json()
        
        if not data:
            print("Çekilecek sayfa kalmadı.")
            break
            
        # Gelen veriyi (Dictionary) üzerinde işlem yapabilmek için Listeye çeviriyoruz
        oyun_listesi = list(data.values())
        
        # VERİ TEMİZLEME (Data Cleansing) AŞAMASI
        for oyun in oyun_listesi:
            owners_str = oyun.get('owners', '0 .. 0')
            try:
                # '20,000 .. 50,000' aralığını alıp model için net bir sayıya (35000) dönüştürüyoruz
                parts = owners_str.replace(',', '').split(' .. ')
                if len(parts) == 2:
                    avg_sales = (int(parts[0]) + int(parts[1])) // 2
                    oyun['estimated_sales'] = avg_sales # Target Variable (Hedef Değişken)
                else:
                    oyun['estimated_sales'] = 0
            except Exception:
                oyun['estimated_sales'] = 0
                
            # Verinin çekildiği günü damgalıyoruz (Time-series analizi için çok kritik)
            oyun['snapshot_date'] = time.strftime('%Y-%m-%d')

        # 3. MongoDB'ye Toplu Kayıt (Bulk Insert)
        if oyun_listesi:
            collection.insert_many(oyun_listesi)
            toplam_kayit += len(oyun_listesi)
            print(f"Sayfa {sayfa} başarıyla MongoDB'ye yazıldı. Toplam Oyun: {toplam_kayit}")
            
        sayfa += 1
        
        # SteamSpy API'sini yormamak ve engellenmemek için güvenli bekleme süresi
        time.sleep(1.5)
        
    except Exception as e:
        print(f"Beklenmeyen bir hata oluştu: {e}")
        break

print(f"\nİşlem Tamamlandı! Toplam {toplam_kayit} oyun MongoDB 'games_metadata' tablosuna kaydedildi.") 