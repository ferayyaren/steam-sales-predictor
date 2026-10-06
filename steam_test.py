import requests

# Örnek olarak Counter-Strike 2'nin Steam App ID'si: 730
app_id = "730"
url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"

# Endpoint'e GET isteği atıyoruz
response = requests.get(url)
data = response.json() # Gelen yanıtı Python sözlüğüne çevirir

# Verinin içinden oyun adını ve ücretsiz olup olmadığını çekelim
oyun_adi = data[app_id]['data']['name']
ucretsiz_mi = data[app_id]['data']['is_free']

print(f"Oyun: {oyun_adi}")
print(f"Ücretsiz mi?: {ucretsiz_mi}")
