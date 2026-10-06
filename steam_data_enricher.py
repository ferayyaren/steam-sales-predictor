import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from pymongo import MongoClient, UpdateOne

# SteamSpy public API (request=appdetails) followers döndürmez.
# Sayı HTML sayfasında: https://steamspy.com/app/{appid}
FOLLOWERS_RE = re.compile(r"<strong>\s*Followers\s*</strong>\s*:\s*([0-9,]+)", re.I)

WORKERS = 6
TIMEOUT = 20

client = MongoClient("localhost", 27017)
db = client["steam_prediction_project"]
collection = db["games_metadata"]

query = {
    "$or": [
        {"followers": {"$exists": False}},
        {"followers": None},
        {"followers": 0},
    ]
}
games = list(collection.find(query, {"appid": 1, "name": 1}))
total_games = len(games)
print(f"[*] Toplam {total_games} oyun için SteamSpy followers çekimi başlıyor...")
print("[*] Kaynak: steamspy.com HTML (API'de followers yok)")

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_followers(app_id):
    url = f"https://steamspy.com/app/{app_id}"
    last_error = None
    for attempt in range(1, 4):
        try:
            resp = requests.get(url, headers=headers, timeout=TIMEOUT)
            if resp.status_code == 429:
                time.sleep(8 * attempt)
                last_error = "HTTP 429"
                continue
            if resp.status_code != 200:
                last_error = f"HTTP {resp.status_code}"
                time.sleep(1.2 * attempt)
                continue
            match = FOLLOWERS_RE.search(resp.text)
            if not match:
                return None, "HTML'de Followers yok"
            return int(match.group(1).replace(",", "")), None
        except Exception as exc:
            last_error = str(exc)
            time.sleep(1.2 * attempt)
    return None, last_error


updated_count = 0
zero_count = 0
failed_count = 0
processed = 0
pending_ops = []


def flush_ops():
    if pending_ops:
        collection.bulk_write(pending_ops, ordered=False)
        pending_ops.clear()


with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = {}
    for game in games:
        raw_id = game.get("appid")
        try:
            app_id = int(raw_id)
        except Exception:
            failed_count += 1
            processed += 1
            continue
        futures[pool.submit(fetch_followers, app_id)] = (app_id, game.get("name"))

    for future in as_completed(futures):
        app_id, name = futures[future]
        processed += 1
        try:
            followers, error = future.result()
        except Exception as exc:
            followers, error = None, str(exc)

        if processed <= 5:
            print(f"[debug] appid={app_id} name={name!r} followers={followers} error={error}")

        if followers is None:
            failed_count += 1
        else:
            pending_ops.append(
                UpdateOne({"appid": app_id}, {"$set": {"followers": followers}})
            )
            if followers > 0:
                updated_count += 1
            else:
                zero_count += 1

        if len(pending_ops) >= 50:
            flush_ops()

        if processed % 50 == 0 or processed == total_games:
            print(
                f"[*] İlerleme: {processed}/{total_games} | "
                f"Güncellenen: {updated_count} | Sıfır: {zero_count} | "
                f"Atlanan: {failed_count}"
            )

flush_ops()
print(
    f"\n[✔] İşlem bitti. {updated_count} oyunda followers > 0, "
    f"{zero_count} oyunda 0, {failed_count} atlandı."
)
