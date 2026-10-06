#!/bin/bash
# Steam Satış Tahminleyici Streamlit Başlatma Betiği

echo "=========================================================="
echo "  STEAM SALES & REVENUE PREDICTOR (STREAMLIT DASHBOARD)   "
echo "=========================================================="

# 1. Sanal ortam veya yerel Python kontrolü
if [ -f ".venv/bin/streamlit" ]; then
    echo "[*] .venv ortamındaki Streamlit kullanılıyor..."
    .venv/bin/streamlit run app.py
elif command -v streamlit &> /dev/null; then
    echo "[*] Sistemdeki Streamlit kullanılıyor..."
    streamlit run app.py
else
    echo "[*] python3 -m streamlit ile başlatılıyor..."
    python3 -m streamlit run app.py
fi
