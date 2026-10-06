# Steam Pre-Launch Sales Predictor

Predict first-month sales, active players, and net developer revenue for Steam pre-launch games using econometric machine learning.

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-1E1218?style=flat-square)](https://www.python.org/downloads/)
[![License MIT](https://img.shields.io/badge/License-MIT-1E1218?style=flat-square)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-810541?style=flat-square)](https://streamlit.io)
[![MongoDB](https://img.shields.io/badge/MongoDB-Included-1E1218?style=flat-square)](https://www.mongodb.com)

## The Problem

Steam keeps sales data private. Developers, investors, and studios can't benchmark launch performance, forecast revenue, or price strategically.

This system solves it by combining three public data sources (Steam API, community XML, HTML scraping) with hybrid econometric + machine learning models.

## How It Works

**Real-time predictions** on first-month sales, revenue, and confidence ranges (conservative, expected, optimistic).

**Hybrid engine** blends econometric curves with a gradient boosting regressor trained on 320+ games. Dynamic feature engineering (40+ signals including followers, velocity, regional markets) adapts predictions across all scales—from micro indie games to AAA launches.

**Three data layers**, no premium APIs or rate limiting. Built with public Steam endpoints and fallback architectures for resilience.

## Features

- **First-month sales forecast** with revenue projections (net developer earnings after Steam's 30% commission)
- **Confidence intervals** for conservative, expected, and optimistic scenarios
- **Time-based forecasts** for week 1, month 1, and year 1
- **40+ engineered features** (log transforms, genre/tag weights, regional indicators, velocity ratios)
- **Dynamic pricing engine** for TBA games and price exploration
- **Real-time dashboard** with Plotly visualizations and Streamlit interactivity

## Getting Started

### Requirements
- Python 3.9+
- MongoDB (local or cloud)
- pip

### Installation

```bash
git clone https://github.com/ferayyaren/steam-sales-predictor.git
cd steam-sales-predictor

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

```bash
# Start MongoDB locally
mongod

# Or use cloud MongoDB (update MONGO_URI in app.py):
# MONGO_URI = "mongodb+srv://user:pass@cluster.mongodb.net/steam_prediction_project"
```

### Run

```bash
streamlit run app.py
# Open http://localhost:8501
```

Or with Docker:
```bash
docker-compose up --build
```

### Usage

Enter a Steam game name or AppID and get instant predictions:
- First-month sales forecast
- Net revenue estimate
- Confidence ranges (P25–P75)
- Time-based projections
- Detailed analytics

## Model Architecture

The prediction engine operates in four phases:

1. **Live data ingestion** — Steam Store API (price, release, genres), Community XML (followers), HTML scraping (tags, regional indicators)
2. **Feature engineering** — 40+ numerical and categorical signals; log transforms and dynamic ratios tuned per genre and scale
3. **Dual-engine prediction** — Gradient boosting (320+ games, R² = 89.41%) blended with econometric saturation curves (fallback for sparse data)
4. **Revenue simulation** — Unit forecasts converted to net developer earnings with P25/P50/P75 confidence bounds

### Data Sources

Three public layers, no authentication required:

1. **Steam Store API** (official JSON) — Price, release date, genres, supported languages
2. **Store Page HTML** (scraping) — Community tags, regional market indicators
3. **Community Group XML** (member count) — Rate-limit safe follower counts

### Model Performance

| Metric | Conversion Model | Revenue Model |
|--------|------------------|---------------|
| **R² Score** | 89.41% | 96.04% |
| **MAE** | ±0.0118x | ±$120,000 |
| **Training Data** | 320+ games | 320+ games |

## Advanced Features

### Feature Engineering (40+ Signals)

**Log-transformed demand metrics** — Log compression prevents tree overfitting across scales (followers range 2 → 1M+)
- `log_followers`, `log_wishlists`, `log_velocity_30d`, `log_price`

**Ratio & interaction features** — Normalized growth rate and price elasticity
- `velocity_ratio` = velocity_30d / followers
- `price_elasticity` = log(followers) / log(price + 1)
- Dynamic wishlist/follower ratio (10.6x–14.9x range per genre)

**Genre & tag encoding** — 20+ one-hot binary features
- Roguelike, Souls-like, Immersive Sim, Boomer Shooter, Horror, Survival, etc.

### Econometric Baseline

For sparse data (<15K wishlists), the system blends ML predictions with log-normal saturation curves validated against Gamalytic and VG Insights methodologies.

**Hybrid strategy:**
- <1K wishlists: 100% econometric (conservative)
- 1K–15K wishlists: Linear interpolation
- >15K wishlists: 100% ML (high confidence)

### Gradient Boosting Regressor

Why gradient boosting for this task?
- Captures non-linear relationships (wishlist→sales is not linear)
- Robust to outliers (AAA games with 1M+ sales don't break training)
- Feature importance interpretability
- Scikit-Learn implementation: 130 trees, max_depth=3, learning_rate=0.04

## Technology Stack

**Backend** — Python 3.9+, Scikit-Learn, Pandas, NumPy, MongoDB
**Frontend** — Streamlit, Plotly, custom CSS (Deep Berry theme)
**Deployment** — Docker, Docker Compose

## Project Structure

```
steam-sales-predictor/
├── app.py                       # Streamlit UI (main entry)
├── train_clean.py               # ML model training
├── predict_cli.py               # CLI predictions
├── build_clean_dataset.py       # Data ETL
├── steam_data_enricher.py       # Follower scraper
├── requirements.txt             # Dependencies
├── Dockerfile                   # Container image
├── docker-compose.yml           # Multi-service config
├── README.md                    # This file
├── LICENSE                      # MIT
└── TECHNICAL_REPORT.md          # Full methodology
```

## Examples

### CLI prediction

```bash
python3 predict_cli.py
# Enter: Valheim
# Output: ~45K copies, $945K net revenue
```

### API usage

```python
from predict_cli import run_forecast

forecast = run_forecast(
    name="My Game",
    followers=5000,
    velocity_7d=150,
    price_usd=19.99,
    tags=["Action", "Roguelike"],
    genres=["Action", "Adventure"]
)

print(f"Expected: {forecast['month1_expected']:,} copies")
print(f"Revenue: ${forecast['month1_net_rev']:,.0f}")
```

## Deployment

**Local:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

**Docker:**
```bash
docker-compose up --build
# Streamlit: http://localhost:8501
# MongoDB: localhost:27017
```

**Cloud:**
Push to GitHub and connect to Streamlit Cloud for automatic deploys.

## Methodology & Validation

Predictions are validated against Gamalytic public reports, VG Insights econometric research, and GameDiscoverCo market analysis.

**Benchmark accuracy:**
- Manaphore (2 followers): Our 9 copies vs. Gamalytic 7 (+28%, in range)
- Cosmo Arena (5 followers): Our 22 vs. Gamalytic 22 (exact match)
- RetroSpace (10.8K followers): Our 70K vs. Gamalytic 71.1K (−1.2%)

## Contributing

Contributions welcome. Areas of interest:
- Additional data sources (SteamDB, IsThereAnyDeal)
- Advanced monetization models (DLC, seasonal passes)
- Geographic market segmentation (China, Japan, Korea)
- Real-time sentiment analysis

## Roadmap

- Bayesian confidence intervals
- Wishlist-to-conversion curves by genre
- Review sentiment analysis
- Regional market breakdown
- DLC & seasonal pass modeling
- Mobile game adaptation
- REST/GraphQL API
- Discord bot integration

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

Validated against and inspired by:
- Gamalytic (Simon Carless, Chris Zukowski)
- Video Game Insights (Boxleiter)
- GameDiscoverCo methodology
- Jake Birkett (Grey Alien Games)

---

For questions or issues, open an issue or discussion on GitHub.
