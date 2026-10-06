# Steam Pre-Launch Sales Predictor 🎮

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live-FF4B4B.svg)](https://streamlit.io)
[![MongoDB](https://img.shields.io/badge/Database-MongoDB-green.svg)](https://www.mongodb.com)

**An econometric machine learning system that predicts first-month sales, active players, and net developer revenue for Steam pre-launch games in real-time.**

---

## 🎯 Problem Statement

Valve keeps all Steam game sales data **private**, leaving developers, investors, and publishers with no way to:
- Benchmark against peer games
- Forecast launch performance
- Make data-driven pricing decisions
- Assess market positioning

**This project solves that problem** by combining three public data layers (Steam API, community XML, HTML scraping) with a hybrid econometric + ML prediction engine.

---

## ✨ Key Features

### 🔮 Real-Time Predictions
- **First-Month Sales Forecast**: Predicted copies sold (paid games) or active players (F2P)
- **Revenue Projection**: Net developer earnings after Steam's 30% commission
- **Confidence Intervals**: Conservative (P25), Expected (P50), Optimistic (P75) scenarios
- **Time-Based Forecasts**: 7-day, 1-month, and lifetime projections

### 🧠 Hybrid ML Architecture
- **Gradient Boosting Regressor**: 130 trees, trained on 320+ games (R² = 89.41%)
- **Econometric Baseline**: Log-normal saturation curve validated against Gamalytic/VG Insights
- **Dynamic Blending**: Seamless transition from econometric (micro-scale) to ML (mid-to-large scale)

### 📊 Advanced Features
- **40+ Engineered Features**: Log transforms, ratios, genre/tag one-hot encoding
- **Dynamic Pricing Engine**: Automatic price recommendations for TBA (to-be-announced) games
- **Regional Market Analysis**: Chinese language support detection, multilingual monetization modeling
- **Real-Time Data**: No rate limiting—uses 3-layer fallback architecture for rate-limit safety

### 🎨 Modern UI
- **Streamlit Dashboard**: Real-time interactive predictions with Plotly visualizations
- **Distinctive Design**: Deep Berry theme (#810541) with professional metrics cards
- **Responsive**: Works smoothly on desktop, tablet, and mobile

---

## 📸 Screenshots

### Dashboard Overview
![Steam Predictor - Main Dashboard](assets/steam1.png)

### Detailed Predictions
![Steam Predictor - Predictions](assets/steam2.png)

### Analytics & Insights
![Steam Predictor - Analytics](assets/steam3.png)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────┐
│         USER INPUT (Game Name/AppID)        │
└────────────────────┬────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
┌───────▼──┐   ┌─────▼──┐   ┌────▼──────┐
│Steam API │   │ Store  │   │Community  │
│(JSON)    │   │Page    │   │XML        │
│          │   │(HTML)  │   │(Members)  │
└───────┬──┘   └─────┬──┘   └────┬──────┘
        │            │            │
        └────────────┼────────────┘
                     │
            ┌────────▼────────┐
            │Feature          │
            │Engineering      │
            │(40+ features)   │
            └────────┬────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
  ┌─────▼──────┐          ┌──────▼────┐
  │Econometric │          │ML Model    │
  │Curve       │          │(Gradient   │
  │(Baseline)  │          │Boosting)   │
  └─────┬──────┘          └──────┬────┘
        │                        │
        └────────────┬───────────┘
                     │
            ┌────────▼────────┐
            │Revenue          │
            │Calculator       │
            │(Paid/F2P/TBA)   │
            └────────┬────────┘
                     │
            ┌────────▼────────┐
            │Final Report     │
            │(UI + JSON API)  │
            └─────────────────┘
```

**Three-Layer Data Architecture (No Premium APIs Required):**
1. **Steam Store API** (official JSON) — Price, release date, genres, supported languages, categories
2. **Store Page HTML** (scraping) — Community tags, clan ID, regional market indicators
3. **Community Group XML** (member count) — Rate-limit safe organic follower counts

---

## ⚙️ How It Works

The prediction pipeline operates in four coordinated phases:

```
[1. User Input] ──► [2. Live Data Ingestion] ──► [3. ML + Econometric Engine] ──► [4. Interactive UI Report]
  - Game Name          - Store API (Price, Tags)    - GradientBoosting (320+ games)    - 4 Key Metrics (KPIs)
  - Steam App ID       - Community XML (Followers)  - Log-normal Saturation Multiplier - Volume & Revenue Charts
                       - HTML Scraping (Langs)      - P25 / P50 / P75 Scenarios        - Econometric Breakdown
```

### 1. 🔍 Live Data Ingestion
* When you enter a game name or App ID (e.g. `Valheim` or `3393110`), the system performs an instant multi-layer crawl across Steam's public endpoints.
* **Store API:** Retrieves developer, publisher, release date, genres, and pricing status.
* **Community XML:** Safely fetches official community follower count without API keys or aggressive rate limits.
* **Storefront HTML:** Parses user-defined community tags and detects high-value localized market support (e.g., Chinese language presence).

### 2. 🧠 Feature Engineering & ML Inference
* Prepares 40+ numerical and categorical features: follower velocity, log-transformed metrics, dynamic tag weights, and price scaling.
* Trains/infers via a regularized **`GradientBoostingRegressor`** trained on 320+ shipped games.

### 3. 💵 Econometric Revenue Simulation
* **First-Month Sales:** Forecasts expected copies sold for paid games or player conversion for Free-to-Play titles.
* **Net Developer Revenue:** Applies Steam's standard 30% store fee.
* **P25 / P50 / P75 Confidence Bounds:** Generates realistic Conservative, Expected, and Optimistic revenue envelopes.

### 4. 📊 High-Craft Visual Intelligence
* Displays four mission-control KPI cards with tabular mono numbers.
* Generates interactive Plotly projections comparing T+7 (first week), 1st month, and lifetime copies.
* Provides full transparency under the Metrics tab, showing exact formulas and model distributions.

---

## 🧠 Model Architecture & Performance

### Dual-Output Regression System

This system employs a **hybrid econometric-ML architecture** inspired by Gamalytic and VG Insights methodologies:

**Model 1: Conversion Multiplier Prediction (R² = 89.41%)**
- Predicts what % of wishlists convert to first-month sales
- Range: 2-45% (industry standard saturation band per Gamalytic research)
- Trained on 320+ shipped games with known sales data
- MAE: ±0.0118x (conversion rate tolerance)

**Model 2: Revenue Prediction (R² = 96.04%)**
- Forecasts net developer earnings (after Steam's 30% commission)
- Accounts for pricing, F2P monetization, and regional markets
- Dual-target approach ensures both unit prediction and financial accuracy

### Industry Alignment

**Gamalytic Methodology:**
- Wishlist/Follower ratio: 11.02x baseline (varies 10.6x - 14.9x by genre/age)
- Conversion curve: Log-normal saturation (41.5% @ 500WL → 20% @ 25KWL → 12% @ 1MWL)
- Confidence bands: P25 (50% of expected), P75 (200% of expected)

**VG Insights Framework:**
- Revenue = Units × Price × Regional ARPU multiplier
- Time decay: T+7 = 55% of month-1, Year-1 = 260% of month-1
- Price elasticity: Larger audiences tolerate higher prices (inverse-nonlinear)

### Feature Engineering (40+ Features)

**Log-Transformed (Demand Signals):**
- `log_followers`, `log_wishlists`, `log_velocity_30d`, `log_price`
- Rationale: Followers range 2→1M. Log compression prevents tree overfitting to large-scale games

**Ratio & Interaction Features:**
- `velocity_ratio` = velocity_30d / followers (normalized growth rate)
- `price_elasticity` = log(followers) / log(price + 1) (captures inverse elasticity)
- `wl_ratio` = dynamic_wishlist_ratio(followers, tags, appid) [10.6x - 14.9x range]

**Genre & Tag One-Hot Encoding (20+ binary features):**
- Roguelike, Souls-like, Immersive Sim, Boomer Shooter, Horror, Survival, RPG, Strategy, Casual, etc.
- Rationale: Different genres have distinct sales curves (Roguelike communities wishlist differently than Story-Rich)

**Dynamic Wishlist Ratio Calculation:**
```
Base: 11.02x (Gamalytic benchmark)
+ Cult bonuses: +0.25 to +0.40 for Immersive Sim, Souls-like, Boomer Shooter
+ Age bonuses: +0.05 to +0.38 depending on AppID (older pages accumulated wishlists longer)
+ Scale bonuses: +0.15 for 5K-35K followers (viral sweet spot)
Result: Final ratio ∈ [10.6x, 14.9x]
```

### Econometric Baseline (Sparse Data Fallback)

For games below 15K wishlists (sparse training region), the system blends econometric curves with ML predictions:

**Log-Normal Saturation Curve:**
```
Conversion(WL) = {
  0.415                              if WL ≤ 500      (core audience, high conviction)
  0.415 - 0.215 × prog               if 500 < WL ≤ 25K   (interpolation)
  0.20 + 0.108 × prog                if 25K < WL ≤ 250K  (inflection point)
  0.308 × (250K / WL)^0.18           if WL > 250K     (mass market, lower conversion)
}
```

**Hybrid Blending Strategy:**
- <1K wishlists: 100% econometric (theory-driven, conservative)
- 1K-15K wishlists: Linear interpolation (econometric + ML)
- >15K wishlists: 100% ML (data-driven, high confidence region)

### Gradient Boosting Regressor (Production Algorithm)

**Why Gradient Boosting?**
- Captures non-linear relationships (wishlist/sales is not linear)
- Robust to outliers (AAA games with 1M+ sales don't break training)
- Feature importance interpretability (can rank which features drive predictions)
- Scikit-Learn `GradientBoostingRegressor`: 130 trees, max_depth=3, learning_rate=0.04

### Validation Results

| Metric | Multiplier Model | Revenue Model | Notes |
|--------|------------------|---------------|-------|
| **R² Score** | **89.41%** | **96.04%** | Explains 89-96% of output variance |
| **MAE** | ±0.0118x | ±$120,000 | Average absolute error |
| **Training Data** | 320+ games | 320+ games | MongoDB games_clean collection |
| **Features** | 40+ engineered | 40+ engineered | Logs, ratios, one-hots, dynamics |

### Benchmark Comparisons (Industry Validation)

| Game | Followers | Scale | Our Forecast | Gamalytic Report | Variance | Status |
|------|-----------|-------|--------------|------------------|----------|--------|
| **Manaphore** | 2 | Micro | ~9 copies | 7 copies | +28% | ✓ In range |
| **Cosmo Arena** | 5 | Micro | ~22 players | 22 players | 0% | ✓ Exact match |
| **Tidy Backpack** | 41 | Micro | ~153 copies | 154 copies | -0.6% | ✓ Near-exact |
| **RetroSpace** | 10.8K | Indie | ~70K copies | 71.1K copies | -1.2% | ✓ Excellent |
| **AION 2** | 104K | AAA | ~279K players | 291.6K players | -4.3% | ✓ Good |

**Sources:** 
- Gamalytic public reports (Simon Carless, GameDiscoverCo)
- VG Insights econometric benchmarks (Boxleiter)
- GameDiscoverCo market analysis methodology

---

## � Installation

### Prerequisites
- **Python 3.9+**
- **MongoDB** (local or cloud)
- **pip** (Python package manager)

### Local Setup

#### 1. Clone the repository
```bash
git clone https://github.com/ferayyarenturasay/steam-sales-predictor.git
cd steam-sales-predictor
```

#### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 3. Install dependencies
```bash
pip install -r requirements.txt
```

#### 4. Configure MongoDB
```bash
# Option A: Local MongoDB
mongod  # Start MongoDB server

# Option B: Cloud MongoDB (Atlas)
# Update connection string in app.py:
# MONGO_URI = "mongodb+srv://username:password@cluster.mongodb.net/steam_prediction_project"
```

#### 5. Run the Streamlit app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

### Docker Deployment

#### Build and run with Docker Compose
```bash
docker-compose up --build
```

This starts:
- **Streamlit app** on port 8501
- **MongoDB** on port 27017 (if using docker-compose MongoDB)

#### Or build custom Docker image
```bash
docker build -t steam-predictor .
docker run -p 8501:8501 steam-predictor
```

---

## 🚀 Quick Start

### 1. Web UI (Streamlit)

**Run the app:**
```bash
streamlit run app.py
```

**Enter a Steam game:**
- Search by game name (e.g., "Valheim", "Elden Ring")
- Or enter AppID (e.g., 3393110)

**Get instant predictions:**
- First-month sales forecast
- Net revenue (developer earnings)
- Confidence ranges
- Time-based projections
- Detailed analytics tabs

### 2. CLI Interface

**For batch predictions without UI:**
```bash
python3 predict_cli.py

# Interactive mode:
# Enter game name or AppID when prompted
# Get console-formatted report
```

---

## 📚 API Documentation

### Programmatic Usage

```python
from predict_cli import run_forecast

# Predict for a game
forecast = run_forecast(
    name="My Game",
    followers=5000,
    velocity_7d=150,
    price_usd=19.99,
    release_date="2024-11-15",
    tags=["Action", "Roguelike"],
    genres=["Action", "Adventure"],
    lang_count=8,
    has_chinese=True,
    is_most_followed=False
)

# Access predictions
print(f"Expected sales: {forecast['month1_expected']:,}")
print(f"Revenue range: ${forecast['month1_rev_low']:,.0f} - ${forecast['month1_rev_high']:,.0f}")
print(f"Confidence: P25={forecast['month1_low']}, P50={forecast['month1_expected']}, P75={forecast['month1_high']}")
```

### Model Outputs

**Response object contains:**
```python
{
    'followers': int,                    # Current live followers
    'wishlists': int,                    # Estimated wishlist count
    'wl_ratio': float,                   # Dynamic wishlist/follower ratio
    'month1_expected': int,              # Expected first-month sales
    'month1_low': int,                   # P25 (conservative)
    'month1_high': int,                  # P75 (optimistic)
    'month1_net_rev': float,             # Net developer revenue (USD)
    'month1_rev_low': float,             # Conservative revenue
    'month1_rev_high': float,            # Optimistic revenue
    't7_copies': int,                    # First-week sales
    'lifetime_copies': int,              # Projected year-1 sales
    'mult_p50': float,                   # Conversion multiplier (%)
    'used_engine': str,                  # "ML" or "Econometric"
}
```

---

## �🛠️ Technology Stack

### Backend
- **Python 3.9+** — Core application language
- **Scikit-Learn** — Gradient Boosting Regressor (ML model)
- **Pandas** — Data manipulation and feature engineering
- **NumPy** — Numerical computations and log transforms
- **MongoDB** — Training data persistence

### Frontend
- **Streamlit** — Real-time interactive dashboard
- **Plotly** — Data visualizations (charts, confidence ranges)
- **Custom CSS** — Modern design system (Deep Berry theme)

### Deployment
- **Docker** — Containerization
- **Docker Compose** — Multi-service orchestration

### Data Sources
- **Steam Store API** — Official (no auth required)
- **Steam Community XML** — Official member feeds
- **SteamSpy** — Third-party follower fallback

---

## 📂 Project Structure

```
steam-sales-predictor/
├── app.py                          # Streamlit UI (main entry point)
├── train_clean.py                  # ML model training pipeline
├── predict_cli.py                  # CLI prediction interface
├── predict_game.py                 # Test harness & validation
├── build_clean_dataset.py          # Data ETL (SteamSpy + Steam API)
├── steam_data_enricher.py          # Follower data scraper
├── steam_regional_enrichment.py    # Regional pricing (future)
├── snapshot_collector.py           # Historical backtest archive
├── steamspy_pipeline.py            # Batch SteamSpy ingestion
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container image
├── docker-compose.yml              # Multi-service config
├── .streamlit/
│   ├── config.toml                 # Streamlit settings
│   └── credentials.toml            # (Optional auth)
├── TECHNICAL_REPORT.md             # Full technical documentation
├── README.md                       # This file
├── LICENSE                         # MIT License
└── baseline_predictions.csv        # Historical validation set
```

---

## 💡 Usage Examples

### Example 1: Predicting a Micro-Game

```bash
python3 predict_cli.py
# Enter: Manaphore
# Output: ~9 copies, $31 net revenue (P25-P75: 4-18 copies)
```

### Example 2: Web UI for AAA Game

1. Run `streamlit run app.py`
2. Search "Baldur's Gate 3"
3. See instant forecast:
   - 100K+ followers
   - ~25M wishlists
   - **~5.5M first-month sales**
   - **~$110M net revenue** (range: $55M-$220M)

---

## 🔬 Methodology

### Data Acquisition (3 Layers)

1. **Steam Store API** (official, no auth)
   - Price, release date, genres, languages, categories

2. **Store Page HTML Scraping**
   - Community tags (top 15-20 user-voted tags)
   - Clan ID (for group member XML lookup)

3. **Community Group XML**
   - Member count (follower count)
   - Never rate-limited (Valve-endorsed)
   - Fallback: app hub or SteamSpy

### Model Training

```bash
python3 train_clean.py
# Reads 320+ games from MongoDB
# Extracts 40+ features
# Trains Gradient Boosting (80/20 split)
# Saves models to disk for inference
```

---

## 🚦 Deployment

### Option 1: Local Streamlit

```bash
pip install -r requirements.txt
streamlit run app.py
# Open http://localhost:8501
```

### Option 2: Docker Compose (Recommended)

```bash
docker-compose up --build
# Streamlit: http://localhost:8501
# MongoDB: localhost:27017 (if using docker MongoDB)
```

### Option 3: Cloud Deployment (Streamlit Cloud)

```bash
# Push to GitHub, connect to Streamlit Cloud
# Automatic CI/CD deploys on every push
```

---

## 📖 Full Documentation

For detailed technical information, see:
- **[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)** — Complete methodology (1,841 lines)

---

## 📋 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m 'Add your feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request

### Areas for Contribution
- Additional data sources (SteamDB, IsThereAnyDeal)
- Advanced monetization models (DLC, seasonal passes)
- Geographic market segmentation
- Real-time sentiment analysis from reviews
- Model ensemble improvements

---

## 🐛 Known Limitations

1. **Micro-games (<500 followers)**: Limited training data, higher uncertainty
2. **TBA games**: Price recommendations are template-based, not predictive
3. **Regional pricing**: Currently USD-only (EUR/GBP conversion planned)
4. **F2P ARPU**: Simplified models, doesn't account for whale spending patterns
5. **Post-launch changes**: Predictions assume no major updates/DLC

---

## 🙏 Acknowledgments

This project was inspired by and validated against:
- **Gamalytic** (Simon Carless, Chris Zukowski)
- **Video Game Insights (VGI)** (Boxleiter's econometric research)
- **GameDiscoverCo** (market analysis methodology)
- **Jake Birkett** (Grey Alien Games, conversion research)

---

## 📈 Roadmap

- [ ] Advanced confidence intervals (Bayesian posterior)
- [ ] Wishlist-to-conversion curves by genre
- [ ] Real-time sentiment analysis (review text)
- [ ] Regional market breakdown (China, Japan, Korea)
- [ ] DLC & seasonal pass modeling
- [ ] Mobile game adaptation (iOS/Android AppStore)
- [ ] API endpoint (REST/GraphQL)
- [ ] Discord bot integration

---

**Happy forecasting! 🎮📊**

For questions or suggestions, open an issue or start a discussion.
