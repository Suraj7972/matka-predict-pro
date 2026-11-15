# Matka Predict Pro v7 — Client Dashboard

## 🚀 Quick Start

### Option 1: Streamlit Cloud (Recommended — no code visible to client)

1. Create a **private** GitHub repository
2. Upload this entire `client_app/` folder to it
3. Go to [streamlit.io/cloud](https://streamlit.io/cloud) → New app
4. Connect your GitHub repo → select `client_app.py`
5. Click **Deploy** — you'll get a URL like `https://your-app.streamlit.app`
6. Share that URL with your client

**Client sees only the web interface. Code stays private in your GitHub repo.**

---

### Option 2: Run Locally

```bash
# 1. Install Python 3.10+ and pip
# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run client_app.py

# 4. Open browser at http://localhost:8501
```

---

## 📁 Folder Structure

```
client_app/
├── client_app.py          ← Main Streamlit app (DO NOT share this file)
├── requirements.txt       ← Python dependencies
├── README.md              ← This file
└── data/
    ├── bt_data.json               ← Historical data (all 13 markets)
    ├── market_assign.json         ← Current model strategy config
    └── newx_strategy_results.json ← NX model strategy config
```

---

## 🎯 Client Instructions (share this section with client)

### How to Use the Dashboard

1. **Select Date** — Use the date picker in the left sidebar to choose the prediction date
2. **Both Models** — The dashboard shows predictions from two independent models:
   - **Current Model**: Single optimized strategy per market
   - **NX Model**: Separate strategies for Open and Close — higher accuracy
3. **Top 5 Markets**: The 5 most reliable markets for today (based on 30-day hit rate)
4. **Bottom 3 Markets**: Markets with lower reliability — use with caution
5. **Top-4 Digits**: For each market, the 4 most likely result digits (Open and Close)

### Updating Data

1. Prepare your Excel file with columns: `Date`, `Market`, `Open`, `Close`
   - `Date`: Format `2026-11-15` or `15/11/2026`
   - `Market`: Full market name (e.g. `Sridevi`, `KalyanNight`)
   - `Open`: Open panna (3-digit, e.g. `168`) or just the digit (0-9)
   - `Close`: Close panna (3-digit) or just the digit (0-9)
2. Click **Upload Excel (.xlsx)** in the sidebar
3. Predictions automatically refresh with the new data

---

## 📊 Understanding Results

| Metric | Meaning |
|--------|---------|
| **Hit Rate %** | % of last 30 sessions where actual digit was in Top-4 (baseline: 40%) |
| **Top-4 Digits** | 4 most likely digits for open/close result |
| **Confidence** | How concentrated the probability is (higher = model more certain) |

---

## ⚙️ Technical Notes

- Uses **35 prediction strategies** evaluated over 365 days of data
- **NX model** uses independent strategies for Open and Close sides
- Data updates are in-memory — refresh the page to reload base data
- For permanent data updates, replace `data/bt_data.json`
