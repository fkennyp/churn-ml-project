# 📉 Customer Churn Prediction — End-to-End ML Project

End-to-end machine learning portfolio project: from raw Kaggle data → model training → prediction API (FastAPI) → web app (Streamlit).

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-frontend-FF4B4B.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-classifier-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-pipeline-F7931E.svg)

## 🎯 Problem

Customer churn (customers canceling their subscriptions) is a silent but massive cost for subscription and telecommunications businesses.
**Goal:** Predict which customers are at risk of churning so the business team can take early retention actions.

## ⚡ Demo

**Streamlit Web UI** (`:8501`) — input form → real-time predictions from the API:

![Streamlit UI](docs/streamlit_demo.png)

**FastAPI Interactive Docs** (`:8000/docs`) — auto-generated Swagger UI:

![Swagger API](docs/swagger_demo.png)

## 🏗️ Architecture

```
┌──────────────┐    ┌───────────────────┐    ┌────────────────────┐
│ data/raw     │───>│ src/data_prep.py  │───>│ data/processed     │
└──────────────┘    └───────────────────┘    └─────────┬──────────┘
                                                       ▼
                                           ┌────────────────────┐
                                           │ src/train_model.py │
                                           │ OneHot + XGBoost   │
                                           └─────────┬──────────┘
                                                     ▼
                             ┌──────────────────────────────────────┐
                             │ api/model.pkl + api/model_meta.json  │
                             └──────────────────┬───────────────────┘
                                                ▼
┌────────────────┐   HTTP POST    ┌─────────────────────────────┐
│ Streamlit UI   │───────────────>│ FastAPI (app.py) :8000      │
│ :8501          │<───────────────│ load model.pkl, predict     │
└────────────────┘   JSON result  └─────────────────────────────┘
```

Frontend and backend are **fully separated** — the UI does not load the model, it only calls the API via HTTP (an industry-standard architectural pattern).

## 📁 Project Structure

```
churn-ml-project/
├── api/
│   ├── model.pkl            # trained model (local, gitignored — regenerate via training)
│   └── model_meta.json      # model card: features, class counts, scale_pos_weight
├── data/
│   ├── raw/                 # raw dataset from Kaggle
│   └── processed/           # cleaned data (gitignored, can be regenerated)
├── docs/                    # screenshots for README
├── src/
│   ├── data_prep.py         # data cleaning & preprocessing pipeline
│   └── train_model.py       # XGBoost training + save model & metadata
├── ui/
│   └── app.py               # Streamlit frontend
├── app.py                   # FastAPI backend
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

```bash
# 1. Clone repo
git clone https://github.com/fkennyp/churn-ml-project.git
cd churn-ml-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare data & train model (generates model.pkl)
python src/data_prep.py
python src/train_model.py

# 4. Run API backend (terminal 1)
python -m uvicorn app:app --reload --port 8000

# 5. Run frontend (terminal 2)
python -m streamlit run ui/app.py
```

Open **http://localhost:8501** → fill out the form → get churn predictions along with probabilities.

## 🔌 API Endpoints

| Method | Endpoint  | Function                                  |
|--------|-----------|-------------------------------------------|
| GET    | `/health` | Health check (model loaded?)              |
| POST   | `/predict`| Predict churn based on customer data      |

Example `POST /predict` request:

```json
{
  "tenure": 2,
  "MonthlyCharges": 85.5,
  "Contract": "Month-to-month",
  "InternetService": "Fiber optic",
  "PaymentMethod": "Electronic check"
}
```

Example response:

```json
{
  "prediction": 1,
  "churn_probability": 0.8738,
  "message": "⚠️ Customer predicted to CHURN with 87.38% probability"
}
```

Interactive docs (Swagger): `http://localhost:8000/docs`

## 📊 Data & Model

- **Dataset:** Customer Churn (Kaggle), ~7,043 customers
- **Features:**
  - Numeric: `tenure`, `MonthlyCharges`
  - Categorical: `Contract`, `InternetService`, `PaymentMethod` (OneHotEncoder)
- **Class imbalance:** 1,495 churn vs 4,139 non-churn in training data → handled with `scale_pos_weight ≈ 2.77` in XGBoost
- **Model:** `XGBClassifier` wrapped in a scikit-learn `Pipeline` (preprocessing + classifier in a single object)
- **Model metadata:** see [`api/model_meta.json`](api/model_meta.json)

**Performance on test data:**

| Metric         | Value |
|----------------|-------|
| Accuracy       | 0.75  |
| ROC-AUC        | 0.84  |
| Recall (churn) | 0.79  |

> **Trade-off note:** Due to imbalanced data (1,495 churn vs 4,139 stay), the model is prioritized for **churn recall** via `scale_pos_weight ≈ 2.77`. A churn precision of 0.52 is an intentional trade-off: for a business, the cost of sending a retention offer to a non-churning customer is far cheaper than losing a customer who actually churns.

## 🧠 Lessons Learned

- Handling **class imbalance** with `scale_pos_weight`, rather than relying on raw accuracy
- Wrapping preprocessing + model in a single **Pipeline** to prevent data leakage during inference
- Separating **frontend & backend** via HTTP APIs, rather than a single monolithic script
- Using **Pydantic** for input validation at the API edge — malformed data is automatically rejected (HTTP 422)
- Saving **model metadata** to make experiments reproducible and documented

## 🗺️ Roadmap

- [ ] Containerization with Docker
- [ ] Deploy API to cloud (Render / Railway)
- [ ] Model drift monitoring
- [ ] CI/CD with GitHub Actions

## 📬 Contact

**Kenny Putrajaya** — [github.com/fkennyp](https://github.com/fkennyp)