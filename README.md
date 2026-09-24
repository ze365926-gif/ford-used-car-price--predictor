# 🚗 Ford Used Car Price Prediction Engine

An end-to-end Machine Learning regression project that predicts used Ford vehicle valuations in the UK market. Features exploratory data analysis, a zero-leakage scikit-learn pipeline, ensemble model training, and an interactive multi-page Streamlit web application.

---

## 📌 Highlights

- **Real Data**: 17,966 UK Ford listings across 23 vehicle models
- **Data Sanitization**: Fixed 6 real bugs from the raw notebook (year 2060 outlier, float truncation, train-test leakage, etc.)
- **Zero-Leakage Pipeline**: `ColumnTransformer` fitted only on training data
- **High Accuracy**: R² **0.9255** · MAE **£853** · RMSE **£1,297**
- **3-Page Web App**: Predictor · Data Insights · About

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| R² Score | **0.9255** |
| Mean Absolute Error | **£853.67** |
| Root Mean Squared Error | **£1,296.93** |

---

## 🗂️ Project Structure

```
ford-used-car-price-predictor/
├── data/
│   └── ford.csv                  ← Dataset (add manually)
├── artifacts/
│   ├── car_price_pipeline.joblib ← Trained pipeline (generated)
│   └── metrics.json              ← Model metrics (generated)
├── src/
│   ├── __init__.py
│   ├── data_prep.py              ← Data cleaning & splitting
│   ├── train.py                  ← Training & evaluation
│   └── predict.py                ← Inference (single & batch)
├── app.py                        ← Streamlit web app
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quickstart

### 1. Clone & Install
```bash
git clone https://github.com/<your-username>/ford-used-car-price-predictor.git
cd ford-used-car-price-predictor
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add Dataset
Place `ford.csv` inside the `data/` folder.

### 3. Train the Model
```bash
python src/train.py
```

### 4. Launch the Web App
```bash
streamlit run app.py
```
Open **http://localhost:8501**

### 5. CLI Prediction (optional)
```bash
python src/predict.py
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.10+ |
| ML | scikit-learn, NumPy, Pandas |
| Serialization | Joblib |
| Web App | Streamlit |
| Visualization | Matplotlib, Seaborn |

---

## 🐞 Bugs Fixed from Original Notebook

| Bug | Fix |
|-----|-----|
| `year == 2060` outlier | Filtered to `year ≤ 2024` |
| `engineSize == 0.0` | Filtered to `engineSize > 0` |
| `.astype(int)` truncating floats (mpg, engineSize) | Removed entirely |
| `pd.get_dummies` before train/test split (data leakage) | Replaced with `OneHotEncoder` inside `Pipeline` |
| Leading spaces in model names (` Fiesta`) | `.str.strip()` applied |
| 154 duplicate rows | `drop_duplicates()` applied |

---

## 📄 License

MIT
