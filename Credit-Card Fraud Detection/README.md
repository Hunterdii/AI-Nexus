# 💳 CreditGuard AI — Credit Card Fraud Detection
### End-to-End ML Pipeline · Production Grade
**Author:** Patel Hetkumar Sandipbhai [2505102310011]
**Institution:** Parul University — Data Mining & Machine Learning | 2025–2026

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
# Opens → http://localhost:8501
```

## 📦 Package Contents

| File / Folder | Description |
|---|---|
| `app.py` | 6-page production Streamlit dashboard |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Dark-theme configuration |
| `Credit_Card_Fraud_Detection_FINAL_REPORT.docx` | Full 10-chapter academic report |
| `Credit_Card_Fraud_Detection_FINAL.ipynb` | Production-grade Jupyter notebook |
| `models/` | 10 trained .pkl model files |
| `images/` | 12 high-resolution visualizations (150 DPI) |

## 📊 Model Results

| Model | PR-AUC | ROC-AUC | F1(θ=0.50) | F1(θ=0.98) | Time |
|---|---|---|---|---|---|
| 🏆 **XGBoost** | **0.8477** | 0.9784 | 0.3446 | **0.7444** | 31s |
| Random Forest | 0.8150 | 0.9861 | 0.6437 | — | 1,373s |
| Logistic Regression | 0.7221 | 0.9714 | 0.1112 | — | 1,866s |
| Gradient Boosting | 0.6935 | 0.9782 | 0.4084 | — | 5,014s |
| Decision Tree | 0.4504 | 0.8779 | 0.1075 | — | 101s |

## 🎯 Dashboard Pages

1. **🏠 Home** — 3D animated credit card + KPI cards + pipeline + radar chart
2. **🔍 Prediction** — Live fraud scoring, probability gauge, threshold guide
3. **📊 Comparison** — Interactive ROC/PR + confusion matrices + threshold sweep
4. **📈 EDA** — Interactive charts, correlation analysis, key findings
5. **🖼️ Gallery** — All 12 visualizations with search/filter
6. **ℹ️ About** — Reference, tech stack, file status checker

## 📁 Models

- `pipeline_bundle.pkl` — Complete production bundle (model + scalers + threshold + metrics)
- `XGBoost.pkl` — Best standalone classifier (PR-AUC=0.8477, optimal θ=0.98 → F1=74.44%)
