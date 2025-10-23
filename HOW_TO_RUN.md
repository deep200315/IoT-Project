# How to Run the Project

## ✅ Quick Start (Recommended)

### Option 1: Automated Pipeline (Easiest)

```bash
# Install dependencies
pip install -r requirements.txt

# Run everything automatically
python3 run_pipeline.py
```

**This will:**
- Generate data
- Create visualizations
- Train both models
- Save all results

**Runtime:** 10-15 minutes  
**Output:** `results/plots/` (17+ images), `models/` (trained models)

---

## 📓 Option 2: Jupyter Notebook (Interactive)

### Important: Restart Kernel After Installation!

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Jupyter
jupyter notebook notebooks/iot_anomaly_detection.ipynb

# 3. In Jupyter: Kernel → Restart Kernel
# 4. Run all cells: Cell → Run All
```

**⚠️ IMPORTANT:** If you get errors like:
- `ValueError: Invalid filepath extension`
- `TypeError: missing required argument`

**Solution:** Restart the Jupyter kernel!
- Click: `Kernel` → `Restart Kernel`
- Then run all cells again

This ensures the notebook uses the updated code.

---

## 📂 Project Structure

```
project/
├── run_pipeline.py          # ← Run this for automated execution
├── notebooks/
│   └── iot_anomaly_detection.ipynb  # ← Or run this interactively
├── src/                     # Source code (all fixed)
├── data/                    # Generated data
├── models/                  # Trained models
└── results/plots/           # Visualizations
```

---

## 📖 Documentation

- **COMPLETE_GUIDE.md** - Comprehensive guide explaining everything
- **README.md** - Full technical documentation
- **docs/SUMMARY.md** - Findings and insights (3 pages)
- **docs/QUICKSTART.md** - 5-minute setup guide

---

## ✅ All Issues Fixed

✓ Data generation works  
✓ save_dataset(df) - requires dataframe argument  
✓ LSTM save_model() - uses .keras extension  
✓ All modules tested and working  
✓ No syntax errors  

---

## 🎯 Expected Results

After running, you'll get:

**Data:**
- `data/sensor_data.csv` (10,000 samples)

**Models:**
- `models/isolation_forest_model.pkl`
- `models/lstm_autoencoder_model.keras`
- `models/feature_scaler.pkl`

**Visualizations (17+ images):**
- Time series plots
- Distributions
- Correlation matrices
- Confusion matrices
- ROC curves
- Model comparisons

**Metrics:**
- `results/isolation_forest_metrics.json`
- `results/lstm_autoencoder_metrics.json`
- `results/model_comparison.csv`

---

## 🐛 Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### Notebook shows old errors
```
Kernel → Restart Kernel (in Jupyter)
```

### "LSTM training is slow"
Edit `config.py`:
```python
LSTM_PARAMS = {
    'epochs': 20,  # Reduce from 50
    ...
}
```

---

## ✨ Everything is Working!

The code is tested and ready to use. Just:
1. Install dependencies
2. Run `python3 run_pipeline.py` OR use notebook (restart kernel first)
3. Check results in `results/plots/`

**That's it!** 🎉
