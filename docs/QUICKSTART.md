# Quick Start Guide

Get up and running with the IoT Anomaly Detection system in under 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- 2GB free disk space
- (Optional) GPU for faster LSTM training

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /Users/deeptanush.kapaka/Documents/project
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages (~500MB).

## Running the Project

### Option A: Complete Automated Pipeline (Recommended)

Run everything with a single command:

```bash
python3 run_pipeline.py
```

**What it does:**
1. ✓ Generates synthetic sensor data (10,000 samples)
2. ✓ Creates 4 EDA visualizations
3. ✓ Engineers 100+ features
4. ✓ Trains Isolation Forest model
5. ✓ Trains LSTM Autoencoder model
6. ✓ Generates 17+ visualization plots
7. ✓ Saves all models and metrics
8. ✓ Creates comparison reports

**Expected time:** 10-15 minutes

**Output locations:**
- Data: `data/sensor_data.csv`
- Models: `models/`
- Visualizations: `results/plots/`
- Metrics: `results/*.json`

### Option B: Interactive Jupyter Notebook

For step-by-step exploration:

```bash
jupyter notebook anomaly_detection_notebook.ipynb
```

Then run cells sequentially (Shift+Enter).

### Option C: Individual Modules

Run specific components:

```bash
# Generate data only
python3 data_generator.py

# Run EDA only
python3 eda_visualization.py

# Train Isolation Forest only
python3 model_isolation_forest.py

# Train LSTM Autoencoder only
python3 model_lstm_autoencoder.py
```

## Viewing Results

### 1. Check Console Output

The pipeline prints progress and metrics to the console.

### 2. View Visualizations

Open the plots directory:

```bash
open results/plots/  # macOS
# or
explorer results\plots\  # Windows
# or
xdg-open results/plots/  # Linux
```

**Key visualizations:**
- `01_time_series.png` - Sensor data with anomalies
- `02_distributions.png` - Data distributions
- `08_if_confusion_matrix.png` - Isolation Forest performance
- `12_lstm_confusion_matrix.png` - LSTM performance
- `16_model_comparison.png` - Side-by-side comparison

### 3. Review Metrics

```bash
cat results/model_comparison.csv
```

### 4. Read Summary

```bash
open SUMMARY.md  # or use your text editor
```

## Expected Results

After running the pipeline, you should see:

**Isolation Forest:**
- Accuracy: ~85-90%
- F1-Score: ~72-82%
- Training time: <10 seconds

**LSTM Autoencoder:**
- Accuracy: ~88-92%
- F1-Score: ~77-87%
- Training time: 5-10 minutes

## Troubleshooting

### Issue: "ModuleNotFoundError"

**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: "LSTM training is slow"

**Solution:** Reduce epochs in config.py
```python
LSTM_PARAMS = {
    'epochs': 20,  # Reduce from 50
    ...
}
```

### Issue: "Out of memory"

**Solution:** Reduce batch size or data size
```python
NUM_SAMPLES = 5000  # Reduce from 10000
LSTM_PARAMS = {
    'batch_size': 16,  # Reduce from 32
    ...
}
```

### Issue: "Plots not displaying"

**Solution:** Check matplotlib backend
```python
import matplotlib
matplotlib.use('Agg')  # For non-interactive backend
```

## Next Steps

1. **Explore Results**
   - Review all visualizations in `results/plots/`
   - Read `SUMMARY.md` for detailed insights

2. **Customize**
   - Edit `config.py` to adjust parameters
   - Modify models in `model_*.py` files

3. **Use Your Data**
   - Replace synthetic data with real sensor data
   - Update sensor names in `config.py`
   - Re-run pipeline

4. **Deploy**
   - Load saved models from `models/`
   - Use for real-time prediction
   - Integrate with monitoring system

## Getting Help

- **Documentation**: See `README.md` for detailed information
- **Code Comments**: All modules are well-documented
- **Summary**: Read `SUMMARY.md` for insights and recommendations

## Project Structure

```
project/
├── run_pipeline.py          ← Run this for complete pipeline
├── config.py                ← Adjust parameters here
├── data_generator.py        ← Synthetic data generation
├── data_preprocessing.py    ← Preprocessing & features
├── eda_visualization.py     ← Exploratory analysis
├── model_isolation_forest.py ← Statistical model
├── model_lstm_autoencoder.py ← Deep learning model
├── model_evaluation.py      ← Evaluation & comparison
├── requirements.txt         ← Dependencies
├── README.md               ← Full documentation
├── SUMMARY.md              ← Findings & insights
└── QUICKSTART.md           ← This file
```

## Success Checklist

After running the pipeline, verify:

- [ ] `data/sensor_data.csv` exists
- [ ] `results/plots/` contains 17+ PNG files
- [ ] `models/` contains saved models
- [ ] `results/model_comparison.csv` exists
- [ ] Console shows "PIPELINE EXECUTION COMPLETE!"

If all checked, you're ready to go! 🎉

---

**Need more details?** See `README.md`

**Want insights?** Read `SUMMARY.md`

**Have questions?** Check code comments in source files

