# HCIA_FocusDNA_ML

FocusDNA — **Study Session Predictor** using synthetic study-session data, scikit-learn preprocessing, and saved classification/regression models.

## Project layout

```text
HCIA_FocusDNA_ML/
├── data/
│   ├── focusdna_dataset.csv
│   ├── classification_results.csv
│   └── regression_results.csv
├── Notebooks/
│   └── FocusDNA.ipynb
├── models/
│   ├── preprocessor.pkl
│   ├── classifier.pkl
│   ├── regressor.pkl
│   ├── input_features.pkl
│   └── processed_features.pkl
└── frontend/
    ├── app.py
    ├── requirements.txt
    ├── assets/styles.css
    ├── core/
    ├── ui/
    └── views/
```

## Streamlit dashboard

### Install

From the project root:

```bash
pip install -r frontend/requirements.txt
```

### Run

```bash
streamlit run frontend/app.py
```

Use the **Appearance** control at the top of the sidebar to switch **Dark** / **Light** mode.

The app loads artifacts from `models/` and data from `data/` via paths relative to the repository root (works regardless of your shell working directory).

### Pages

- **Overview** — dataset summary and metrics from saved evaluation CSVs  
- **Predictor** — form driven by `input_features.pkl`; live inference  
- **Focus Simulator** — baseline vs what-if using the same saved models  
- **Model Lab** — classification/regression comparison charts  
- **Dataset Explorer** — distributions, filters, correlation heatmap  
- **Why This Prediction?** — linear model coefficients (associations, not causation)

Predictions use: `preprocessor.transform` → `classifier.predict` / `regressor.predict` (classifier applies its internal `StandardScaler` only once on processed features).
