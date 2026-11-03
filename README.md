# Predicting Power Plant Target Achievement Using Machine Learning

**Course:** INT234 – Predictive Analytics (CA2)
**Author:** Hemant Kumar
**University:** Lovely Professional University, Phagwara, India

## About the project
This project uses the **Monthly Power Generation October 2023** dataset from
[data.gov.in](https://www.data.gov.in/resource/monthly-power-generation-october-2023).
It cleans the report-style file, explores it, and compares five machine learning models that
predict whether a power plant **met its October 2023 generation program** (1 = met, 0 = missed).

## Dataset
- Raw file: `October_1_1.csv` (1,026 rows, 24 columns, includes subtotal rows)
- After cleaning: 429 plants with a valid target (238 met, 191 missed)
- Fuel types: Coal, Hydro, Natural Gas, Nuclear, Diesel

## Project files
| File | What it does |
|---|---|
| `cleaning.py` | Renames columns, removes subtotal rows, fills missing values, fixes percentages, saves `power_cleaned.csv` |
| `eda.py` | Creates the target column and 14 EDA graphs, saves `power_eda.csv` |
| `models.py` | Trains and compares 5 models, saves `model_results.csv` and charts |

Run them in this order, in the same folder: `cleaning.py` → `eda.py` → `models.py`.

## Models and results (test set of 86 plants)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7326 | 0.7451 | 0.7917 | **0.7677** | 0.7714 |
| KNN | 0.6279 | 0.6481 | 0.7292 | 0.6863 | 0.6960 |
| Decision Tree | **0.7442** | **0.8250** | 0.6875 | 0.7500 | 0.7555 |
| Random Forest | 0.7326 | 0.7551 | 0.7708 | 0.7629 | **0.8048** |
| Gradient Boosting | 0.7093 | 0.7674 | 0.6875 | 0.7253 | 0.7887 |

Setup: stratified 80:20 split, `random_state=42`, Min-Max scaling. Columns built from actual
October generation were left out of the inputs to avoid data leakage.

## Key findings
- Coal and nuclear plants met their program more often than hydro plants.
- Large plants (above 500 MW) met target about 70% of the time, small and medium plants about 43%.
- The Western region performed best and the Southern region worst.
- Plant attributes explain part of the outcome, but not outages, water inflow or fuel supply.

## How to run
```
pip install -r requirements.txt
python cleaning.py
python eda.py
python models.py
```

## Report
The IEEE-format report is in `report/IEEE_Report_Hemant_Kumar.pdf`.

## Note
This is an educational project and not a tool for real grid planning.
