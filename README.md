# Predicting Student Performance

Can a student's final grade be predicted from their background, study habits, and
behaviour — before any exam results are known?

## Dataset

Student Performance dataset, UCI Machine Learning Repository (Cortez & Silva, 2008):
two Portuguese secondary schools, two subjects.

- `data/student-mat.csv` — Mathematics, 395 students (primary analysis)
- `data/student-por.csv` — Portuguese language, 649 students (comparison)

Both files are semicolon-separated. Target: **G3** (final grade, 0–20). No missing
values and no duplicate rows in either file.

## Question

Predict G3 from background and behaviour features only. G1 and G2 (first- and
second-period grades) are deliberately **excluded** from the main models because
they are near-proxies of the final grade; a comparison at the end of the notebook
quantifies how much they would add.

## Methods

- **EDA** — G3 distribution; G3 by study time, past failures, extra support,
  paid classes, sex, guardian, alcohol use, parental education; absences vs G3.
- **Feature engineering** — parental education average `(Medu+Fedu)/2`, total
  alcohol use `Dalc+Walc`, social time index `goout+freetime`, high-absence flag
  (10+ absences).
- **Hypothesis tests** — one-way ANOVA across study-time levels and across
  guardian types; Welch's t-test for school (GP vs MS, two groups only).
- **Modelling** — 80/20 train/test split (`random_state=42`); one-hot encoding
  and scaling inside a scikit-learn `Pipeline`. Linear Regression (baseline),
  Ridge, Random Forest and XGBoost, the last two tuned with `RandomizedSearchCV`
  (12 iterations, 3-fold CV). Metrics: RMSE, MAE, R² on the held-out test set.

## Key findings

- **Study time shows a modest gradient, but it is not significant.** Mean
  grades rise from 10.1 (under 2 hours a week) to 11.3 (over 10 hours), yet the
  differences across study-time levels do not reach significance
  (ANOVA F = 1.73, p = 0.161).
- **School and guardian do not reach significance** in the mathematics sample:
  school GP vs MS, Welch's t = 0.96, p = 0.343 (MS n = 46, so the
  test has limited power); guardian, ANOVA F = 1.59, p = 0.205.
- **Parental education is one of the clearest gradients**: setting aside the
  2–3 students whose parents have no formal education, mean grades rise
  steadily from about 8.7–9.2 (primary) to 11.4–11.8 (higher education).
- **Past failures and absences dominate the model.** The best model's three most
  important features are absences (0.20), failures (0.16), health (0.05).
  Notably, absences tops the list despite a near-zero linear correlation
  (r = 0.03) — the pattern is non-monotonic: zero-absence students average 8.4,
  below the 11.5 of students with 1–10 absences.
- **Extra school support looks negative — but that is selection**: support goes
  to students who are already struggling.
- **Best model (mathematics, without G1/G2): RandomForest** — test RMSE 3.90,
  MAE 3.12, R² 0.26. Predictions are typically within about
  3.12 grade points of the true final grade.
- **Adding G1/G2 lifts R² to 0.81** (XGBoost, RMSE 1.99) — the
  jump shows how much signal sits in earlier grades, and why the no-G1/G2 model
  is the honest one for early warning.
- **Portuguese comparison:** best model RandomForest, R² 0.21 (RMSE 2.77,
  MAE 2.00) — broadly consistent with mathematics.

**Caveat:** background and behaviour factors alone explain only part of the
variance. Most of what determines a final grade (prior knowledge, teaching
quality, motivation) is not in this dataset, and with only 395 students the
tuning grids were kept deliberately modest.

## Model comparison (mathematics, test set, without G1/G2)

| Model | RMSE | MAE | R² |
| --- | ---: | ---: | ---: |
| LinearRegression | 4.181 | 3.395 | 0.147 |
| Ridge | 4.179 | 3.394 | 0.148 |
| RandomForest | 3.901 | 3.124 | 0.258 |
| XGBoost | 3.965 | 3.148 | 0.233 |

## Files

- `student-performance.ipynb` — the full executed analysis
- `figures/` — all charts as PNGs
- `data/` — the two source CSVs (plus the UCI documentation files)
