# What Predicts a Country's Life Expectancy?

A portfolio modeling project for **AAE 718 (Summer 2026), Project 04**. It trains a
random-forest regressor to predict national life expectancy from development indicators,
and asks which factors matter most.

**Read the report:** [`report.md`](report.md) (a PDF version is submitted to Canvas).

## Question

Life expectancy is one of the clearest single summaries of a country's wellbeing. Using the
WHO Life Expectancy dataset (193 countries, 2000-2015, ~20 indicators per country-year), this
project asks: **how well can we predict a country's life expectancy from its health, economic,
and social indicators, and which factors carry the most weight?**

## Findings (short version)

- A random forest predicts life expectancy with **test R² ≈ 0.97** and a mean absolute error
  of about **1 year**, far above a linear baseline (R² ≈ 0.82).
- The most important predictors are **HIV/AIDS prevalence**, the **human-development "income
  composition" index**, **adult mortality**, **schooling**, and **BMI**.

## Data

- Source: **WHO Life Expectancy** dataset on Kaggle (`kumarajarshi/life-expectancy-who`),
  pulled at runtime through the **Kaggle API**. No data files are committed (see `.gitignore`).

## Getting a Kaggle API key (required)

The Kaggle API needs a free key:

1. Create a free account at https://www.kaggle.com and sign in.
2. Go to **Account → Settings → API → "Create New Token."** This downloads `kaggle.json`.
3. Place it where the Kaggle library looks for it:
   - **macOS/Linux:** `~/.kaggle/kaggle.json` (then `chmod 600 ~/.kaggle/kaggle.json`)
   - **Windows:** `C:\Users\<you>\.kaggle\kaggle.json`
4. `kaggle.json` is listed in `.gitignore` so the key is never committed.

## How to run

```bash
pip install -r requirements.txt
python life_expectancy_model.py
```

`fetch_data()` connects to the Kaggle API, downloads and caches the dataset under `data/`
(git-ignored), then the script cleans it, splits train/test, fits the model, prints the
training and test scores, and writes three figures to `images/`.

## Repository layout

```
.
├── life_expectancy_model.py   # fetch (Kaggle API) + clean + model + figures
├── report.md                  # written report
├── README.md
├── requirements.txt
├── .gitignore                 # excludes data/, *.csv, kaggle.json, *.pdf
└── images/                    # generated figures (fig1-fig3)
```

## Method notes

Target: life expectancy at birth. Features: all other indicators except country name, with
Status encoded as Developed/Developing. Missing values are imputed with **training-set**
medians (computed after the split, to avoid leakage). Model: `RandomForestRegressor`
(300 trees), compared against a linear-regression baseline, evaluated by R² on a held-out 20%
test set.
