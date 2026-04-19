# AI-Driven Crime Trend Analysis in India

A machine-learning study of India's criminal-justice investigation pipeline using NCRB *Crime in India* data (2018–2021).

> Course project for **EE 769: Introduction to Machine Learning**.

## What it does

We took four years of NCRB crime reports, cleaned them into a single dataset (519 rows, 125 IPC crime heads), and tried to predict two things:

- **Chargesheeting rate** — how often police file a chargesheet after disposing a case
- **Pendency percentage** — how many investigations stay unresolved at year end

We benchmarked 7 models (Linear, Ridge, Lasso, Random Forest, SVR, Gradient Boosting) against two honest baselines using time-ordered cross-validation, and ran K-Means to find natural "behaviour profiles" among crime types.

## Key observations

- **A dumb baseline beat every ML model** on chargesheeting rate. Just predicting each crime's historical average (RMSE 8.74) outperformed Random Forest (15.21). Year-to-year variation is tiny compared to differences between crimes.
- **Lasso narrowly won on pendency** (RMSE 8.03), and a single feature — `pendency_load` — drove **89%** of the Random Forest's predictions. Backlog begets backlog.
- **COVID hit volume, not quality.** 2020 saw a dip in reported cases and a spike in pendency (40% → 44%), but chargesheeting stayed flat.
- **K-Means found 3 clean clusters:** well-prosecuted crimes (violent, assault), property-crime backlogs (theft, fraud), and rare high-pendency crimes (counterfeiting, trafficking).
- **Watch for target leakage.** Our first feature set had `fr_rate` correlated −0.99 with the target — it was basically the target in disguise. Always screen correlations before modelling.

## Files

```
├── Datasets/                # Raw NCRB CSVs (2018–2021)
├── EE769-Project.ipynb      # Full pipeline notebook
├── EE769_Report.pdf         # Project report
└── pipeline_master.csv      # Cleaned dataset
```

## Run it

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook EE769-Project.ipynb
```

## Authors

- **Dheeraj Kumar Mishra** — EE, IIT Bombay
- **Dhanushk Sonkusare** — CSE, IIT Bombay

Guided by **Prof. Amit Sethi**, IIT Bombay. Data courtesy of the National Crime Records Bureau (NCRB).
