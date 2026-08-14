# AQM OLS Diagnostics

Testing for heteroskedasticity, multicollinearity, and serial autocorrelation in
OLS regressions of EU stock returns on EU macroeconomic indicators.

## Context

This is a coursework project for **Applied Quantitative Methods**, Zurich University
of Applied Sciences (ZHAW), **February–June 2021** — not a professional publication.
The written paper that originally accompanied this analysis no longer exists; this
repository preserves the underlying code and data.

## Methodology

- **Data**: monthly stock prices for 10 European companies (BP, Total SE, Volkswagen,
  BMW, Allianz, Deutsche Bank, L'Oréal, Schneider Electric, Danone, Vinci) and 10 EU
  macroeconomic indicators (consumer prices, construction output, unemployment,
  interest rates, economic sentiment, production, retail sales, exports, etc.).
- **Regression**: multivariate OLS of each stock's price on macro-indicator
  combinations — both an exhaustive search over every 7-of-10 variable combination
  (120 combinations × 10 companies) and a single 10-variable model per company,
  selecting each company's best 7-variable model by adjusted R².
- **Multicollinearity**: variance inflation factor (VIF) on the macro indicators,
  with a variable-combination fix applied where VIF was too high.
- **Heteroskedasticity**: Breusch-Pagan test on regression residuals.
- **Normality**: Jarque-Bera test on regression residuals.
- **Serial autocorrelation**: Durbin-Watson statistic on regression residuals.

The methodology itself (including the multicollinearity fix in the notebook) is
unchanged from the original coursework submission — this repository only makes the
existing analysis portable and runnable outside of Google Colab.

## Repository contents

```
AQM_Diagnostics.ipynb   the analysis notebook
aqm_lib.py               shared regression/diagnostics code (RegressionDiagnostics,
                          the parallel-fit worker) used by the notebook
data/
  AQM_1.xlsx             StockPrice + MacroIndicators sheets (source data)
  AQM_VIF.xlsx           original precomputed VIF output, kept for reference
requirements.txt
LICENSE                  MIT (code only)
```

`output/` (git-ignored) is created by running the notebook — it holds the sqlite
database, per-combination prediction spreadsheets, and several hundred plots from
the 7-variable and 10-variable regression loops. It is not committed.

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook AQM_Diagnostics.ipynb
```

The notebook reads directly from `data/AQM_1.xlsx`; no external API, Google Drive,
or Colab environment is required. The 7-variable loop's 1,200 regressions (120
combinations × 10 companies) run in parallel across worker processes (via
`aqm_lib.py`, using `concurrent.futures.ProcessPoolExecutor`), which keeps a full
run to a few minutes on a modern multi-core machine rather than the much longer
sequential runtime a naive loop would take. To just see the results, reading
through the notebook's own printed output and existing cell structure is enough;
you don't need to run it end-to-end.

## Note on the data

The stock price and macroeconomic data in `data/AQM_1.xlsx` was collected as part
of the original 2021 coursework. It's published here as originally gathered, for
transparency about what the analysis was run on.
