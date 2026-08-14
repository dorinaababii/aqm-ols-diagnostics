"""Shared regression/diagnostics logic for AQM_Diagnostics.ipynb.

This module exists (rather than living inline in the notebook) because
concurrent.futures.ProcessPoolExecutor uses the 'spawn' start method on macOS
and Windows, which requires the worker target to be importable from a real
module -- a function/class defined in a notebook's __main__ namespace can't
be pickled back into a fresh worker interpreter.
"""

import os
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt

# Populated once per worker process by init_worker(), and once in the main
# process directly -- avoids re-pickling the full DataFrames for every task.
_D5 = None
_D6 = None


def init_worker(data_path: str) -> None:
    """ProcessPoolExecutor initializer: load the input data once per worker."""
    global _D5, _D6
    _D5 = pd.read_excel(data_path, sheet_name="StockPrice", index_col=0)
    _D6 = pd.read_excel(data_path, sheet_name="MacroIndicators", index_col=0)


class RegressionDiagnostics:
    """Fit-once diagnostics for a single OLS model.

    Two ways to build one:
    - RegressionDiagnostics(company, x, y): fits a new OLS model (used by
      fit_combination in worker processes, and directly in the notebook for
      the 10-variable single-model-per-company section).
    - RegressionDiagnostics.from_result(...): reconstructs a plotting-capable
      instance from an already-computed ComboResult, with no refit -- used by
      the best-model step so the winning combination's diagnostics aren't
      recomputed a second time.
    """

    def __init__(self, company: str, x: pd.DataFrame, y: pd.Series):
        self.company = company
        self.y = y
        x_ = sm.add_constant(x)
        self.model = sm.OLS(y, x_, missing="drop")
        self.results = self.model.fit()
        self.predictions = self.results.predict(x_)
        self.residuals = self.y - self.predictions
        self.adj_r2 = self.results.rsquared_adj

    @classmethod
    def from_result(cls, company: str, y: pd.Series, predictions, residuals, index_labels, adj_r2: float):
        # index_labels are the actual labels statsmodels retained after
        # missing='drop' -- must not assume the dropped rows (if any) were a
        # trailing slice of y, since sm.OLS can drop rows anywhere.
        obj = cls.__new__(cls)
        obj.company = company
        obj.y = y
        obj.predictions = pd.Series(predictions, index=index_labels)
        obj.residuals = pd.Series(residuals, index=index_labels)
        obj.adj_r2 = adj_r2
        obj.model = None
        obj.results = None
        return obj

    def breusch_pagan(self):
        return sms.het_breuschpagan(self.residuals, self.results.model.exog)

    def jarque_bera(self):
        return sms.jarque_bera(self.residuals)

    def durbin_watson(self):
        return sm.stats.stattools.durbin_watson(self.residuals)

    def summary(self):
        return self.results.summary()

    def plot_observed_vs_predicted(self, title: str, path: str) -> None:
        df = pd.DataFrame(self.y)
        df["Prediction"] = self.predictions.values
        plt.plot(df)
        plt.title(title)
        plt.savefig(path)
        plt.close()

    def plot_residuals(self, title: str, path: str) -> None:
        plt.plot(self.residuals)
        plt.title(title)
        plt.savefig(path)
        plt.close()


@dataclass
class ComboResult:
    """Picklable result of fitting one (company, 7-variable combination) pair."""

    company: str
    combination: Tuple[str, ...]
    combo_index: int
    adj_r2: float
    bp_result: Tuple[float, ...]
    jb_result: Tuple[float, ...]
    dw_statistic: float
    predictions: List[float]
    residuals: List[float]
    index_labels: list
    summary_text: str


def fit_combination(args):
    """Fit and diagnose one (company, combination) pair. The shared, single
    source of truth for FR-001: called both directly (from fit_company, once
    per combination) and available for standalone use/testing.
    """
    company, combo_vars, combo_index = args
    x = _D6[list(combo_vars)]
    y = _D5[company]
    diag = RegressionDiagnostics(company, x, y)
    bp = diag.breusch_pagan()
    jb = diag.jarque_bera()
    dw = diag.durbin_watson()
    return ComboResult(
        company=company,
        combination=tuple(combo_vars),
        combo_index=combo_index,
        adj_r2=float(diag.adj_r2),
        bp_result=tuple(float(v) for v in bp),
        jb_result=tuple(float(v) for v in jb),
        dw_statistic=float(dw),
        predictions=diag.predictions.to_list(),
        residuals=diag.residuals.to_list(),
        index_labels=diag.predictions.index.to_list(),
        summary_text=str(diag.summary()),
    )


@dataclass
class CompanyResult:
    """Picklable summary of one company's full 120-combination loop.

    The ProcessPoolExecutor worker target (fit_company) does all of that
    company's own plotting and Excel writes itself, into files namespaced
    uniquely to the company -- no other worker ever touches those paths, so
    there's no cross-worker race. Only the two genuinely shared outputs (the
    best-models workbook, the sqlite db) still need data back in the main
    process, plus enough to reproduce the notebook's printed diagnostics.
    """

    company: str
    min_adj_r2: float
    max_adj_r2: float
    best_combination: Tuple[str, ...]
    best_combo_index: int
    best_bp_result: Tuple[float, ...]
    best_jb_result: Tuple[float, ...]
    best_dw_statistic: float
    best_summary_text: str
    best_predictions: List[float]
    best_index_labels: list


def fit_company(args):
    """ProcessPoolExecutor worker target: one company's full 7-variable loop.

    Fits all 120 combinations, writes this company's own prediction workbook
    and per-combination/best-model plots, and returns a compact CompanyResult
    for the main process to print and to write into the shared best-models
    workbook and sqlite db.
    """
    company, combinations_list = args

    os.makedirs("output/7VarOLS_PredictedData", exist_ok=True)
    os.makedirs("output/ObservedVsPredictedPricesGraphs", exist_ok=True)
    os.makedirs("output/ResidualsGraphs", exist_ok=True)
    os.makedirs("output/BestModels", exist_ok=True)

    w1 = pd.ExcelWriter(f"output/7VarOLS_PredictedData/AQM_{company}.xlsx", engine="xlsxwriter")

    y = _D5[company]
    adj_r2_values = []
    best = None

    for idx, combo_vars in enumerate(combinations_list, start=1):
        result = fit_combination((company, combo_vars, idx))
        adj_r2_values.append(result.adj_r2)

        diag = RegressionDiagnostics.from_result(
            company, y, result.predictions, result.residuals, result.index_labels, result.adj_r2
        )

        m = str(idx)
        regression1 = pd.DataFrame(diag.y)
        regression1["Prediction"] = diag.predictions
        regression1.to_excel(w1, sheet_name=m)

        diag.plot_observed_vs_predicted(
            f"Observed vs. Predicted Stock Price{company} {m}",
            f"output/ObservedVsPredictedPricesGraphs/ObsVsPred {company} {m}.png",
        )
        diag.plot_residuals(
            f"Residuals{company} {m}",
            f"output/ResidualsGraphs/Resid {company} {m}.png",
        )

        if best is None or result.adj_r2 > best.adj_r2:
            best = result

    w1.close()

    dm = str(best.combo_index - 1)  # 0-based, matching the original lst.index(max(lst)) numbering

    plt.rc("figure", figsize=(12, 7))
    plt.text(0.01, 0.05, best.summary_text, {"fontsize": 10}, fontproperties="monospace")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"output/BestModels/Table_{company}_{dm}.png")
    plt.close()

    best_diag = RegressionDiagnostics.from_result(
        company, y, best.predictions, best.residuals, best.index_labels, best.adj_r2
    )
    best_diag.plot_observed_vs_predicted(
        f"Observed vs. Predicted Stock Price{company} {dm}",
        f"output/ObservedVsPredictedPricesGraphs/ObsVsPred {company} {dm}.png",
    )
    best_diag.plot_residuals(
        f"Residuals{company} {dm}",
        f"output/ResidualsGraphs/Resid {company} {dm}.png",
    )

    return CompanyResult(
        company=company,
        min_adj_r2=float(min(adj_r2_values)),
        max_adj_r2=float(max(adj_r2_values)),
        best_combination=best.combination,
        best_combo_index=best.combo_index,
        best_bp_result=best.bp_result,
        best_jb_result=best.jb_result,
        best_dw_statistic=best.dw_statistic,
        best_summary_text=best.summary_text,
        best_predictions=best.predictions,
        best_index_labels=best.index_labels,
    )
