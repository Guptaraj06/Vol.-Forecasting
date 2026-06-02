import warnings

import numpy as np
import pandas as pd
from arch.unitroot import ADF
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

warnings.filterwarnings("ignore")


class DiagnosticsEngine:
    def __init__(self, returns: pd.Series):
        self.r = returns.dropna()
        self.name = returns.name or "returns"

    def stationarity(self) -> dict:
        adf_res = adfuller(self.r, maxlag=10, regression="c", autolag="AIC")
        try:
            kpss_res = kpss(self.r, regresssion="c", nlags="auto")
            kpss_stat, kpss_val = kpss_res[0]
        except:
            kpss_stat, kpss_val = np.nan, np.nan

        adf_stationary = adf_res[1] < 0.05
        kpss_stationary = kpss_val > 0.05

        return {
            "adf_stat": round(adf_res[0], 4),
            "adf_pvalue": round(adf_res[1], 4),
            "adf_stationay": adf_stationary,
            "kpss_stat": round(kpss_stat, 4),
            "kpss_pvalue": round(kpss_val, 4),
            "kpss_stationary": kpss_stationary,
            "conclusion": (
                "Staionary"
                if adf_stationary and kpss_stationary
                else "Non-stationary-check"
            ),
        }

    def serial_correlation(self, lags=20) -> dict:
        lb_ret = acorr_ljungbox(self.r, lags=[5, 10, 20], return_df=True)
        lb_sq = acorr_ljungbox(self.r**2, lags=[5, 10, 20], return_df=True)
        lb_abs = acorr_ljungbox(self.r.abs(), lags=[5, 10, 20], return_df=True)

        return {
            "lb_returns": lb_ret["lb_pvalue"].to_dict(),
            "lb_squared": lb_sq["lb_pvalue"].to_dict(),
            "lb_absolute": lb_abs["lb_pvalue"].to_dict(),
            "arch_in_returns": bool((lb_sq["lb_pvalue"] < 0.05).any()),
        }

    def arch_lm_test(self, lags=5) -> dict:
        r = self.r.values
        resid = r - r.mean()
        resid_sq = resid**2

        X = np.column_stack(
            [resid_sq[i : len(resid_sq) - lags + i] for i in range(lags)]
        )
        Y = resid_sq[lags:]

        T = len(Y)
        X = np.column_stack([np.ones(T), X])
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        y_hat = X @ beta
        ss_res = ((Y - y_hat) ** 2).sum()
        ss_tot = ((Y - Y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        lm_stat = T * r2
        pvalue = 1 - stats.chi2.cdf(lm_stat, df=lags)

        return {
            "lm_stat": round(lm_stat, 4),
            "pvalue": round(pvalue, 4),
            "arch_present": pvalue < 0.05,
            "lags_tested": lags,
        }

    def distributional(self) -> dict:
        r = self.r.values
        jb_stat, jb_pvalues = stats.jarque_bera(r)

        sorted_r = np.sort(np.abs(r))[::-1]
        k = max(10, int(len(r) ** 0.5))
        hill_idx = 1.0 / (np.log(sorted_r[:k]).mean() - np.log(sorted_r[k]))

        return {
            "mean": round(r.mean(), 4),
            "std": round(r.std(), 4),
            "skewness": round(float(pd.Series(r).skew()), 4),
            "excess_kurtosis": round(float(pd.Series(r).kurt()), 4),
            "jarque_bera_stat": round(jb_stat, 4),
            "jarque_bera_pval": round(jb_pvalues, 4),
            "is_normal": jb_pvalues > 0.05,
            "hill_tail_index": round(hill_idx, 4),
            "ann_vol_pct": round(r.std() * np.sqrt(252) * 100, 4),
        }

    def full_report(self):

        stat = self.stationarity()
        sc = self.serial_correlation()
        arch = self.arch_lm_test()
        dist = self.distributional()

        print(f"\n{'='*55}")
        print(f"DIAGNOSTIC REPORT - {self.name.upper()}")
        print(f"{'='*55}")
        print(f"Stationarity : {stat['conclusion']}")
        print(f"ARCH effects : {'YES - FIT GARCH' if arch['arch_present'] else 'NO'}")
        print(
            f"Excess kurt  : {dist['excess_kurtosis']} ({'use Student-t' if dist['excess_kurtosis']>1 else 'Normal OK'})"
        )
        print(f"Skewness     : {dist['skewness']}")
        print(
            f"JB normality : {'NOT normal (p={:.4f})'.format(dist['jarque_bera_pval'])}"
        )
        print(f"Ann. vol     : {dist['ann_vol_pct']}%")
        print(f"{'='*55}\n")
        return {
            "stationarity": stat,
            "serial_correlation": sc,
            "arch_lm": arch,
            "distribution": dist,
        }
