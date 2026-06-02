import warnings
from itertools import product

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf

warnings.filterwarnings("ignore")


class ARIMAFitter:
    def __init__(self, max_p=4, max_q=4, d=0, ic="bic"):
        self.max_p = max_p
        self.max_q = max_q
        self.d = d
        self.ic = ic
        self.best_order = None
        self.best_result = None
        self.ic_table = None

    def acf_pacf_orders(self, returns: np.ndarray, nlags=20, alpha=0.05) -> dict:
        acf_vals, acf_ci = acf(returns, nlags=nlags, alpha=alpha)
        pacf_vals, pacf_ci = pacf(returns, nlags=nlags, alpha=alpha)

        acf_sig = [
            i for i in range(1, nlags + 1) if not (acf_ci[i][0] <= 0 <= acf_ci[i][1])
        ]
        pacf_sig = [
            i for i in range(1, nlags + 1) if not (pacf_ci[i][0] <= 0 <= pacf_ci[i][1])
        ]

        suggested_q = max(acf_sig, default=0)
        suggested_p = max(pacf_sig, default=0)

        return {
            "suggested_p": min(suggested_p, self.max_p),
            "suggested_q": min(suggested_q, self.max_q),
            "acf_sig_lags": acf_sig,
            "pacf_sig_lags": pacf_sig,
        }

    def grid_search(self, returns: pd.Series) -> pd.DataFrame:
        result = []
        orders = list(product(range(self.max_p + 1), [self.d], range(self.max_q + 1)))

        for order in orders:
            try:
                m = ARIMA(returns, order=order, trend="c")
                res = m.fit(method="innovations_mle", low_memory=True)
                result.append(
                    {
                        "order": order,
                        "p": order[0],
                        "d": order[1],
                        "q": order[2],
                        "aic": round(res.aic, 2),
                        "bic": round(res.bic, 2),
                        "lif": round(res.llf, 2),
                        "n_params": order[0] + order[2] + 1,
                    }
                )

            except:
                pass

        print(f"____{res}_____")
        self.ic_table = pd.DataFrame(result).sort_values(by=self.ic)
        self.best_order = tuple(self.ic_table.iloc[0]["order"])
        return self.ic_table

    def fit_best(self, returns: pd.Series) -> object:
        if self.best_order is None:
            self.grid_search(returns)

        m = ARIMA(returns, order=self.best_order, trend="c")
        self.best_result = m.fit()
        print(f"Best Arima {self.best_order} : BIC = {self.best_result.bic:.2f}")
        return self.best_result

    def residual_diagnostics(self, result=None) -> dict:
        res = result or self.best_result
        resid = res.resid.dropna()

        lb_resid = acorr_ljungbox(resid, lags=[10], return_df=True)
        lb_sq = acorr_ljungbox(resid**2, lags=[10], return_df=True)
        jb_stat, jb_p = stats.jarque_bera(resid)

        diag_engine = DiagnosticsEngine(resid)
        arch_test = diag_engine.arch_lm_test(lags=5)

        return {
            "lb_resid_pval_lag10": float(lb_resid["lb_pvalue"].iloc[0]),
            "lb_squared_pval_lag10": float(lb_sq["lb_pvalue"].iloc[0]),
            "arch_lm_pval": arch_test["pvalue"],
            "arch_present": arch_test["arch_present"],
            "jb_pval": round(jb_p, 4),
            "excess_kurt": round(float(resid.kurt()), 4),
            "mean_resid": round(float(resid.mean()), 6),  # should be ≈ 0
            "verdict": (
                "Proceed to GARCH" if arch_test["arch_present"] else "No ARCH effects"
            ),
        }
