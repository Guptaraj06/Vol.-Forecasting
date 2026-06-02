import numpy as np
import pandas as pd


class ForecastEvaluator:
    def __init__(self, vix: pd.Series = None):
        self.vix = vix

    def _align(self, fc_df: pd.DataFrame, proxy_col: str = "actual_var") -> tuple:
        var_fc = fc_df["var_forecast"]
        rv = fc_df[proxy_col]
        idx = var_fc.index.intersection(rv.index)
        return var_fc.loc[idx], rv.loc[idx]

    def loss_functions(self, fc_df: pd.DataFrame) -> dict:
        var_fc, rv = self._align(fc_df)
        e = var_fc - rv
        mse = (e**2).mean()
        mae = e.abs().mean()
        qlike = (rv / var_fc - np.log(rv / var_fc) - 1).mean()
        hmse = ((1 - var_fc / rv) ** 2).mean()
        r2 = 1 - (e**2).sum() / ((rv - rv.mean()) ** 2).sum()

        return {
            "MSE": mse,
            "RMSE": mse**0.5,
            "MAE": mae,
            "QLIKE": qlike,
            "HMSE": hmse,
            "R2": r2,
        }

    def mincer_zarnowitz(self, forecasts: dict) -> pd.DataFrame:
        rows = {}
        for name, fc_df in forecasts.items():
            var_fc, rv = self._align(fc_df)
            T = len(rv)
            X = np.column_stack([np.ones(T), var_fc.values])
            Y = rv.values
            b, res, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            y_hat = X @ b
            ss_res = ((Y - y_hat) ** 2).sum()
            ss_tot = ((Y - Y.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot
            sigma2 = ss_res / (T - 2)

            V = sigma2 * np.linalg.inv(X.T @ X)
            R = np.array([[1, 0], [0, 1]])
            r_vec = np.array([0, 1])
            diff = R @ b - r_vec
            wald = diff @ np.linalg.inv(R @ V @ R.T) @ diff.T
            pval_wald = 1 - stats.chi2.cdf(wald, df=2)

            rows[name] = {
                "alpha": round(b[0], 4),
                "beta": round(b[1], 4),
                "R2": round(r2, 4),
                "wald_stat": round(wald, 4),
                "wald_pval": round(pval_wald, 4),
                "unbiased": pval_wald > 0.05,
            }

        return pd.DataFrame(rows).T

    def diebold_mariano(
        self, forecasts: dict, benchmark: str = "GARCH", loss: str = " QLIKE"
    ) -> pd.DataFrame:
        bench_df = forecasts[benchmark]
        var_bm, rv_bm = self._align(bench_df)
        loss_bm = rv_bm / var_bm - np.log(rv_bm / var_bm) - 1

        rows = {}
        for name, fc_df in forecasts.items():
            if name == benchmark:
                continue
            var_fc, rv_fc = self._align(fc_df)
            loss_fc = rv_fc / var_fc - np.log(rv_fc / var_fc) - 1

            idx = loss_bm.index.intersection(loss_fc.index)
            d = loss_bm.loc[idx] - loss_fc.loc[idx]
            T = len(d)

            bw = int(T ** (1 / 3))
            var_d = np.var(d, ddof=1)
            for k in range(1, bw + 1):
                gamma_k = np.cov(d[k:], d[:-k])[0, 1]
                var_d += 2 * (1 - k / (bw + 1)) * gamma_k

            dm_stat = d.mean() / np.sqrt(var_d / T)
            pval = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

            rows[name] = {
                "DM_stat": round(dm_stat, 3),
                "pvalue": round(pval, 4),
                "beats_benchmark": dm_stat > 0 and pval < 0.05,
            }

        return pd.DataFrame(rows).T

    def vix_comparison(self, fc_df: pd.DataFrame) -> dict:
        if self.vix is None:
            return {}

        vix_vol = (self.vix / 100) / np.sqrt(252)
        vix_var = vix_vol**2

        idx = fc_df.index.intersection(vix_var.index)

        garch_var = fc_df["var_forecast"].loc[idx]
        vix_v = vix_var.loc[idx]
        rv = fc_df["actual_var"].loc[idx]

        corr_garch_vix = float(garch_var.corr(vix_v))
        corr_garch_rv = float(garch_var.corr(rv))
        corr_vix_rv = float(vix_v.corr(rv))

        return {
            "garch_vix_corr": round(corr_garch_vix, 3),
            "garch_rv_corr": round(corr_garch_rv, 3),
            "vix_rv_corr": round(corr_vix_rv, 3),
            "interpretation": (
                "VIX forecasts forward-looking implied vol; GARCH uses historical data. "
                "VIX typically higher than GARCH (variance risk premium). "
                f"VIX beats GARCH on RV correlation: {corr_vix_rv > corr_garch_rv}"
            ),
        }
