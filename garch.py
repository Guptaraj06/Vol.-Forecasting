import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import EGARCH, FIGARCH, GARCH


class GARCHFitter:
    GARCH_SPECS = {
        "GARCH": lambda r: arch_model(
            r, vol="Garch", p=1, q=1, dist="t", mean="AR", lags=1
        ),
        "GARCH-N": lambda r: arch_model(
            r, vol="Garch", p=1, q=1, dist="normal", mean="Constant"
        ),
        "EGARCH": lambda r: arch_model(
            r, vol="EGarch", p=1, q=1, dist="t", mean="AR", lags=1
        ),
        "GJR-GARCH": lambda r: arch_model(
            r, vol="Garch", p=1, o=1, q=1, dist="t", mean="AR", lags=1
        ),
        "ARCH": lambda r: arch_model(r, vol="ARCH", p=5, dist="t", mean="Constant"),
    }

    def fit_all(self, returns: pd.Series, scale: float = 100.0) -> dict:
        r_scaled = returns * scale
        results = {}
        for name, spec_fn in self.GARCH_SPECS.items():
            try:
                model = spec_fn(r_scaled)
                res = model.fit(
                    disp="off", show_warning=False, options={"maxiter": 500}
                )
                results[name] = {
                    "result": res,
                    "aic": round(res.aic, 2),
                    "bic": round(res.bic, 2),
                    "loglik": round(res.loglikelihood, 2),
                    "params": res.params.to_dict(),
                    "std_resid": res.std_resid,
                    "cond_vol": res.conditional_volatility / scale,
                }
                print(
                    f"  {name:12s} | AIC={res.aic:.1f} BIC={res.bic:.1f} LL={res.loglikelihood:.1f}"
                )
            except Exception as e:
                print(f" {name} :FAILED ({e})")

        return results

    def sign_bias_test(self, std_resid: np.ndarray):
        std_resid = np.asarray(std_resid, dtype=float)
        std_resid = std_resid[np.isfinite(std_resid)]
        z = std_resid[:-1]
        z_sq_nxt = std_resid[1:] ** 2
        S_neg = (z < 0).astype(float)
        S_pos = (z > 0).astype(float)

        T = len(z)
        X = np.column_stack([np.ones(T), S_neg, S_neg * z, S_pos * z])
        beta = np.linalg.lstsq(X, z_sq_nxt, rcond=None)[0]
        resid_reg = z_sq_nxt - X @ beta
        sigma2_reg = (resid_reg**2).mean()
        se = np.sqrt(np.diag(sigma2_reg * np.linalg.inv(X.T @ X)))

        t_stats = beta / se
        p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=T - 4))

        return {
            "sign_bias_pval": round(p_vals[1], 4),
            "neg_size_bias_pval": round(p_vals[2], 4),
            "pos_size_bias_pval": round(p_vals[3], 4),
            "use_asymmetric": bool(any(p < 0.05 for p in p_vals[1:4])),
        }


def persistence(self, params: dict, model_name: str) -> dict:
    if model_name == "EGARCH":
        persistence = abs(params.get("beta[1]", params.get("b[1]", 0)))
    else:
        alpha = params.get("alpha[1]", 0)
        beta = params.get("beta[1]", 0)
        gamma = params.get("gamma[1]", 0)
        persistence = alpha + beta + 0.5 * gamma

    halflife_vol = np.log(0.5) / np.log(persistence) if 0 < persistence < 1 else np.inf

    return {
        "persistence": round(persistence, 4),
        "stationary": persistence < 1,
        "halflife_days": round(halflife_vol, 1),
    }
