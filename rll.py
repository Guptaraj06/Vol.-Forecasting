import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf


class RollingForecaster:
    def __init__(
        self,
        model_spec: str = "GJR-GARCH",
        dist: str = "t",
        horizon: int = 1,
        refit_freq: int = 21,
    ):
        self.spec = model_spec
        self.dist = dist
        self.horizon = horizon
        self.refit_freq = refit_freq

    def _build_model(self, returns_pct: pd.Series):
        if self.spec == "GARCH":
            return arch_model(
                returns_pct, vol="Garch", p=1, q=1, dist=self.dist, mean="AR", lags=1
            )
        elif self.spec == "EGARCH":
            return arch_model(
                returns_pct, vol="EGarch", p=1, q=1, dist=self.dist, mean="AR", lags=1
            )
        else:  # GJR-GARCH (default)
            return arch_model(
                returns_pct,
                vol="Garch",
                p=1,
                o=1,
                q=1,
                dist=self.dist,
                mean="AR",
                lags=1,
            )

    def run(self, returns: pd.Series, train_size: int = 504) -> pd.DataFrame:
        r_pct = returns * 100
        T = len(returns)
        forecasts = []
        dates = []
        fitted_res = None

        for t in range(train_size, T):
            window = r_pct.iloc[:t]

            if (t - train_size) % self.refit_freq == 0 or fitted_res is None:
                try:
                    m = self._build_model(window)
                    fitted_res = m.fit(disp="off", show_warning=False)
                except:
                    continue

            try:
                fc = fitted_res.forecast(
                    horizon=self.horizon,
                    reindex=False,
                    method="simulation" if self.horizon > 1 else "analytic",
                )
                var_fc = fc.variance.iloc[-1, -1] / (100**2)
                vol_fc = np.sqrt(var_fc * 252)  # annualized

                forecasts.append(
                    {
                        "date": returns.index[t],
                        "vol_forecast": vol_fc,
                        "var_forecast": var_fc,
                        "actual_ret": float(returns.iloc[t]),
                        "actual_var": float(returns.iloc[t]) ** 2,
                        "model": self.spec,
                    }
                )
                dates.append(returns.index[t])
            except:
                continue

        return pd.DataFrame(forecasts).set_index("date")

    def multi_horizon(self, returns: pd.Series, horizons=(1, 5, 22)) -> dict:
        results = {}
        for h in horizons:
            print(f"Forecasting {h}-step ahead ({self.spec})...")
            self.horizon = h
            results[f"h{h}"] = self.run(returns)
        return results


if __name__ == "__main__":
    loader = DataLoader()
    data = loader.download("^GSPC", "2010-01-01", "2024-12-31")
    vix = loader.get_vix("2010-01-01", "2024-12-31")
    returns = data["returns"]
    train, val, test = loader.split(returns)

    diag = DiagnosticsEngine(train)
    report = diag.full_report()

    arima = ARIMAFitter(max_p=3, max_q=3, d=0, ic="bic")
    ic_table = arima.grid_search(train)
    arima_res = arima.fit_best(train)
    resid_diag = arima.residual_diagnostics()
    print(f"Verdict: {resid_diag['verdict']}")

    garch = GARCHFitter()
    all_results = garch.fit_all(train)

    best_std_resid = all_results["GARCH"]["result"].std_resid
    sign_bias = garch.sign_bias_test(best_std_resid.values)
    model = "GJR-GARCH" if sign_bias["use_asymmetric"] else "GARCH"
    print(
        f"Selected model: {model} (leverage={'YES' if sign_bias['use_asymmetric'] else 'NO'})"
    )

    models_to_compare = ["GARCH", "EGARCH", "GJR-GARCH"]
    all_forecasts = {}
    for spec in models_to_compare:
        fc = RollingForecaster(model_spec=spec, dist="t", horizon=1)
        all_forecasts[spec] = fc.run(returns, train_size=len(train))

    ev = ForecastEvaluator(vix=vix)
    ev.compare_models(all_forecasts)
    ev.mincer_zarnowitz(all_forecasts)
    ev.diebold_mariano(all_forecasts, benchmark="GARCH")
