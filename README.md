Main Point

1. Built a complete ARIMA-GARCH volatility forecasting engine for S&P 500 conducted full diagnostic pipeline (ADF/KPSS stationarity, Ljung-Box serial correlation, Engle's ARCH-LM test) confirming ARCH effects at p < 0.001 and excess kurtosis of 7.2 justifying Student-t error distribution

2. Implemented and compared GARCH(1,1), EGARCH(1,1), and GJR-GARCH(1,1) with Student-t innovations using the arch library; selected GJR-GARCH via Engle-Ng sign bias test confirming significant leverage effect — negative shocks increasing volatility 40% more than positive shocks of equal magnitude.

3. Conducted expanding-window rolling forecast evaluation; GJR-GARCH(1,1) outperformed benchmark GARCH(1,1) on QLIKE loss (0.042 vs. 0.068) and achieved Mincer-Zarnowitz R² of 0.41, with Diebold-Mariano test confirming statistically significant improvement

4. Verified GARCH(1,1) volatility persistence of α + β = 0.97 with half-life of 23 days; computed news impact curves demonstrating asymmetric response to negative shocks; estimated unconditional annual volatility of 16.8%.
