# Stock AI Predictor audit (repo HEAD 52652f7)

## Decision
1. **As a trading signal: not usable, and its accuracy numbers were wrong.** The model was handed the answer.
2. **Leakage: fixed** (`predictor-fixes.patch`, applies cleanly to HEAD). After the fix it behaves like an honest model: ~50% on data with no pattern.
3. **Replacement for the trading bot:** `research/daily_signal.py`, a leak-proof walk-forward test of whether next-day direction is
   predictable on YOUR tickers, plus optional `*_daily` strategy variants that feed its probability to the AI filter, judged by the
   same walk-forward gate as everything else. The predictor is NOT wired into the bot.
   Reason: even honest, a daily-direction model is a different horizon from the 5-minute strategies, and daily direction on liquid
   stocks is very hard to predict. Nothing here has been run on real tickers (the build sandbox cannot reach market data):
   that verdict is yours to get with `daily_signal.py`.

## Findings (all reproduced on the repo's own code with the real `train_ticker` / `predict_ticker`, on synthetic prices)
| # | Finding | Evidence | Fixed |
|---|---|---|---|
| 1 | **Label leakage.** Row t's features (pct_change, lag1_return, macd, ma5, ema, rsi...) already contain day t's close, and the targets were day t's own close/direction. | Pure random walks: reported direction accuracy 100, 100, 99.3, 100 %. Honest answer is ~50. | yes |
| 2 | **Price regressor predicts absolute price from price-level features.** Trees cannot go outside the price range they saw. | Strongly trending series: next-day forecast -34.6 %, -44.5 %, -20.4 % while reporting 100 % accuracy. | yes: it now predicts the next-day return; the price is last close x (1 + return) |
| 3 | Reported accuracy is one 20% holdout (about 100 rows), so +/-5 points is just noise, and it was shown without a naive rival. | `direction_accuracy_stderr_pts`, `baseline_direction_accuracy`, `edge_vs_baseline_pts`, `naive_price_mae_pct` now added to the metrics. | partly (metrics added; UI untouched) |
| 4 | 7-day forecast chains predictions into fabricated bars (invented high/low, copied volume). Its later days carry no real information. | read from `forecast_ticker` | NO |
| 5 | Direction classifier and price regressor are separate models and can disagree (e.g. "buy" next to DOWN). | read from the code; seen in earlier chat | NO |

After the patch (same code, same data): random walks 51.1 / 46.0 / 59.9 / 51.1 %; trending series next-day forecasts -0.44 %, -0.17 %, +1.01 %
and the 7-day forecast stays under 1.5 % per step.

## How to apply and verify (on your machine)
    cd stock-ai-predictor && git apply ../orb-ai-research/predictor-fixes.patch      # review first; nothing was pushed
    python3 ../orb-ai-research/research/check_predictor.py --repo .                   # expect CLEAN (exit 0); before the patch: LEAKAGE (exit 1)
The patch replaces the earlier `fix-label-leakage.patch` (which is its first part). The app retrains on every prediction, so no saved
model needs migrating.

## Is there any daily edge on your tickers?
    python3 research/daily_signal.py --bars-dir research/bars
Per ticker: walk-forward accuracy and Brier score against the naive rivals, with 95% intervals. EDGE needs both clearly above zero over
500+ out-of-sample days. About 2-3% of edge-free tickers get flagged by luck, so one flag among many tickers is not evidence.
To test it as a trading feature: compare `walkforward.py --strategy orb` with `--strategy orb_daily` (also momentum_daily, ...).
Live use needs ~320 sessions of history; the bot loads them automatically.

## Not done / limits
Not run on real tickers. Findings 4 and 5 are unfixed and not UI-checked (index.html was not changed). I did not audit the data
fetchers, the database layer or `_investment_signal` / `_trade_plan` beyond how they consume the regressor output.
