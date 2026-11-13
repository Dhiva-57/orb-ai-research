# Project phases (resume here in a new chat: upload this ZIP and say "continue from PHASES.md")
Goal: Alpaca PAPER bot, many strategies, multiple timeframes, structured so live trading is possible LATER (not before weeks of
paper evidence + a real-data walk-forward PASS per strategy). Stock AI Predictor: use if sound; else fix leakage (fix-label-leakage.patch); else drop.

| Phase | Scope | Status |
|---|---|---|
| 0 | ORB dataset, meta-models + calibration, risk engine, backtest, walk-forward + gate, ledger, Alpaca downloader | done (58 tests) |
| 1 | Paper bot core: broker layer, write-ahead journal, 5-min trade manager, model store, tick loop | DONE (97 tests, fake broker only) |
| 2 | Strategy plug-in interface (`strategies.py`), ORB ported onto it, contract checker, strategy-specific ledger/model gate | DONE (109 tests) |
| 3 | Strategies momentum, pullback, vwap, meanrev (`strategies_more.py`); contract + end-to-end data-path tests; all FAIL walk-forward on pure noise as they should | DONE (125 tests). Real-data verdicts are YOURS to run |
| 4 | Permanent trade log (`trades.jsonl`) with exit capture, `compare.py` paper-vs-backtest report, `router.py` multi-strategy arbitration, per-strategy order gate | DONE (147 tests, fake broker only) |
| 5 | Predictor audit (`PREDICTOR_AUDIT.md`), `predictor-fixes.patch`, `check_predictor.py`, honest `daily_signal.py`, `*_daily` variants | DONE (163 tests). Verdict on REAL tickers is YOURS to run |
| 6 | Sub-minute bars: `bar_seconds` scaling for every strategy, `tickbars.py` (trades->bars, recorder, loader), `stream.py`, `live_runner.py` (record / run / replay), bar-size-specific gate | DONE (189 tests). Websocket + threading glue NOT tested against the live service |
Live trading: NOT built, by design.

## Where things stand
All six phases are built. Nothing has been run on real market data or against live Alpaca (the build sandbox cannot reach either):
that is the next step, on your machine, in this order: (1) 5-minute pipeline from README.md "Order of operations", dry-run for weeks;
(2) `live_runner.py record` running every session to build sub-minute history; (3) only when you have ~250+ recorded days per
strategy, walk-forward at 30 s / 5 s. Live (real-money) trading is NOT built, by design.
