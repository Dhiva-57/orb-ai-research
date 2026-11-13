# ORB + AI meta-label research kit

Long-only 15-minute Opening Range Breakout; an AI filter only decides take/skip. Everything here is research code:
it places no orders and holds no keys.

## Setup
    git clone https://github.com/Jakeschincariol/day-trader-skill   # next to this folder, or set DAY_TRADER_DIR
    pip install numpy scikit-learn
    python3 -m unittest discover -s tests            # 58 tests

## Run on real data (on YOUR machine)
    export ALPACA_API_KEY=...  ALPACA_SECRET_KEY=...   # free plan = iex feed; keys never as arguments
    python3 research/alpaca_bars.py --symbols SPY QQQ AAPL MSFT NVDA AMD TSLA META AMZN GOOGL --start 2024-11-01 --out research/bars
    python3 research/walkforward.py --bars-dir research/bars          # SPY/QQQ are used as market context only
    python3 research/walkforward.py --synthetic                       # harness self-check

Files: orb_dataset (labels, features, splits) | meta_models (models, calibration, bootstrap) | backtest (risk engine,
pessimistic fills, gate, ledger) | walkforward (rolling windows + gate) | alpaca_bars (read-only downloader).

## Verdicts
The only thing the gate can say is "PASS: send to human review" or "FAIL". Needs >=3 windows, >=30 out-of-sample AI trades,
positive bootstrap lower bound, positive expectancy and Sharpe, and AI ahead of ORB-only in >=60% of windows.

## Not verified
Nothing was run against live Alpaca or Yahoo (the build sandbox blocks them); the downloader is tested against a local fake
server. With ~33 trades (one ticker, 60 days) the harness correctly refuses to fit anything: you need many tickers and a year+.
`fix-label-leakage.patch` fixes label leakage in stock-ai-predictor's `train_ticker` (git apply --check passes). Not applied.

---
# Phase 1 (paper bot core) - added
New files in research/: `broker.py` (Alpaca PAPER only + an in-memory FakeBroker), `journal.py` (write-ahead order journal,
atomic state), `trade_manager.py` (reconcile + exit rules, bar-size agnostic), `model_store.py` (fit/save/load the filter,
sha256 + ledger link), `paper_bot.py` (the tick loop). Tests: `tests/test_paper_bot.py`, `tests/test_model_store.py`
(97 tests total).

## Order of operations on YOUR machine
    export ALPACA_API_KEY=... ALPACA_SECRET_KEY=...            # PAPER-account keys, environment only
    python3 research/alpaca_bars.py --symbols SPY QQQ AAPL MSFT NVDA AMD TSLA META AMZN GOOGL --start 2024-11-01 --out research/bars
    python3 research/walkforward.py --bars-dir research/bars    # must say PASS on REAL data, else stop here
    python3 research/model_store.py --bars-dir research/bars    # links the model to that walk-forward ledger entry
    python3 research/paper_bot.py tick                          # DRY-RUN every 5 min, 9:35-15:55 New York (~19:05-01:25 IST)
    python3 research/paper_bot.py tick --paper-orders           # only after weeks of dry-run review (research/out/bot/decisions.jsonl)
Refresh bars (`alpaca_bars.py --force`) and re-run walk-forward + model_store regularly; a PASS belongs to ONE model.
Schedule a tick shortly AFTER each 5-minute bar closes (e.g. :00:30, :05:30 ...) and one at 15:56 New York.

## Safety properties (each has a test)
paper host hard-wired | dry-run by default | orders need a real-data PASS tied to this model | intent saved to disk before
sending | orders never retried; unclear ones resolved by client_order_id | bot only touches positions it opened |
flatten at close-5min (early-close days: no entries) | stale data / missed trigger / price drift = no trade | kill file.

## Known gaps
Backtest enters at the trigger bar close + 1 bp; bot sends a market order after that bar (paper fills are kinder than real).
Alpaca order fields are sent as strings per its docs; unconfirmed against the live service. start-of-day equity is the equity at
the day's first tick. Model files are pickles: load only your own.

---
# Phase 2 (strategy plug-ins) - added
`research/strategies.py`: a strategy implements only `find_trigger`, `levels`, `features` (see the file's docstring). Labelling
(triple barrier, same-bar = stop), live candidates (latest-bar trigger only) and dataset building are shared, so the bot and the
backtest cannot disagree. ORB now runs through it and is verified row-for-row against the original `orb_dataset.candidate`.

## Adding a strategy
1. Subclass `strategies.Strategy`, decorate with `@register`, set `name`; implement the three methods (read only bars <= k and
   COMPLETED earlier sessions).
2. In a test: `problems, n = strategies.check_contract(MyStrategy(), tm, sessions)` must give `[]` (it catches future-peeking
   triggers, levels and features, and live-vs-history differences; `tests/test_strategies.py` proves it catches a cheater).
3. `python3 research/walkforward.py --bars-dir research/bars --strategy NAME`, then `model_store.py --strategy NAME`,
   then `paper_bot.py tick --strategy NAME`. Ledger entries and models carry the strategy name: one strategy's PASS can never
   authorise another's orders (old ledger entries count as `orb`).
Limits: long-only, 5-minute bars (the bot refuses strategies with another `bar_seconds`).

---
# Phase 3 (more strategies) - added
`research/strategies_more.py`: `momentum`, `pullback`, `vwap`, `meanrev` (rules in the file's docstring). All long-only, 5-minute,
first valid trigger per day, same shared feature set. A trigger only counts if its stop/target are valid (found by the contract
checker: otherwise a day's one trigger could be burnt on an untradable setup).

## Run any of them on your real data
    python3 research/walkforward.py --bars-dir research/bars --strategy momentum     # then pullback, vwap, meanrev
    python3 research/model_store.py --bars-dir research/bars --strategy momentum
    python3 research/paper_bot.py tick --strategy momentum                          # dry-run first
Messages that say "ORB-only" mean "this strategy without the AI filter". Each strategy has its own ledger entry, model and gate.

## Honest expectations
No strategy here has been shown to have an edge. On random-walk data every one FAILS the gate (tested), which proves the gate can
say no. On real data expect most of them to fail too; a PASS only means a human should look. Run several strategies in the SAME
account only after each has weeks of dry-run evidence, and remember that testing five strategies makes one lucky PASS likelier:
trust a PASS more if it repeats on later data or on tickers you did not tune on.

---
# Phase 4 (comparison + router) - added
**Trade log.** Every finished paper trade is appended to `research/out/bot/trades.jsonl` (planned entry/stop/target, fill, exit price,
exit reason, strategy). It survives the daily state reset. Exit price/reason come from the filled bracket leg when Alpaca shows it
(`legs` on the order; unconfirmed against the live service), else the last price before the bot's own close, else unknown.

**Paper vs backtest.** After some days of paper trading, refresh the bars and compare:
    python3 research/alpaca_bars.py --symbols ... --start ... --out research/bars --force
    python3 research/compare.py --out research/out/bot --bars-dir research/bars --min-trades 30
It lines every paper trade up with the backtest's version of the same day (signal parity, entry slippage in bps, paper R vs
backtest R, exit reason), lists the historical candidates the bot did not trade and why (ai_veto, risk_veto, stale_data,
missed_trigger...), and prints CANDIDATE explanations for gaps. Below --min-trades it says "too early" and draws no conclusion.
R is gross on both sides. Differences caused by feed (live IEX bar vs the consolidated historical bar) show up as signal mismatches.

**Router (several strategies, one account).** Give each strategy its own model dir, then run them together:
    python3 research/walkforward.py --bars-dir research/bars --strategy momentum
    python3 research/model_store.py --bars-dir research/bars --strategy momentum --out research/out/models/momentum   # repeat per strategy
    python3 research/paper_bot.py tick --strategies orb momentum vwap [--paper-orders]
Each strategy proposes, its own model must agree, survivors rank by EV = p*reward_R - (1-p), the risk engine takes the first it
accepts. With --paper-orders only strategies whose OWN real-data PASS is linked to their OWN model may order; the others are
logged as `shadow_candidate`. One position per symbol per day. This is arbitration, not regime detection.

---
# Phase 5 (Stock AI Predictor) - added
Read `PREDICTOR_AUDIT.md`. Short version: the predictor leaked the answer (100% accuracy on random walks) and predicted absolute prices
(forecasts of -20..-44% on trending series). `predictor-fixes.patch` fixes both; `research/check_predictor.py --repo <checkout>` tells
LEAKAGE/CLEAN. `research/daily_signal.py` is the leak-proof test of whether daily direction is predictable on your tickers, and
`orb_daily`, `momentum_daily`, `pullback_daily`, `vwap_daily`, `meanrev_daily` add its probability as ONE extra feature (`ctx_daily_p_up`),
judged by the usual walk-forward gate. The predictor itself is not connected to the bot.

---
# Phase 6 (sub-minute bars) - added
**What exists.** Every strategy takes `bar_seconds` (`strategies.get("orb", 30)`): time-based parameters keep their wall-clock
meaning (15-minute opening range, 14:30 last entry, 60-minute ATR...), the risk engine counts slots of the same size, and
trigger scans are linear (a full 5-second day = 4,680 bars scans in well under a second). Gates, ledger entries and models carry the
bar size: a 5-minute PASS can never authorise a 30-second strategy.
`tickbars.py` builds bars from trades (causal, completed bars only, late trades dropped never repainted, flat-fill gaps),
records them to `<dir>/<SYMBOL>/<date>.csv.gz`, and loads them for backtests. `stream.py` is the Alpaca trade websocket (read-only).
`live_runner.py` = record, or run the bot on live tick bars (throttled ticks, same safety rules as the 5-minute bot).

**The honest part - read before planning.**
1. Alpaca's free history stops at 1-minute bars. 5 s / 30 s strategies can ONLY be backtested on trades YOU record:
       python3 research/live_runner.py record --symbols AAPL MSFT NVDA ... --ticks-dir research/ticks     # every session
2. The walk-forward gate wants ~223 candidate days per window (the run on 110 recorded days said "not enough data"):
   plan on roughly a year of recording before a verdict. Until then fast strategies can only run as dry-run.
3. Costs dominate at short horizons. The backtest charges 1 bp entry + 3 bp stop slippage per trade, tiny stops make that a large
   share of R, and real sub-minute fills are worse than paper fills. Expect most fast strategies to FAIL; that is the gate working.
4. Pattern meaning changes with bar size ("green bar", "previous bar's high" on 5-second bars is closer to noise). Scaling keeps
   durations, not semantics. No evidence of any edge at any bar size exists in this project.
5. IEX trades are a slice of the market; bars built from them differ from Alpaca's consolidated bars. Do not mix them in one test.
6. Free plans typically allow ONE websocket connection: do not run `record` and `run` as two processes (run records too).
7. NOT tested against the live service: the websocket connection, reconnects, threading in `live()`. TESTED: parsing, back-off,
   watchdog, aggregation, recording/loading, and a full replay of raw trades -> order -> close with the fake broker.
   Test the stream with `record` alone for a few sessions first.
Dry-run `run` (no --paper-orders) logs what it would do. Market-context features (SPY/QQQ) are not used for tick bars.
