#!/usr/bin/env python3
"""Sub-minute live loop: trades -> TickBars -> the SAME bot tick (paper_bot.tick) -> Alpaca PAPER orders.

    python3 research/live_runner.py record --symbols AAPL MSFT --ticks-dir research/ticks        # record only, no orders
    python3 research/live_runner.py run --symbols AAPL MSFT --strategy orb --bar-seconds 30 \\
            --ticks-dir research/ticks [--paper-orders]                                           # dry-run unless flagged

Step 1 is ALWAYS `record`: Alpaca's free history stops at 1-minute bars, so 5 s / 30 s strategies can only be backtested on
trades you recorded. Weeks later: walkforward.py --ticks-dir ... --bar-seconds 30 --strategy orb, then model_store.py with the
same flags, then `run`. Everything the 5-minute bot guarantees still holds (paper host only, dry-run default, gate tied to
strategy + bar size + model, write-ahead journal, own positions only, flatten at close - 5 min).

Runner (pure, deterministic, TESTED by replaying recorded trades): feeds trades to one TickBars per symbol; calls
paper_bot.tick when a bar completed (at most every `min_tick_s`) or every `manage_every_s` to manage positions.
The websocket + threading glue in `live()` is NOT tested against the live service.
"""

import argparse
import datetime
import os
import sys
import threading
import time

import orb_dataset as o
import paper_bot as pb
import strategies as ss
import tickbars as tb


class Runner:
    def __init__(self, broker, tm, bcfg, model, meta, history, router=None, paper_orders=False,
                 min_tick_s=15.0, manage_every_s=60.0, lateness_s=1.0):
        strat = (router.options[0].strategy if router else bcfg.strategy)
        self.broker, self.tm, self.bcfg, self.model, self.meta = broker, tm, bcfg, model, meta
        self.history, self.router, self.paper_orders = history, router, paper_orders
        self.min_tick_s, self.manage_every_s = min_tick_s, manage_every_s
        self.books = {s: tb.TickBars(tm, strat.bar_seconds, lateness_s) for s in bcfg.symbols}
        self.last_tick, self.dirty, self.results = -1e18, False, []
        self.lock = threading.RLock()

    def on_trade(self, sym, ts, price, size):
        book = self.books.get(sym.upper())
        if book is not None:
            with self.lock:
                if book.add(ts, price, size):
                    self.dirty = True

    def _source(self, sym, now):
        return self.books[sym].completed_bars(now)

    def maybe_tick(self, now_epoch, force=False):
        with self.lock:
            for book in self.books.values():
                if book.advance(now_epoch):
                    self.dirty = True
            due = force or self.dirty or now_epoch - self.last_tick >= self.manage_every_s
            if not due or now_epoch - self.last_tick < self.min_tick_s:
                return None
            now_utc = datetime.datetime.utcfromtimestamp(now_epoch).replace(microsecond=0)
            out = pb.tick(self.broker, self.tm, self.bcfg, self.model, self.meta, now_utc, self.paper_orders, self.history,
                          self.router, bars_source=lambda s: self._source(s, now_epoch))
            self.last_tick, self.dirty = now_epoch, False
            self.results.append(out)
            return out


def replay(runner, trades):
    """Deterministic replay of recorded trades [(symbol, epoch, price, size)] in time order. Returns the tick results."""
    last = None
    for sym, ts, px, sz in sorted(trades, key=lambda t: t[1]):
        runner.on_trade(sym, ts, px, sz)
        runner.maybe_tick(ts)
        last = ts
    if last is not None:
        runner.maybe_tick(last + 1.0)
    return runner.results


# ---------------------------------------------------------------- live glue (not tested against the real service)

def live(a):
    import broker as bk
    import model_store as ms
    import stream
    key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        print("set ALPACA_API_KEY and ALPACA_SECRET_KEY in the environment")
        return 2
    tm = o.load_pack()
    symbols = [s.upper() for s in a.symbols]
    rec = tb.TickRecorder(tm, a.ticks_dir) if a.ticks_dir else None
    runner = None
    if a.command == "run":
        bcfg = pb.BotConfig(symbols, out_dir=a.out, model_dir=a.model, ledger=a.ledger, context=(),
                            strategy=ss.get(a.strategy, a.bar_seconds))
        model, meta = None, {}
        if not a.no_model:
            try:
                model, meta = ms.load(a.model)
            except (OSError, ms.ModelError) as e:
                print("no usable model (%s): strategy-only DRY-RUN" % e)
        if a.paper_orders and model is None:
            print("--paper-orders needs a model; refusing")
            return 2
        hist = {s: tb.load_sessions(tm, a.ticks_dir, s, a.bar_seconds)[-bcfg.strategy.history_sessions:] for s in symbols}
        runner = Runner(bk.AlpacaPaperBroker(), tm, bcfg, model, meta, hist, None, a.paper_orders)

    def on_trade(sym, ts, px, sz):
        if rec:
            rec.add(sym, ts, px, sz)
        if runner:
            runner.on_trade(sym, ts, px, sz)

    st = stream.TradeStream(key, secret, symbols, on_trade, lambda kind, detail: print("[stream]", kind, detail, flush=True))
    th = threading.Thread(target=st.run, daemon=True)
    th.start()
    try:
        while True:
            now = time.time()
            if rec:
                rec.flush()
            if runner:
                runner.maybe_tick(now)
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("stopping")
    finally:
        st.stop = True
        if rec:
            rec.close()
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Record trades / run the sub-minute PAPER bot.")
    p.add_argument("command", choices=["record", "run"])
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--ticks-dir", default="research/ticks")
    p.add_argument("--strategy", default="orb", choices=ss.names())
    p.add_argument("--bar-seconds", type=int, default=30)
    p.add_argument("--model", default="research/out/model")
    p.add_argument("--ledger", default="research/out/ledger.jsonl")
    p.add_argument("--out", default="research/out/bot")
    p.add_argument("--paper-orders", action="store_true")
    p.add_argument("--no-model", action="store_true")
    return live(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
