#!/usr/bin/env python3
"""Trades -> N-second bars (5 s, 30 s, 60 s ...), plus a recorder and loader so you can BUILD your own sub-minute history.

Why it exists: Alpaca's free historical data gives 1-minute bars at the finest, so 5 s / 30 s strategies can only be
backtested on trades you recorded yourself. Workflow: `live_runner.py record` for weeks -> `walkforward.py --ticks-dir` ->
only then trade.

TickBars (one symbol, one bar size, regular session 09:30-16:00 New York):
  * causal and no repainting: a bar is emitted only once its window has ended (watermark - lateness); a trade that arrives
    after its bar was emitted is DROPPED and counted (late_dropped), never used to rewrite history
  * the clock can complete bars for quiet symbols (advance / completed_bars(now))
  * interior gaps are filled flat with zero volume, exactly like the 5-minute pipeline; no opening bar -> no data
  * bar dicts use the same schema as the rest of the project: {date, slot, start, o, h, l, c, v}
Bars built from IEX trades differ from Alpaca's consolidated bars: do not mix the two in one backtest.
Recorded files: <dir>/<SYMBOL>/<YYYY-MM-DD>.csv.gz with lines "epoch_seconds,price,size" (gzip members can be appended).
"""

import calendar
import datetime
import gzip
import os
import re

SESSION_SECONDS = 390 * 60
_TS = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(\.\d+)?Z?$")


def parse_rfc3339(text):
    """'2026-03-02T14:30:00.123456789Z' -> epoch seconds (float). Raises ValueError on anything else."""
    m = _TS.match(text)
    if not m:
        raise ValueError("bad timestamp %r" % text)
    base = calendar.timegm(datetime.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").timetuple())
    return base + (float(m.group(2)) if m.group(2) else 0.0)


class TickBars:
    def __init__(self, tm, bar_seconds, lateness_s=1.0, max_gap_bars=None):
        if bar_seconds <= 0 or SESSION_SECONDS % bar_seconds:
            raise ValueError("bar_seconds must divide the 23400-second session (got %r)" % bar_seconds)
        self.tm, self.bar_seconds, self.lateness = tm, bar_seconds, lateness_s
        self.max_gap = max_gap_bars if max_gap_bars is not None else max(6, 1800 // bar_seconds)   # ~30 min of nothing
        self.date = None
        self.open_epoch = self.close_epoch = None
        self.buckets, self.emitted, self.watermark = {}, 0, 0.0
        self.late_dropped = self.out_of_session = 0
        self._seq = 0

    # -------------------------------------------------------------- session bookkeeping
    def _roll(self, ts):
        local = self.tm.to_eastern(datetime.datetime.utcfromtimestamp(ts))
        d = local.date()
        o = self.tm.eastern_to_utc(datetime.datetime(d.year, d.month, d.day, 9, 30))
        self.date = d.isoformat()
        self.open_epoch = calendar.timegm(o.timetuple())
        self.close_epoch = self.open_epoch + SESSION_SECONDS
        self.buckets, self.emitted, self.watermark = {}, 0, 0.0

    def _cutoff_slot(self, clock):
        """Slots strictly below this are complete at `clock` (epoch seconds)."""
        n = int((clock - self.lateness - self.open_epoch) // self.bar_seconds)
        return max(0, min(n, SESSION_SECONDS // self.bar_seconds))

    # -------------------------------------------------------------- input
    def add(self, ts, price, size):
        """Feed one trade. Returns the number of bars that became complete because of it."""
        if price is None or price <= 0 or not size or size <= 0:
            return 0
        if self.open_epoch is None or not (self.open_epoch <= ts < self.close_epoch):
            d = self.tm.to_eastern(datetime.datetime.utcfromtimestamp(ts)).date().isoformat()
            if self.date is None or d > self.date:                  # only ever roll FORWARD: a stray old trade cannot wipe today
                self._roll(ts)
            if not (self.open_epoch <= ts < self.close_epoch):
                self.out_of_session += 1
                return 0
        slot = int((ts - self.open_epoch) // self.bar_seconds)
        if slot < self.emitted:
            self.late_dropped += 1                                   # its bar is already out: no repainting
            return 0
        self._seq += 1
        b = self.buckets.get(slot)
        if b is None:
            self.buckets[slot] = {"o": price, "h": price, "l": price, "c": price, "v": float(size),
                                  "t_o": (ts, self._seq), "t_c": (ts, self._seq)}
        else:
            b["h"], b["l"], b["v"] = max(b["h"], price), min(b["l"], price), b["v"] + size
            if (ts, self._seq) < b["t_o"]:
                b["o"], b["t_o"] = price, (ts, self._seq)
            if (ts, self._seq) > b["t_c"]:
                b["c"], b["t_c"] = price, (ts, self._seq)
        before = self.emitted
        self.watermark = max(self.watermark, ts)
        self.emitted = max(self.emitted, self._cutoff_slot(self.watermark))
        return self.emitted - before

    def advance(self, now_epoch):
        """Let the wall clock complete bars (quiet symbols). Returns how many became complete."""
        if self.open_epoch is None or not (self.open_epoch <= now_epoch < self.close_epoch + 60):
            return 0
        before = self.emitted
        self.emitted = max(self.emitted, self._cutoff_slot(now_epoch))
        return self.emitted - before

    # -------------------------------------------------------------- output
    def completed_bars(self, now_epoch=None):
        """-> (bars, last_real_slot, why). Completed bars only, flat-filled between real ones."""
        if now_epoch is not None:
            self.advance(now_epoch)
        done = [s for s in self.buckets if s < self.emitted]
        if not done or 0 not in done:
            return [], (max(done) if done else None), "no_opening_bar"
        last = max(done)                                     # last slot with a real trade
        end = self.emitted - 1                               # last COMPLETE slot: the clock fills quiet slots up to here, so a
        missing = end + 1 - len(done)                        # breakout bar that is no longer the latest reads as 'missed'
        if missing > self.max_gap * 6:
            return [], last, "gappy_data"
        out, prev = [], None
        for i in range(end + 1):
            b = self.buckets.get(i)
            if b is not None and i < self.emitted:
                prev = {"o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"], "v": b["v"]}
                bar = dict(prev)
            else:
                bar = {"o": prev["c"], "h": prev["c"], "l": prev["c"], "c": prev["c"], "v": 0.0}
            t = i * self.bar_seconds
            bar.update(date=self.date, slot=i, start="%02d:%02d:%02d" % ((570 + t // 60) // 60, (570 + t // 60) % 60, t % 60))
            out.append(bar)
        return out, last, "ok"


# ---------------------------------------------------------------- recording and loading

class TickRecorder:
    """Appends trades to <dir>/<SYMBOL>/<ET date>.csv.gz. Call flush() every few seconds and close() at the end."""

    def __init__(self, tm, directory):
        self.tm, self.dir, self.files, self.n = tm, directory, {}, 0

    def add(self, symbol, ts, price, size):
        d = self.tm.to_eastern(datetime.datetime.utcfromtimestamp(ts)).date().isoformat()
        key = (symbol, d)
        fh = self.files.get(key)
        if fh is None:
            os.makedirs(os.path.join(self.dir, symbol), exist_ok=True)
            fh = self.files[key] = gzip.open(os.path.join(self.dir, symbol, d + ".csv.gz"), "at")
        fh.write("%.6f,%s,%s\n" % (ts, repr(float(price)), repr(float(size))))
        self.n += 1

    def flush(self):
        for fh in self.files.values():
            fh.flush()

    def close(self):
        for fh in self.files.values():
            fh.close()
        self.files = {}


def read_ticks(path):
    """Yield (epoch, price, size) from one recorded file. A truncated last gzip member (crash) is tolerated."""
    try:
        with gzip.open(path, "rt") as fh:
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) == 3:
                    try:
                        yield float(parts[0]), float(parts[1]), float(parts[2])
                    except ValueError:
                        continue
    except (EOFError, OSError):
        return


def day_bars(tm, path, bar_seconds):
    """One recorded day -> (date, bars) with the session flat-filled to its end, or None if the day is not usable."""
    tb = TickBars(tm, bar_seconds)
    for ts, px, sz in sorted(read_ticks(path)):
        tb.add(ts, px, sz)
    if tb.open_epoch is None:
        return None
    tb.advance(tb.close_epoch + 1)
    bars, last, why = tb.completed_bars()
    total = SESSION_SECONDS // bar_seconds
    if why != "ok" or last is None or last < int(0.95 * total) - 1:
        return None                                              # partial day (recorder started late / stopped early)
    while len(bars) < total:
        p = bars[-1]
        bars.append(dict(p, slot=len(bars), o=p["c"], h=p["c"], l=p["c"], c=p["c"], v=0.0))
    return tb.date, bars


def load_sessions(tm, ticks_dir, symbol, bar_seconds):
    sdir = os.path.join(ticks_dir, symbol)
    if not os.path.isdir(sdir):
        return []
    out = []
    for name in sorted(os.listdir(sdir)):
        if name.endswith(".csv.gz"):
            r = day_bars(tm, os.path.join(sdir, name), bar_seconds)
            if r:
                out.append(r)
    return sorted(out)


def load_rows(strategy, tm, ticks_dir, symbols=None):
    """Like strategy.load_dir, but from recorded ticks at strategy.bar_seconds. No market-context features (SPY/QQQ)
    for sub-minute bars. -> (rows, info)."""
    syms = symbols or sorted(d for d in os.listdir(ticks_dir) if os.path.isdir(os.path.join(ticks_dir, d)))
    rows, per = [], {}
    for s in syms:
        sessions = load_sessions(tm, ticks_dir, s.upper(), strategy.bar_seconds)
        got = strategy.build(tm, sessions, None, s.upper())
        per[s.upper()] = {"sessions": len(sessions), "candidates": len(got)}
        rows += got
    return rows, {"symbols": per, "context": []}
