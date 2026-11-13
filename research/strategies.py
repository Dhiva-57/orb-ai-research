#!/usr/bin/env python3
"""Phase 2: the strategy plug-in interface. A strategy only answers three questions about ONE session of bars D
(list of bar dicts {date, slot, start, o, h, l, c, v}, oldest first):

    find_trigger(tm, D, prior)        -> index k of the FIRST bar that triggers, or None
    levels(tm, D, k)                  -> (entry, stop, target) for a long entered at bar k's close, or None
    features(tm, D, k, prior, ctx)    -> {feature: number | None} frozen as of bar k's close

Everything else is shared and identical for every strategy, so no strategy can leak or drift from live behaviour:

    label_candidate  (history)  triple-barrier label: 1 target first, 0 stop first (a bar touching both = STOP), 2 timeout
    live_candidate   (the bot)  same row WITHOUT a label, and only if bar k is the latest completed bar
    check_contract   (tests)    leakage + live/backtest equivalence checks every strategy must pass

Contract (check_contract enforces it): the first trigger found on the whole day is found identically on D[:k+1]; nothing
triggers on D[:k]; levels and features computed on D[:k+1] equal those on the whole day (they may read only bars <= k and
COMPLETED earlier sessions); stop < entry < target; live row == historical row (minus label fields).
Long-only, 5-minute bars (bar_seconds = 300) for now. Standard library only (plus orb_dataset's helpers).
"""

import math

import orb_dataset as o

REGISTRY = {}


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


def names():
    return sorted(REGISTRY)


def get(name, bar_seconds=300, **params):
    if name not in REGISTRY:
        raise SystemExit("unknown strategy %r (known: %s)" % (name, ", ".join(names())))
    return REGISTRY[name](bar_seconds=bar_seconds, **params)


def scale(bars_at_5min, bar_seconds):
    """A count of 5-minute bars expressed in `bar_seconds` bars (same wall-clock duration), at least 1."""
    return max(1, round(bars_at_5min * 300.0 / bar_seconds))


def triple_barrier(D, k, entry, stop, target, last):
    """(label, exit_slot, R). Reads bars AFTER k only. Same rule as orb_dataset.candidate."""
    risk = entry - stop
    label, exit_slot = 2, last
    for i in range(k + 1, last + 1):
        b = D[i]
        if b["l"] <= stop:
            label, exit_slot = 0, i
            break
        if b["h"] >= target:
            label, exit_slot = 1, i
            break
    tr = (target - entry) / risk
    R = tr if label == 1 else -1.0 if label == 0 else (D[last]["c"] - entry) / risk
    return label, exit_slot, R


class Strategy:
    name = "base"
    bar_seconds = 300
    history_sessions = 25        # completed earlier sessions the bot must load for this strategy (backtest uses all)
    warmup_reason = "warming_up"

    def __init__(self, last=None, latest_entry=None, bar_seconds=300, **params):
        self.bar_seconds = bar_seconds
        self.last = last if last is not None else scale(76, bar_seconds)                 # 15:50 New York
        self.latest_entry = latest_entry if latest_entry is not None else scale(60, bar_seconds)   # 14:30
        self.params = params

    # ---- the three things a strategy implements
    def min_bars(self):
        return 1

    def find_trigger(self, tm, D, prior):
        raise NotImplementedError

    def levels(self, tm, D, k):
        raise NotImplementedError

    def features(self, tm, D, k, prior, context):
        raise NotImplementedError

    # ---- shared machinery
    def _row(self, tm, date, D, k, prior, context):
        lv = self.levels(tm, D, k)
        if lv is None:
            return None, "bad_stop"
        entry, stop, target = lv
        if not (stop < entry < target):
            return None, "bad_stop"
        row = {"date": date, "entry_time": D[k]["start"], "entry_slot": k, "entry": entry, "stop": stop,
               "target": target, "risk": entry - stop, "strategy": self.name}
        row.update(self.features(tm, D, k, prior, context))
        return row, "trigger"

    def label_candidate(self, tm, date, D, prior, context=None):
        """Historical row with its triple-barrier label, or None."""
        if len(D) < self.min_bars():
            return None
        k = self.find_trigger(tm, D, prior)
        if k is None:
            return None
        row, _ = self._row(tm, date, D, k, prior, context)
        if row is None:
            return None
        row["label"], row["exit_slot"], row["R"] = triple_barrier(D, k, row["entry"], row["stop"], row["target"], self.last)
        return row

    def live_candidate(self, tm, symbol, date, D, prior, context=None):
        """(row | None, reason) from the COMPLETED bars so far. Only a trigger on the latest bar is tradable."""
        if len(D) < self.min_bars():
            return None, self.warmup_reason
        k = self.find_trigger(tm, D, prior)
        if k is None:
            return None, "no_trigger"
        if k != len(D) - 1:
            return None, "missed_trigger"
        row, why = self._row(tm, date, D, k, prior, context)
        if row is not None:
            row["symbol"] = symbol
        return row, why

    def build(self, tm, sessions, context=None, symbol="?"):
        rows = []
        for i, (d, D) in enumerate(sessions):
            r = self.label_candidate(tm, d, D, sessions[:i], context)
            if r:
                r["symbol"] = symbol
                rows.append(r)
        return rows

    def load_dir(self, tm, directory, context=("SPY", "QQQ")):
        """Like orb_dataset.load_dir but through this strategy. -> (rows, info)."""
        import os
        files = {n.split(".")[0].upper(): os.path.join(directory, n) for n in sorted(os.listdir(directory))
                 if n.endswith((".json", ".json.gz"))}
        if not files:
            raise SystemExit("no .json / .json.gz bar files in %s" % directory)
        ctx_syms = [c.upper() for c in context if c.upper() in files]
        ctx = {c: {d: D for d, D in o.load_sessions(tm, files[c], c)} for c in ctx_syms}
        rows, per = [], {}
        for sym, path in files.items():
            if sym in ctx_syms:
                continue
            sessions = o.load_sessions(tm, path, sym)
            got = self.build(tm, sessions, ctx or None, sym)
            per[sym] = {"sessions": len(sessions), "candidates": len(got)}
            rows += got
        return rows, {"symbols": per, "context": ctx_syms}


@register
class ORB(Strategy):
    """Long-only Opening Range Breakout: first 5-minute close above the opening range high."""
    name = "orb"
    warmup_reason = "opening_range_not_complete"

    def __init__(self, or_bars=None, target_r=1.0, stop_rule="or_mid", last=None, latest_entry=None, bar_seconds=300, **kw):
        or_bars = or_bars if or_bars is not None else scale(3, bar_seconds)       # 15 minutes
        super().__init__(last=last, latest_entry=latest_entry, bar_seconds=bar_seconds, or_bars=or_bars, target_r=target_r,
                         stop_rule=stop_rule, **kw)
        self.or_bars, self.target_r, self.stop_rule = or_bars, target_r, stop_rule

    def min_bars(self):
        return self.or_bars + 1

    def find_trigger(self, tm, D, prior):
        oh, _ = tm.opening_range(D, self.or_bars)
        top = min(self.latest_entry, len(D) - 1)
        return next((i for i in range(self.or_bars, top + 1) if D[i]["c"] > oh), None)

    def levels(self, tm, D, k):
        oh, ol = tm.opening_range(D, self.or_bars)
        e = round(D[k]["c"], 2)
        stop = tm.rule_stop({"stop_rule": self.stop_rule, "stop_pct": 1.0}, oh, ol, e, True)
        if e - stop <= 0:
            return None
        return e, stop, round(e + self.target_r * (e - stop), 2)

    def features(self, tm, D, k, prior, context):
        oh, ol = tm.opening_range(D, self.or_bars)
        f = o.features_at(D, k, oh, ol, prior, self.bar_seconds)
        e, stop, _target = self.levels(tm, D, k)
        f["risk_pct"] = (e - stop) / e * 100.0
        f.update(o.context_features(context, D[k]["date"], k))
        return f


def _same(a, b):
    if a is None or b is None:
        return a is b
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-9 * max(1.0, abs(a))
    return a == b


def check_contract(strategy, tm, sessions, context=None, max_sessions=None):
    """Run the leakage and live-equivalence checks over real or synthetic sessions [(date, D)]. -> list of problems
    (empty = the strategy honours the contract) and how many sessions produced a candidate: (problems, triggered)."""
    problems, triggered = [], 0
    for i, (date, D) in enumerate(sessions[:max_sessions]):
        prior = sessions[:i]
        k = strategy.find_trigger(tm, D, prior) if len(D) >= strategy.min_bars() else None
        if k is None:
            continue
        triggered += 1
        tag = "%s %s k=%d" % (strategy.name, date, k)
        cut = D[:k + 1]
        if strategy.find_trigger(tm, cut, prior) != k:
            problems.append("%s: trigger changes when future bars are removed" % tag)
        if k > 0 and len(D[:k]) >= strategy.min_bars() and strategy.find_trigger(tm, D[:k], prior) is not None:
            problems.append("%s: triggers earlier on truncated data (not the FIRST trigger)" % tag)
        lv_full, lv_cut = strategy.levels(tm, D, k), strategy.levels(tm, cut, k)
        if lv_full != lv_cut:
            problems.append("%s: levels depend on bars after k" % tag)
        if lv_full is None or not (lv_full[1] < lv_full[0] < lv_full[2]):
            problems.append("%s: need stop < entry < target, got %s" % (tag, lv_full))
        f_full, f_cut = strategy.features(tm, D, k, prior, context), strategy.features(tm, cut, k, prior, context)
        for key in set(f_full) | set(f_cut):
            if not _same(f_full.get(key), f_cut.get(key)):
                problems.append("%s: feature %s depends on bars after k (%r vs %r)" % (tag, key, f_full.get(key), f_cut.get(key)))
        hist = strategy.label_candidate(tm, date, D, prior, context)
        live, why = strategy.live_candidate(tm, "X", date, cut, prior, context)
        if hist is None or live is None:
            problems.append("%s: historical row %s / live row %s (%s)" % (tag, hist is not None, live is not None, why))
            continue
        if hist["label"] not in (0, 1, 2):
            problems.append("%s: bad label %r" % (tag, hist["label"]))
        for key in live:
            if key not in ("symbol",) and not _same(live[key], hist.get(key)):
                problems.append("%s: live and historical rows differ on %s (%r vs %r)" % (tag, key, live[key], hist.get(key)))
        if len(D) > k + 1 and strategy.live_candidate(tm, "X", date, D, prior, context)[1] != "missed_trigger" and k != len(D) - 1:
            problems.append("%s: a trigger that is not on the latest bar must be 'missed_trigger'" % tag)
    return problems, triggered


import strategies_more  # noqa: E402,F401  (registers momentum, pullback, vwap, meanrev)
import strategies_daily  # noqa: E402,F401  (registers orb_daily, momentum_daily, ...)
