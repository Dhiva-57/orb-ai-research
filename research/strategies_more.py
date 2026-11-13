#!/usr/bin/env python3
"""Phase 3 strategies (long-only, 5-minute bars). Registered into strategies.REGISTRY on import of `strategies`.

    momentum  new session high, above VWAP and above the day's open            stop = ATR multiple, target = R multiple
    pullback  up-leg, shallow pull-back that holds VWAP, then a resumption bar   stop = swing low of the pull-back
    vwap      3+ closes below VWAP, then a green close back above it             stop = ATR multiple, target = R multiple
    meanrev   close below its 20-bar lower band, then a green reversal bar       stop = below the reversal lows, target = the 20-bar mean

All of them: first trigger of the day only, latest entry slot honoured, every number computed from bars <= the trigger bar
(strategies.check_contract proves it). Features are the shared orb_dataset feature set (same columns, so models/walk-forward
work unchanged); each strategy's model is trained and gated separately. None of these has been shown to have an edge: that is
what the real-data walk-forward is for, and most will probably FAIL it.
"""

import math

import itertools

import orb_dataset as o
import strategies as ss


def vwap_series(D):
    """Causal cumulative VWAP at every bar (typical price; zero-volume filled bars carry no weight)."""
    out, pv, vol = [], 0.0, 0.0
    for b in D:
        tp = (b["h"] + b["l"] + b["c"]) / 3.0
        pv += tp * b["v"]
        vol += b["v"]
        out.append(pv / vol if vol > 0 else b["c"])
    return out


def atr(D, k, n=12):
    """Mean high-low range of the last n bars up to and including k."""
    w = D[max(0, k - n + 1):k + 1]
    return sum(b["h"] - b["l"] for b in w) / len(w)


class _Base(ss.Strategy):
    start = 6

    def __init__(self, stop_atr=1.5, target_r=1.0, atr_n=None, or_bars=None, last=None, latest_entry=None, bar_seconds=300, **kw):
        atr_n = atr_n if atr_n is not None else ss.scale(12, bar_seconds)          # 60 minutes
        or_bars = or_bars if or_bars is not None else ss.scale(3, bar_seconds)     # 15 minutes
        super().__init__(last=last, latest_entry=latest_entry, bar_seconds=bar_seconds, stop_atr=stop_atr,
                         target_r=target_r, atr_n=atr_n, or_bars=or_bars, **kw)
        self.stop_atr, self.target_r, self.atr_n, self.or_bars = stop_atr, target_r, atr_n, or_bars
        self.start = ss.scale(type(self).start, bar_seconds)                      # class attr = the 5-minute count

    def min_bars(self):
        return self.start + 1

    def trigger_at(self, D, i, vw, pre):
        raise NotImplementedError

    def prefix(self, D):
        """Causal running quantities computed ONCE per call so scanning a whole day is linear, not quadratic."""
        return {"ph": list(itertools.accumulate((b["h"] for b in D), max))}       # ph[i] = highest high of D[:i+1]

    def find_trigger(self, tm, D, prior):
        vw = vwap_series(D)                                  # causal: vw[i] uses bars <= i only
        pre = self.prefix(D)
        top = min(self.latest_entry, len(D) - 1)
        # a trigger only counts if its stop/target are valid: otherwise the day's one trigger would be burnt on a setup
        # that cannot be traded. levels() reads bars <= i only.
        return next((i for i in range(self.start, top + 1)
                     if self.trigger_at(D, i, vw, pre) and self.levels(tm, D, i) is not None), None)

    def levels(self, tm, D, k):
        e = round(D[k]["c"], 2)
        a = atr(D, k, self.atr_n)
        if a <= 0:
            return None
        stop = round(e - self.stop_atr * a, 2)
        if stop >= e:
            return None
        return e, stop, round(e + self.target_r * (e - stop), 2)

    def features(self, tm, D, k, prior, context):
        oh, ol = tm.opening_range(D, self.or_bars)
        f = o.features_at(D, k, oh, ol, prior, self.bar_seconds)
        lv = self.levels(tm, D, k)
        f["risk_pct"] = (lv[0] - lv[1]) / lv[0] * 100.0 if lv else None
        f.update(o.context_features(context, D[k]["date"], k))
        return f


@ss.register
class Momentum(_Base):
    name = "momentum"
    start = 6

    def trigger_at(self, D, i, vw, pre):
        return D[i]["c"] > pre["ph"][i - 1] and D[i]["c"] > vw[i] and D[i]["c"] > D[0]["o"]


@ss.register
class VwapReclaim(_Base):
    name = "vwap"
    start = 4

    def __init__(self, below_bars=None, **kw):
        below_bars = below_bars if below_bars is not None else ss.scale(3, kw.get("bar_seconds", 300))
        super().__init__(below_bars=below_bars, **kw)
        self.below_bars = below_bars

    def trigger_at(self, D, i, vw, pre):
        n = self.below_bars
        if i < n:
            return False
        below = all(D[j]["c"] < vw[j] for j in range(i - n, i))
        return below and D[i]["c"] > vw[i] and D[i]["c"] > D[i]["o"]


@ss.register
class Pullback(_Base):
    name = "pullback"
    start = 7

    def __init__(self, min_leg_pct=0.3, min_depth_atr=0.4, max_risk_atr=3.0, **kw):
        super().__init__(min_leg_pct=min_leg_pct, min_depth_atr=min_depth_atr, max_risk_atr=max_risk_atr, **kw)
        self.min_leg_pct, self.min_depth_atr, self.max_risk_atr = min_leg_pct, min_depth_atr, max_risk_atr

    def _state(self, D, i, vw):
        """Reference (slow, O(i)): (index of the session high before i, lowest low since it) or None."""
        hi = max(b["h"] for b in D[:i])
        jh = max(j for j in range(i) if D[j]["h"] == hi)
        if jh > i - 2:
            return None
        return jh, min(b["l"] for b in D[jh + 1:i + 1])

    def prefix(self, D):
        """Linear-time version of _state for every bar: (high before i, its index, lowest low since it, through i)."""
        pre = super().prefix(D)
        hi, jh, run, rows = float("-inf"), -1, float("inf"), []
        for i, b in enumerate(D):
            rows.append((hi, jh, min(run, b["l"])))
            if b["h"] >= hi:
                hi, jh, run = b["h"], i, float("inf")
            else:
                run = min(run, b["l"])
        pre["state"] = rows
        return pre

    def trigger_at(self, D, i, vw, pre):
        hi, jh, swing = pre["state"][i]
        if i < 1 or jh > i - 2:
            return False
        a = atr(D, i, self.atr_n)
        up_leg = hi >= D[0]["o"] * (1 + self.min_leg_pct / 100.0)
        deep_enough = hi - swing >= self.min_depth_atr * a
        holds_vwap = swing > vw[i] * 0.998
        resumes = D[i]["c"] > D[i - 1]["h"] and D[i]["c"] > D[i]["o"]
        return up_leg and deep_enough and holds_vwap and resumes

    def levels(self, tm, D, k):
        W = D[:k + 1]
        vw = vwap_series(W)
        st = self._state(W, k, vw)
        if st is None:
            return None
        e = round(D[k]["c"], 2)
        stop = round(st[1] - 0.01, 2)
        a = atr(W, k, self.atr_n)
        if stop >= e or a <= 0 or (e - stop) > self.max_risk_atr * a:
            return None
        return e, stop, round(e + self.target_r * (e - stop), 2)


@ss.register
class MeanReversion(_Base):
    name = "meanrev"
    start = 21

    def __init__(self, window=None, z=2.0, **kw):
        window = window if window is not None else ss.scale(20, kw.get("bar_seconds", 300))
        super().__init__(window=window, z=z, **kw)
        self.window, self.z = window, z
        self.start = window + 1

    def prefix(self, D):
        pre = super().prefix(D)
        c0 = D[0]["c"]                                         # centred sums: no catastrophic cancellation
        pre["cs"] = [0.0] + list(itertools.accumulate(b["c"] - c0 for b in D))
        pre["cs2"] = [0.0] + list(itertools.accumulate((b["c"] - c0) ** 2 for b in D))
        pre["c0"] = c0
        return pre

    def _band(self, D, j, pre=None):
        n = self.window
        if pre is None:
            w = [b["c"] for b in D[j - n + 1:j + 1]]
            m = sum(w) / len(w)
            return m, math.sqrt(sum((x - m) ** 2 for x in w) / len(w))
        s1, s2 = pre["cs"][j + 1] - pre["cs"][j + 1 - n], pre["cs2"][j + 1] - pre["cs2"][j + 1 - n]
        mu = s1 / n
        return mu + pre["c0"], math.sqrt(max(s2 / n - mu * mu, 0.0))

    def trigger_at(self, D, i, vw, pre):
        m, s = self._band(D, i - 1, pre)
        oversold = s > 1e-9 and D[i - 1]["c"] < m - self.z * s
        return oversold and D[i]["c"] > D[i]["o"] and D[i]["c"] > D[i - 1]["c"]

    def levels(self, tm, D, k):
        m, _s = self._band(D, k - 1)
        e = round(D[k]["c"], 2)
        stop = round(min(D[k - 1]["l"], D[k]["l"]) - 0.01, 2)
        target = round(m, 2)
        if not (stop < e < target):
            return None
        return e, stop, target
