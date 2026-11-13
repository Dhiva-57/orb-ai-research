#!/usr/bin/env python3
"""Strategy router: when several strategies trigger on the same symbol and bar, which one (if any) gets the trade?

    1. every strategy proposes its own candidate (strategies.live_candidate: same code as the backtest)
    2. each strategy's OWN model must say yes (calibrated P(target before stop) >= its own threshold)
    3. survivors are ranked by expected R:  EV = p * reward_in_R - (1 - p)       (timeouts ignored: a simple, stated rule)
    4. the bot walks down the ranking: the first one the risk engine, sizing and drift guard accept is the trade
    5. ORDERS are allowed only for strategies whose own real-data walk-forward PASS is linked to their own model
       (paper_bot.gate_ok, per strategy). Others still run in dry-run so you can watch them.
One position per symbol per day regardless of how many strategies agree. This is arbitration, not a regime detector:
it does not know what kind of market it is in, only which candidate looks best on its own model's numbers.
"""

import os

import model_store as ms
import strategies as ss


class Option:
    def __init__(self, strategy, model=None, meta=None):
        self.strategy, self.model, self.meta = strategy, model, meta or {}
        self.name = strategy.name


def ev_r(p, row):
    """Expected R of a trade with calibrated win probability p and the candidate's own reward/risk."""
    reward = (row["target"] - row["entry"]) / row["risk"]
    return p * reward - (1.0 - p)


def rank(scored):
    """scored: [{"ev", "opt", "cand", "p"}] -> best first (ties broken by strategy name, so it is deterministic)."""
    return sorted(scored, key=lambda s: (-s["ev"], s["opt"].name))


class Router:
    def __init__(self, options):
        names = [o.name for o in options]
        if len(set(names)) != len(names):
            raise ValueError("duplicate strategies in the router: %s" % names)
        if any(o.strategy.bar_seconds != options[0].strategy.bar_seconds for o in options):
            raise ValueError("a router needs strategies on the same bar size")
        self.options = options


def load_options(names, model_root):
    """One Option per strategy; its model comes from <model_root>/<name>/ (model_store.py --out <model_root>/<name>).
    A strategy without a usable model still runs, but only as a model-less DRY-RUN (it can never place orders)."""
    out = []
    for n in names:
        strat = ss.get(n)
        try:
            model, meta = ms.load(os.path.join(model_root, n))
        except (OSError, ms.ModelError) as e:
            print("strategy %s: no usable model (%s): dry-run only" % (n, e))
            model, meta = None, {}
        out.append(Option(strat, model, meta))
    return out
