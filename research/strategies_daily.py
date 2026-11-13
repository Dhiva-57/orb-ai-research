#!/usr/bin/env python3
"""*_daily variants of every strategy: identical triggers/levels, plus ONE extra feature, ctx_daily_p_up = the as-of-yesterday
probability that today closes up (daily_signal.p_up_from_sessions, from completed sessions only). The ctx_ prefix makes
orb_dataset.feature_columns() pick it up with no other change.

Nothing says this feature helps: daily_signal.py usually finds no daily edge. The variants exist so that the question gets
answered by the SAME walk-forward gate as everything else:  walkforward.py --strategy orb   vs   --strategy orb_daily.
They need ~320 prior sessions (history_sessions) for live use; the bot loads that many automatically.
"""

import daily_signal as dsg
import strategies as ss


def make_daily(base):
    class Daily(base):
        name = base.name + "_daily"
        history_sessions = dsg.LOOKBACK + 20

        def features(self, tm, D, k, prior, context):
            f = super().features(tm, D, k, prior, context)
            f["ctx_daily_p_up"] = dsg.p_up_from_sessions(prior)
            return f

    Daily.__name__ = base.__name__ + "Daily"
    Daily.__doc__ = (base.__doc__ or "") + "  [+ ctx_daily_p_up feature]"
    return Daily


for _name in ("orb", "momentum", "pullback", "vwap", "meanrev"):
    ss.register(make_daily(ss.REGISTRY[_name]))
