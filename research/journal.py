#!/usr/bin/env python3
"""Durable, append-only records for the paper bot. Standard library only.

Rule: write the INTENT to disk (fsync) BEFORE any order leaves the machine, write the broker's answer right after,
and save state atomically (temp file + rename) so a crash never leaves half a file.
"""

import datetime
import json
import os


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def append_jsonl(path, rec):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    out.append({"kind": "corrupt_line", "raw": line[:200]})
    return out


def save_json_atomic(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class Journal:
    """<dir>/decisions.jsonl (every tick decision), <dir>/orders.jsonl (intent + broker answers), <dir>/state.json."""

    def __init__(self, directory):
        self.dir = directory
        self.decisions_path = os.path.join(directory, "decisions.jsonl")
        self.orders_path = os.path.join(directory, "orders.jsonl")
        self.state_path = os.path.join(directory, "state.json")
        self.trades_path = os.path.join(directory, "trades.jsonl")      # permanent: one record per finished trade

    def decision(self, kind, **kw):
        append_jsonl(self.decisions_path, dict(kw, kind=kind, at=now_iso()))

    def order(self, kind, **kw):
        append_jsonl(self.orders_path, dict(kw, kind=kind, at=now_iso()))

    def trade(self, **kw):
        append_jsonl(self.trades_path, dict(kw, at=now_iso()))

    def load_state(self):
        return load_json(self.state_path, {})

    def save_state(self, state):
        save_json_atomic(self.state_path, state)
