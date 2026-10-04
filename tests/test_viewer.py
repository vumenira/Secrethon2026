"""``sbf view``: the page is written, its data adds up, and a second run reuses the recordings."""

import json
import re

import numpy as np
from shockbench_flow_gym import dashboard

from sbf_starter.viewer import view
from tests.conftest import ROOT


TEMPLATE = ROOT / "agents" / "template"


def _data(page) -> dict:
    text = page.read_text()
    (blob,) = re.findall(r"const DATA = (\{.*?\});</script>", text, flags=re.S)
    return json.loads(blob.replace("<\\/", "</"))


def test_view_writes_a_page_whose_numbers_add_up(tmp_path, capsys):
    page = view(str(TEMPLATE), "tiny", episode=0, quick=True, n_jobs=1, out=str(tmp_path))
    assert page == tmp_path / "viewer.html"
    data = _data(page)
    (ep,) = data["episodes"]
    rec = dashboard.load_record(tmp_path / "record_ep0.npz")
    T = rec["meta"]["T"]

    assert ep["T"] == T and len(ep["weeks"]["state"]) == T
    assert len(data["net"]["nodes"]) == len(rec["static"]["nodes"]["id"])
    assert abs(sum(ep["weeks"]["cost"]) - rec["meta"]["J_cents"] / 100) <= T  # whole dollars per week
    assert "naive" in ep and len(ep["naive"]["cost"]) == T

    shipped = dashboard.shipped_units(rec)
    assert np.allclose(np.sum(ep["weeks"]["executed"], axis=1), shipped.sum(axis=1), rtol=1e-3)
    slots, lanes = data["net"]["slots"], data["net"]["lanes"]
    span = [len(lanes[s["lane"]]["edges"]) if s["lane"] is not None else 1 for s in slots]  # a lane counts on each edge
    per_edge = [sum(q for _, _, q in week) for week in ep["weeks"]["flows"]]
    assert np.allclose(per_edge, shipped @ np.array(span), rtol=1e-3)
    assert all(q > 0 for week in ep["weeks"]["transit"] for *_, q in week)
    assert all(s["plan"] >= 0 for s in data["net"]["slots"])
    assert "release" not in ep  # the template never takes over a strait's release

    capsys.readouterr()
    view(str(TEMPLATE), "tiny", episode=0, quick=True, n_jobs=1, out=str(tmp_path))
    assert "unchanged since the last run" in capsys.readouterr().out


def test_layout_keeps_kinds_in_order_and_straits_in_the_gaps():
    import gymnasium as gym

    from sbf_starter.viewer.export import KIND_COLUMN, layout_positions, normal_plan

    for task in ("ShockBench/Tiny-v0", "ShockBench/Small-v0"):
        _obs, info = gym.make(task).reset(options={"episode": 0})  # the module's dashboard import registered them
        static = info["static"]
        pos = layout_positions(static)
        types = static["nodes"]["type"]
        assert len(pos) == len(types)
        for t, (x, _) in zip(types, pos):  # straits on odd (gap) columns, every other kind on even ones
            assert x % 2 == (1 if t == "chokepoint" else 0)
        kinds = sorted({t for t in types if t != "chokepoint"}, key=KIND_COLUMN.get)
        xs = [min(x for t, (x, _) in zip(types, pos) if t == k) for k in kinds]
        assert xs == sorted(xs)
        assert len({p for p in pos}) == len(pos)  # no two nodes on one spot

        plan = normal_plan(static)
        assert len(plan) == len(static["action_slots"]["edge"])
        assert min(plan) >= 0 and sum(q > 0 for q in plan) > 0
