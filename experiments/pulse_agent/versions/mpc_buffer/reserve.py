"""Three changes to the planner's window LP: a fuel reserve at grids, a ramp on shed load, and a window-end value.

The planner (``sbflow``, the benchmark's ``mpc_det``) plans on "what I see now stays as it is", so it holds no fuel
against events that have not started, it is indifferent to when a shortage it expects to last is taken, and it prices
whatever is left when its window ends at the end-of-episode salvage price. The changes are added to the benchmark's own
builder (``sbflow.oracle.lp._Builder``) just before it stacks the model, so ``sbflow/`` stays as vendored. They are
columns, rows and entries of the planning rules' price table (``meta["priority"]``), which enters the objective only:
the cost parts, the action mapping and the warm start (the same columns and rows in every window week) are untouched.

The reserve (``reserve_weeks`` > 0). For a fuel k that a grid g burns, in every window week t:

    I[g, k, t] + short[g, k, t] >= target[g, k, t],   short >= 0,   priced reserve_price * VOLL_g per unit-week

- ``target = min(cover * shares[k] * deliverable_g, storage[g, k])``: ``cover`` weeks of the grid's full-rate burn of
  the fuel (a fuel unit is one GWh; nominal values of the instance, never the week's impaired ones), with
  ``cover = min(reserve_weeks, weeks of the episode after t)``, so nothing is asked for after the last week.
- ``reserve_feeders``: the node that feeds the grid over a zero-lead edge (its terminal) holds the same fuel for it.
  With the switch on, the cover the grid's own storage cannot hold is asked of those nodes, each up to its storage, so
  ``reserve_weeks`` may exceed the grid's storage (about 4 weeks of burn).

The ramp (``shed_ramp`` > 0). Shed base load of window week h is priced ``shed_ramp * VOLL_g * (H - h)`` above VOLL at
every grid (H the window's weeks): shedding a week later is cheaper by ``shed_ramp * VOLL``, so of two plans that shed
the same the planner takes the one that sheds later. It is also the reserve's guard. A price on low stock rewards a
plan that sheds early and keeps the fuel when its forecast says the shortage lasts (a unit shed in week h instead of
week h' > h saves h' - h weeks of the reserve's price), at the same grid, which the simulator decides, or at another
grid by sending the fuel elsewhere, which the agent decides. With ``shed_ramp`` above ``reserve_price`` that never
pays, and the reserve is filled only with fuel no grid would burn in the window.

The end value (``end_value`` > 0), only while the window ends before the episode does. Goods at the window's end are
credited ``end_value * v_k`` (the commodity's customs value) where that is above the salvage price the builder gives
them: stock at nodes other than supply nodes and chokepoints, flows still in transit (times the strait's open share
when they head for one), lots and OSAT starts still in process (at their input's value, as the builder's salvage).
Queues at straits keep their salvage price, and so do the stocks the planning rules price (a fab's wafers, an OSAT's raw
chips), which the simulator moves on by itself.
"""

from dataclasses import dataclass

import numpy as np
from sbflow.marks import fixed_fp_errors
from sbflow.oracle.lp import _Builder
from sbflow.policies import lp_common


FUELS = ("all", "rationed")


@dataclass(frozen=True)
class Settings:
    """The knobs of ``agent.PARAMS`` that this module reads, checked, with the week the window starts in."""

    weeks: float  # reserve_weeks
    price: float  # reserve_price
    fuels: str  # reserve_fuels
    feeders: bool  # reserve_feeders
    ramp: float  # shed_ramp
    end_value: float
    week: int  # the episode week of the window's first week
    T: int  # the episode's weeks

    @classmethod
    def of(cls, params: dict, week: int, T: int) -> "Settings":
        out = cls(
            weeks=float(params["reserve_weeks"]),
            price=float(params["reserve_price"]),
            fuels=params["reserve_fuels"],
            feeders=bool(params["reserve_feeders"]),
            ramp=float(params["shed_ramp"]),
            end_value=float(params["end_value"]),
            week=int(week),
            T=int(T),
        )
        numbers = (out.weeks, out.price, out.ramp, out.end_value)
        if not all(np.isfinite(v) and v >= 0 for v in numbers) or out.fuels not in FUELS:
            raise ValueError(f"reserve settings out of range: {out}")
        return out

    @property
    def reserve(self) -> bool:
        return self.weeks > 0 and self.price > 0


class Builder(_Builder):
    """The benchmark's builder with the reserve, the ramp and the end value added before the model is stacked."""

    def __init__(self, inst, marks, planning_rules: bool, settings: Settings) -> None:
        super().__init__(inst, marks, planning_rules=planning_rules)
        self.settings = settings

    def finish(self):
        st = self.settings
        end_value = st.end_value > 0 and st.week + self.T - 1 < st.T  # the window ends before the episode does
        changed = st.reserve or st.ramp > 0 or end_value
        if changed:
            if not self.rules:  # the builder keeps the price table only with the planning rules
                raise ValueError("the reserve, the ramp and the end value are prices of the planning rules' table")
            keys = list(self.cols.keys)  # the builder's own columns, before the reserve's
            where = {key: j for j, key in enumerate(keys)}
            ruled = frozenset(self.priority)  # the columns the planning rules price
            if end_value:
                self.add_end_value(keys, ruled)
            if st.ramp > 0:
                self.add_ramp(where)
            if st.reserve:
                self.add_reserve(where)
        model = super().finish()
        model.meta["reserve"] = changed  # False: the model is the benchmark's own
        return model

    def add_price(self, j: int, price) -> None:
        """Add ``price`` (a scalar or one value per window week) to column template ``j``'s entry of the table."""
        self.priority[j] = np.asarray(self.priority.get(j, 0.0), dtype=float) + price

    # ----- the reserve -----------------------------------------------------------------------------------------------
    def reserve_targets(self) -> list[tuple[tuple, int, np.ndarray, float]]:
        """(name, stock slot, target per window week, price per unit-week) of every soft minimum."""
        inst, st = self.inst, self.settings
        S, E = inst.stock_slots, inst.edges
        left = st.T - (st.week + self.t - 1)  # weeks of the episode after each window week
        cover = np.minimum(st.weeks, np.maximum(left, 0)).astype(float)
        out = []
        asked: dict[int, list] = {}  # feeder slot -> [target, the highest price of the grids it feeds]
        for go, g in enumerate(inst.grids):
            ga = inst.nodes[g].grid
            price = st.price * ga.voll
            for k in ga.fuels:
                if st.fuels == "rationed" and k != ga.rationed:
                    continue
                s = self.slot(g, k, "grid fuel")
                want = cover * ga.shares[k] * ga.deliverable
                target = np.minimum(want, S[s].storage)
                out.append((("reserve", go, k), s, target, price))
                rest = want - target
                if not st.feeders or not rest.any():
                    continue
                for e in inst.in_edges[g]:  # the cover its storage cannot hold, at the node that feeds it at once
                    ed = E[e]
                    if k not in ed.K or ed.tau != 0 or ed.tail in self.chk or ed.tail in self.supply:
                        continue
                    fs = inst.slot_index.get((ed.tail, k))
                    if fs is None:
                        continue
                    entry = asked.setdefault(fs, [np.zeros(self.T), 0.0])
                    entry[0] = entry[0] + rest
                    entry[1] = max(entry[1], price)
                    break  # one feeder per grid and fuel: the first in edge order
        for fs, (target, price) in asked.items():
            out.append((("reserve_feeder", fs), fs, np.minimum(target, S[fs].storage), price))
        return out

    def add_reserve(self, where: dict[tuple, int]) -> None:
        """The soft minimum of every reserve slot (module docstring)."""
        for name, s, target, price in self.reserve_targets():
            j = self.cols.add(name)  # short: the stock missing below the target
            self.priority[j] = price
            r = self.ub.row(name, -target)  # -I - short <= -target
            self.ub.add(r, where[("I", s)], -1.0)
            self.ub.add(r, j, -1.0)

    # ----- the ramp --------------------------------------------------------------------------------------------------
    def add_ramp(self, where: dict[tuple, int]) -> None:
        """Shed base load costs ``shed_ramp * VOLL`` more for every window week still to come (module docstring)."""
        inst = self.inst
        later = (self.T - self.t).astype(float)  # the window weeks after each week
        for go, g in enumerate(inst.grids):
            self.add_price(where[("ysh", go)], self.settings.ramp * inst.nodes[g].grid.voll * later)

    # ----- the window's end ------------------------------------------------------------------------------------------
    def add_end_value(self, keys: list[tuple], ruled: frozenset) -> None:
        """Credit goods at the window's end ``end_value * v_k`` where that exceeds their salvage (module docstring)."""
        inst, marks, share, t, T = self.inst, self.marks, self.settings.end_value, self.t, self.T
        K, S, E = inst.commodities, inst.stock_slots, inst.edges
        last = (t == T).astype(float)
        for j, key in enumerate(keys):
            tag = key[0]
            if tag == "I":
                slot = S[key[1]]
                if slot.node in self.supply or j in ruled:
                    continue
                gain, when = share * K[slot.k].v - slot.salvage, last
            elif tag == "x":
                e, k = key[1], key[2]
                ed = E[e]
                if ed.head in self.supply:
                    continue
                gain, when = share * K[k].v - self.nu(ed.head, k, "arrival"), (t + ed.tau > T).astype(float)
                if ed.head in self.chk:  # still short of a strait: worth what the strait lets through
                    when = when * min(1.0, max(0.0, float(marks.o[T - 1, inst.chokepoint_ordinal[ed.head]])))
            elif tag == "p":
                fo = key[1]
                fa = inst.nodes[inst.fabs[fo]].fab
                nu = S[self.slot(inst.fabs[fo], fa.input, "fab input")].salvage
                gain, when = share * K[fa.input].v - nu, (t + fa.tau > T) * (1.0 - marks.sigma_scr[:, fo])
            elif tag == "xi":
                oo, kp = key[1], key[2]
                o = inst.osats[oo]
                oa = inst.nodes[o].osat
                kr = next(raw for raw, packaged in oa.packages.items() if packaged == kp)
                gain = share * K[kr].v - S[self.slot(o, kr, "OSAT raw chip")].salvage
                when = (t + oa.tau > T).astype(float)
            else:
                continue
            if gain > 0:
                self.add_price(j, -gain * when)


@fixed_fp_errors
def build_lp(inst, marks, settings: Settings, *, planning_rules: bool):
    """``sbflow.oracle.lp.build_lp`` with ``Builder``: the same instance rung and the same checks (V3)."""
    inst = inst.at_digest(marks.instance_digest)
    if marks.instance_hash != inst.hash or marks.T != inst.T or marks.instance_digest != inst.content_digest:
        raise ValueError("marks were computed for another instance or horizon (V3)")
    return Builder(inst, marks, planning_rules, settings).build()


def changes(model) -> bool:
    """Whether ``model`` carries a reserve, a ramp or an end value (False: it is the benchmark's own model)."""
    return bool(model.meta.get("reserve"))


def rolled_lp(inst, obs: dict, arrays, H: int, params: dict, *, planning_rules: bool):
    """``lp_common.rolled_lp`` of week ``obs["week"]`` with the reserve, the ramp and the end value of ``params``."""
    settings = Settings.of(params, int(obs["week"]), inst.T)
    inst_r, backlog = lp_common.rolled_window(inst, obs, H)
    marks = lp_common.window_marks(inst_r, arrays, backlog)
    return build_lp(inst_r, marks, settings, planning_rules=planning_rules)
