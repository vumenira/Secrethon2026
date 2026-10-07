"""How long each impairment seen this week will last: the planner's window with recoveries in it.

``mpc_det`` plans a closed strait, a struck port or a grid short of power as staying so for the whole window. Here each
impairment the observation shows ends as the published duration law of its event type says, given how many weeks it has
been seen: a weather closure (median 6 days) is mostly over by next week, a militarised one (median 31 weeks) is not.

The event type is read from the observation, after the benchmark's own table (``shockbench_flow.policies.scenarios``,
which its ``mpc_scen`` samples from) and finer where the observed value tells more: a strait shut with no war-risk class
is a weather closure, a strait partly open under a war-risk class a militarised closure, a strait shut under one the
year-long closure of a regional conflict (kept); an edge left with the share of capacity a port stoppage leaves is a
stoppage, one in the range a slowdown leaves a slowdown, any other loss a war's or a sanction's (kept); a grid short of
power an energy shock; a supply cap a material outage; a fab or OSAT below full restoration a regional conflict. The
laws are the generator's defaults in ``sbflow/disruption/params.py``. Nothing here reads warnings or announcements.
"""

import math

import numpy as np
from sbflow.disruption.params import FlagshipMarkLaws, MarkLaws, PoissonParams
from sbflow.instance.schema import FLAGSHIP_KINDS
from sbflow.policies import lp_common as L
from scipy.special import gamma, gammaincc, ndtr


# impairment kind -> (its entry of the planner's memory, its window array)
KINDS = {
    "closure": ("open", "o"),
    "capacity": ("u", "u"),
    "grid": ("grid_G", "G_bar"),
    "supply": ("supply", "supply"),
    "fab": ("fab_R", "R"),
    "osat": ("osat_R", "R_osat"),
}
STEPS = 4  # points per week of the integral over a week (Simpson's rule: even)
TINY = 1e-300
NEAR = 1e-6  # two observed shares closer than this are the same severity


def components(mixture):
    """A duration law as (weight, kind, a, b) parts in weeks: lognormal (mu, sigma), Weibull (shape, scale), constant."""
    out = []
    for share, law in mixture:
        for w, part in law.components():
            p = part.params
            if part.kind == "lognormal_days":
                out.append((share * w, "lognormal", p[0] - math.log(7.0), p[1]))
            elif part.kind == "lognormal_weeks":
                out.append((share * w, "lognormal", p[0], p[1]))
            elif part.kind == "weibull_days":
                out.append((share * w, "weibull", p[0], p[1] / 7.0))
            elif part.kind == "constant":
                out.append((share * w, "constant", float(p[0]), 0.0))
            else:
                raise ValueError(f"no survival function for a {part.kind} duration")
    return tuple(part for part in out if part[0] > 0)


def tail(parts, x):
    """The integral of the survival function from x weeks on (x >= 0), a vector."""
    x = np.maximum(np.asarray(x, dtype=float), TINY)
    out = np.zeros_like(x)
    for w, kind, a, b in parts:
        if kind == "lognormal":
            z = (np.log(x) - a) / b
            out += w * np.maximum(math.exp(a + 0.5 * b * b) * ndtr(b - z) - x * ndtr(-z), 0.0)
        elif kind == "weibull":
            out += w * b * gamma(1.0 + 1.0 / a) * gammaincc(1.0 / a, (x / b) ** a)
        else:
            out += w * np.maximum(a - x, 0.0)
    return out


def still_on(parts, age, seen, r):
    """P(the impairment lasts more than r weeks from now), r a vector, after ``age`` whole weeks of seeing it.

    ``seen``: it was shown nominal the week before it was first shown impaired, so it began within the week before
    that and is between ``age`` and ``age`` + 1 weeks old. Otherwise it was already there when first shown (week 1,
    or after an unobserved week) and its start is unknown: the length-biased law of an impairment met in progress.
    """
    r = np.asarray(r, dtype=float)
    if seen:
        top = tail(parts, age + r) - tail(parts, age + 1.0 + r)
        bottom = float(tail(parts, age) - tail(parts, age + 1.0))
    else:
        top, bottom = tail(parts, age + r), float(tail(parts, age))
    return np.clip(top / bottom, 0.0, 1.0) if bottom > 0 else np.zeros_like(r)


def shown(inst, obs):
    """Per kind, which elements this week's observation shows (a blackout week or a masked entry shows nothing)."""
    probe = L.ObservedGraph.nominal(inst)
    for key, _field in KINDS.values():
        probe.values[key] = np.full(probe.values[key].shape, np.nan)
    probe.update(inst, obs)
    return {kind: ~np.isnan(probe.values[key]) for kind, (key, _field) in KINDS.items()}


def strike(share, poisson):
    """The port strike that leaves an edge this share of its capacity; None: another event's cut, which is kept."""
    low, high = (1.0 - s for s in reversed(poisson.slowdown_severity.params))
    if abs(share - (1.0 - poisson.stoppage_severity.params[0])) < NEAR:
        return "port_stoppage"
    return "port_slowdown" if low + NEAR < share < high - NEAR else None


def present(inst, memory, poisson):
    """(kind, element, event type, value, nominal value) of every impairment the planner's memory holds.

    The event type is None where the impairment is kept for the whole window.
    """
    v = memory.values
    out = []
    for c in np.flatnonzero(v["open"] < 1.0):
        if v["war_risk"][c] == 0:
            event = "weather_closure"
        else:  # shut under a war-risk class: a regional conflict's closure of its strait, a year or more
            event = "militarised_closure" if v["open"][c] > NEAR else None
        out.append(("closure", int(c), event, float(v["open"][c]), 1.0))
    for e, edge in enumerate(inst.edges):
        if edge.u0 is not None and v["u"][e] < edge.u0:
            out.append(("capacity", e, strike(v["u"][e] / edge.u0, poisson), float(v["u"][e]), float(edge.u0)))
    for g, node in enumerate(inst.grids):
        nominal = inst.nodes[node].grid.deliverable
        if v["grid_G"][g] < nominal:
            out.append(("grid", g, "energy_shock", float(v["grid_G"][g]), float(nominal)))
    sources = set(inst.supply_nodes)
    for s, slot in enumerate(inst.stock_slots):
        if slot.node in sources and slot.supply > 0.0 and v["supply"][s] < slot.supply:
            out.append(("supply", s, "material_outage", float(v["supply"][s]), float(slot.supply)))
    for f in np.flatnonzero(v["fab_R"] < 1.0):
        out.append(("fab", int(f), "regional_conflict", float(v["fab_R"][f]), 1.0))
    for o in np.flatnonzero(v["osat_R"] < 1.0):
        out.append(("osat", int(o), "regional_conflict", float(v["osat_R"][o]), 1.0))
    return out


class Recovery:
    """The window of one episode's planner: persistence, with each seen impairment ending as its law says.

    ``params``: ``kinds``, the impairment kinds forecast (the others persist); ``mode``, "mean" (each week holds the
    expected share of it the impairment is still on) or "quantile" (on until the ``quantile`` of its remaining life,
    off after); ``caution``, the power the still-on probability is raised to (below 1: recoveries come later).
    """

    def __init__(self, inst, params):
        self.inst, self.params = inst, params
        laws = dict((FlagshipMarkLaws() if inst.kind in FLAGSHIP_KINDS else MarkLaws()).duration)
        self.poisson = PoissonParams()
        self.parts = {name: components([(1.0, law)]) for name, law in laws.items()}
        self.parts["port_stoppage"] = components([(1.0, self.poisson.stoppage_duration)])
        self.parts["port_slowdown"] = components([(1.0, self.poisson.slowdown_duration)])
        self.kappa0 = np.array([inst.nodes[c].chokepoint.kappa0 for c in inst.chokepoints], dtype=float).reshape(-1, 2)
        self.book = {}  # (kind, element) -> (the first week it was shown impaired, whether its start was seen)
        self.week, self.shown, self.elements = None, None, []
        self.shares = {}

    def update(self, obs, memory):
        """Book this week's impairments; ``memory`` already holds this week's observation."""
        t = int(obs["week"])
        before = self.shown if self.week == t - 1 else None
        self.elements = present(self.inst, memory, self.poisson)
        book = {}
        for kind, element, *_ in self.elements:
            key = (kind, element)
            book[key] = self.book.get(key) or (t, before is not None and bool(before[kind][element]))
        self.book, self.week, self.shown = book, t, shown(self.inst, obs)

    def share(self, event, age, seen, H):
        """(H,) the share of each window week the impairment is planned as still on."""
        key = (event, age, seen, H)
        if key not in self.shares:
            r = np.arange(H * STEPS + 1) / STEPS
            p = still_on(self.parts[event], float(age), seen, r) ** float(self.params["caution"])
            if self.params["mode"] == "quantile":  # on for the remaining life's quantile, then off
                past = np.flatnonzero(p <= 1.0 - float(self.params["quantile"]))
                end = r[past[0]] if past.size else math.inf
                out = np.clip(end - np.arange(H), 0.0, 1.0)
            else:  # Simpson's rule over each week
                weights = np.array([1.0, 4.0, 2.0, 4.0, 1.0]) / 12.0
                out = np.array([float(weights @ p[h * STEPS : (h + 1) * STEPS + 1]) for h in range(H)])
            self.shares[key] = out
        return self.shares[key]

    def series(self, kind, element, event, value, nominal, t, H):
        """(H,) the element's value in each window week."""
        first, seen = self.book[(kind, element)]
        return nominal - (nominal - value) * self.share(event, t - first, seen, H)

    def arrays(self, obs, memory, H):
        """The planner's window arrays of weeks t..t + H - 1 (``lp_common.persistence_arrays`` with recoveries)."""
        t = int(obs["week"])
        out = {name: a.copy() for name, a in L.persistence_arrays(self.inst, obs, memory, H).items()}
        for kind, element, event, value, nominal in self.elements:
            if event is None or kind not in self.params["kinds"]:
                continue
            out[KINDS[kind][1]][:, element] = self.series(kind, element, event, value, nominal, t, H)
            if kind == "closure":  # a strait's throughput follows its open fraction
                out["kappa"][:, element, :] = self.kappa0[element] * out["o"][:, element, None]
        return L.read_only(L.with_now(out))
