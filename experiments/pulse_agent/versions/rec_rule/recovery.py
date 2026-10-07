"""How long each impairment seen this week will last: the planner's window with recoveries in it.

``mpc_det`` plans a closed strait, a struck port or a grid short of power as staying so for the whole window. Here each
impairment the observation shows ends as the published duration law of its event type says, given how many weeks it has
been seen: a weather closure (median 6 days) is mostly over by next week, a militarised one (median 31 weeks) is not.

The event type is read from the observation, after the benchmark's own table (``shockbench_flow.policies.scenarios``,
which its ``mpc_scen`` samples from) and finer where the observed value tells more: an edge left with the share of
capacity a port stoppage leaves is a stoppage, one in the range a slowdown leaves a slowdown, any other loss a war's or
a sanction's (kept); a supply cap a material outage; a fab or OSAT below full restoration a regional conflict.

A strait's open fraction and a grid's deliverable power are products of the surviving shares of every event in force
(``marks._graph_marks``), so these two kinds are booked as layers, one per event, each with its own type, age and end:

- a strait partly open is under a militarised closure (severity 0.27 to 0.97, so never shut);
- a strait shut is under a weather closure (severity 1), whatever its war-risk class: the class comes from a militarised
  closure or a regional conflict nearby, and a storm can fall on either. The one exception is the closure a conflict
  between a dyad of ``strait_closures`` derives on its strait (CN-TW, the Taiwan Strait: shut for the conflict's length
  and 52 weeks at least), told by the war's cut on every edge between the two regions, which starts the same instant;
  it is kept for the whole window;
- a grid short of power is under energy shocks;
- a share that falls while impaired is a further event on top (begun this week); a share that rises without reaching
  nominal is the end of the layers whose removal explains it.

The laws are the generator's defaults in ``sbflow/disruption/params.py``. Nothing here reads warnings or announcements.
"""

import math
from itertools import combinations

import numpy as np
from sbflow.disruption.params import FlagshipMarkLaws, MarkLaws, PoissonParams
from sbflow.instance.schema import FLAGSHIP_KINDS
from sbflow.marks import MarkParams
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
LAYERED = ("closure", "grid")  # kinds whose shown value is a product of the surviving shares of the events in force
RULE = ("mode", "quantile", "caution")  # what a rule sets, per kind or per event type
STEPS = 4  # points per week of the integral over a week (Simpson's rule: even)
TINY = 1e-300
NEAR = 1e-6  # two observed shares closer than this are the same severity
MAX_LAYERS = 6  # more events than this on one element are booked as one


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

    The event type is None where the impairment is kept for the whole window, and "layers" for a strait or a grid,
    whose events are told apart in ``Recovery.relayer``.
    """
    v = memory.values
    out = []
    for c in np.flatnonzero(v["open"] < 1.0):
        out.append(("closure", int(c), "layers", float(v["open"][c]), 1.0))
    for e, edge in enumerate(inst.edges):
        if edge.u0 is not None and v["u"][e] < edge.u0:
            out.append(("capacity", e, strike(v["u"][e] / edge.u0, poisson), float(v["u"][e]), float(edge.u0)))
    for g, node in enumerate(inst.grids):
        nominal = inst.nodes[node].grid.deliverable
        if v["grid_G"][g] < nominal:
            out.append(("grid", g, "layers", float(v["grid_G"][g]), float(nominal)))
    sources = set(inst.supply_nodes)
    for s, slot in enumerate(inst.stock_slots):
        if slot.node in sources and slot.supply > 0.0 and v["supply"][s] < slot.supply:
            out.append(("supply", s, "material_outage", float(v["supply"][s]), float(slot.supply)))
    for f in np.flatnonzero(v["fab_R"] < 1.0):
        out.append(("fab", int(f), "regional_conflict", float(v["fab_R"][f]), 1.0))
    for o in np.flatnonzero(v["osat_R"] < 1.0):
        out.append(("osat", int(o), "regional_conflict", float(v["osat_R"][o]), 1.0))
    return out


def dyad_edges(inst, laws):
    """Per strait (its ordinal) that a dyad's conflict closes: the edges between the dyad's two regions.

    The list is empty where the instance has no such edge with a capacity: the conflict cannot be told there.
    """
    out = {}
    for a, b, strait in getattr(laws, "strait_closures", ()):
        pair = {inst.region_index.get(a), inst.region_index.get(b)}
        edges = [
            e
            for e, edge in enumerate(inst.edges)
            if not edge.coupling
            and edge.u0 is not None
            and {inst.nodes[edge.tail].region, inst.nodes[edge.head].region} == pair
        ]
        for c in inst.chokepoints:
            if inst.nodes[c].id == strait:
                out[inst.chokepoint_ordinal[c]] = edges
    return out


class Recovery:
    """The window of one episode's planner: persistence, with each seen impairment ending as its law says.

    ``params``: ``kinds``, the impairment kinds forecast (the others persist); ``mode``, "mean" (each week holds the
    expected share of it the impairment is still on) or "quantile" (on until the ``quantile`` of its remaining life,
    off after; quantile 0: planned as over from this week on); ``caution``, the power the still-on probability is
    raised to (below 1: recoveries come later); ``rules``, {a kind or an event type: {"mode", "quantile", "caution"}}
    replacing those three where it names them (an event type's rule before its kind's).
    """

    def __init__(self, inst, params):
        self.inst, self.params = inst, params
        self.laws = FlagshipMarkLaws() if inst.kind in FLAGSHIP_KINDS else MarkLaws()
        laws = dict(self.laws.duration)
        self.poisson = PoissonParams()
        self.parts = {name: components([(1.0, law)]) for name, law in laws.items()}
        self.parts["port_stoppage"] = components([(1.0, self.poisson.stoppage_duration)])
        self.parts["port_slowdown"] = components([(1.0, self.poisson.slowdown_duration)])
        self.dyads = dyad_edges(inst, self.laws)
        self.war_share = 1.0 - MarkParams().war_profile_belligerent[0]  # what a war leaves of an edge between the two
        self.kappa0 = np.array([inst.nodes[c].chokepoint.kappa0 for c in inst.chokepoints], dtype=float).reshape(-1, 2)
        self.book = {}  # (kind, element) -> (the first week it was shown impaired, whether its start was seen)
        self.layers = {}  # (kind, element) of a layered kind -> [(surviving share, event type, first week, seen)]
        self.week, self.shown, self.elements = None, None, []
        self.shares, self.rules = {}, {}

    def event(self, kind, element, factor, memory):
        """The event type behind one layer of a strait or a grid that leaves ``factor`` of the element; None: kept."""
        if kind == "grid":
            return "energy_shock"
        if factor > NEAR:
            return "militarised_closure"
        v = memory.values
        if v["war_risk"][element] != 0 and element in self.dyads:
            # shut by its dyad's conflict: every edge between the two regions is cut (no such edge: it may be)
            edges = self.dyads[element]
            if all(v["u"][e] <= self.war_share * self.inst.edges[e].u0 * (1.0 + NEAR) for e in edges):
                return None  # a year or more, and conflicts of one dyad cluster: no law is used for it
        return "weather_closure"

    def relayer(self, old, kind, element, share, t, seen, memory):
        """The layers behind an element shown at ``share`` of its nominal value, after last week's layers ``old``."""
        if not old:
            return [(share, self.event(kind, element, share, memory), t, seen)]
        before = math.prod(layer[0] for layer in old)
        if abs(share - before) <= NEAR:
            return old
        oldest = min(old, key=lambda layer: layer[2])
        if share < before:  # a further event began on top of those already there
            factor = share / before
            if len(old) < MAX_LAYERS:
                return [*old, (factor, self.event(kind, element, factor, memory), t, seen)]
        else:  # some ended: the layers left are those whose product is the share shown now
            for size in range(len(old) - 1, 0, -1):
                for kept in combinations(old, size):
                    if abs(math.prod(layer[0] for layer in kept) - share) <= NEAR:
                        return list(kept)
        # no reading of the change: one impairment as old as the oldest layer
        return [(share, self.event(kind, element, share, memory), oldest[2], oldest[3])]

    def update(self, obs, memory):
        """Book this week's impairments; ``memory`` already holds this week's observation."""
        t = int(obs["week"])
        before = self.shown if self.week == t - 1 else None
        book, layers, elements = {}, {}, []
        for kind, element, event, value, nominal in present(self.inst, memory, self.poisson):
            key = (kind, element)
            seen = before is not None and bool(before[kind][element])
            if kind in LAYERED:
                layers[key] = self.relayer(self.layers.get(key), kind, element, value / nominal, t, seen, memory)
                if all(layer[1] is None for layer in layers[key]):
                    event = None
            else:
                book[key] = self.book.get(key) or (t, seen)
            elements.append((kind, element, event, value, nominal))
        self.book, self.layers, self.elements = book, layers, elements
        self.week, self.shown = t, shown(self.inst, obs)

    def rule(self, kind, event):
        """(mode, quantile, caution) of an event type of a kind."""
        key = (kind, event)
        if key not in self.rules:
            own = self.params.get("rules") or {}
            pick = {**{name: self.params[name] for name in RULE}, **own.get(kind, {}), **own.get(event, {})}
            self.rules[key] = (str(pick["mode"]), float(pick["quantile"]), float(pick["caution"]))
        return self.rules[key]

    def share(self, kind, event, age, seen, H):
        """(H,) the share of each window week the impairment is planned as still on."""
        mode, quantile, caution = self.rule(kind, event)
        key = (event, age, seen, H, mode, quantile, caution)
        if key not in self.shares:
            r = np.arange(H * STEPS + 1) / STEPS
            p = still_on(self.parts[event], float(age), seen, r) ** caution
            if mode == "quantile":  # on for the remaining life's quantile, then off
                past = np.flatnonzero(p <= 1.0 - quantile)
                end = r[past[0]] if past.size else math.inf
                out = np.clip(end - np.arange(H), 0.0, 1.0)
            else:  # Simpson's rule over each week
                weights = np.array([1.0, 4.0, 2.0, 4.0, 1.0]) / 12.0
                out = np.array([float(weights @ p[h * STEPS : (h + 1) * STEPS + 1]) for h in range(H)])
            self.shares[key] = out
        return self.shares[key]

    def series(self, kind, element, event, value, nominal, t, H):
        """(H,) the element's value in each window week."""
        if kind not in LAYERED:
            first, seen = self.book[(kind, element)]
            return nominal - (nominal - value) * self.share(kind, event, t - first, seen, H)
        layers = self.layers[(kind, element)]
        if len(layers) == 1:
            _factor, event, first, seen = layers[0]
            return nominal - (nominal - value) * self.share(kind, event, t - first, seen, H)
        left = np.ones(H)  # the share of nominal the layers leave: each cuts its own share while it is on
        for factor, event, first, seen in layers:
            on = 1.0 if event is None else self.share(kind, event, t - first, seen, H)
            left = left * (1.0 - (1.0 - factor) * on)
        now = value / nominal
        return value + (nominal - value) * np.clip((left - now) / (1.0 - now), 0.0, 1.0)

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
