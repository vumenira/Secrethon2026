"""Weekly series of one episode's column vector (ours or an optimised one), by name, in natural units."""

from collections import defaultdict

import numpy as np


class View:
    """One episode's columns by name: a (T, nc) array read through the model's template."""

    def __init__(self, cal):
        self.cal, self.inst, self.tmpl = cal, cal.inst, cal.tmpl
        inst = cal.inst
        self.K = [c.id for c in inst.commodities]
        self.kid = {c.id: k for k, c in enumerate(inst.commodities)}
        self.node = [nd.id for nd in inst.nodes]
        self.flows = defaultdict(list)  # (edge, k) -> template columns (its lanes)
        for key, j in cal.tmpl.items():
            if key[0] == "x":
                self.flows[(key[1], key[2])].append(j)
        self.chk = set(inst.chokepoints)
        self.fuels = sorted({k for g in inst.grids for k in inst.nodes[g].grid.fuels})

    def col(self, a, key):
        j = self.tmpl.get(key)
        return np.zeros(a.shape[0]) if j is None else a[:, j]

    def flow(self, a, e, k):
        return a[:, self.flows[(e, k)]].sum(axis=1) if (e, k) in self.flows else np.zeros(a.shape[0])

    def stock(self, a, node, k):
        s = self.inst.slot_index.get((node, k))
        if s is None:
            return np.zeros(a.shape[0])
        if node in self.chk:
            out = np.zeros(a.shape[0])
            for key, j in self.tmpl.items():
                if key[0] == "Q" and key[1] == node and key[2] == k:
                    out += a[:, j]
            return out
        return self.col(a, ("I", s))

    def into(self, a, node, k, arrive=False):
        """Flow of k dispatched toward ``node`` per week (by dispatch week; ``arrive``: by arrival week)."""
        T = a.shape[0]
        out = np.zeros(T)
        for e in self.inst.in_edges[node]:
            ed = self.inst.edges[e]
            if (e, k) in self.flows:
                f = self.flow(a, e, k)
                if arrive and ed.tau:
                    g = np.zeros(T)
                    g[ed.tau :] = f[: T - ed.tau]
                    f = g
                out += f
        return out

    def out_of(self, a, node, k):
        out = np.zeros(a.shape[0])
        for e in self.inst.out_edges[node]:
            if (e, k) in self.flows:
                out += self.flow(a, e, k)
        return out

    def terminal(self, go, k):
        """The node that feeds grid ``go`` its fuel k over a zero-lead edge, and that edge; (None, None) without one."""
        g = self.inst.grids[go]
        for e in self.inst.in_edges[g]:
            ed = self.inst.edges[e]
            if k in ed.K and ed.tau == 0:
                return ed.tail, e
        return None, None


def series(v, a):
    """{name: (T,) array} of one column vector: the quantities the opening is compared on, in natural units."""
    inst, cal, marks = v.inst, v.cal, v.cal.marks
    out = {}
    wafer = None
    for fo, f in enumerate(inst.fabs):
        fa = inst.nodes[f].fab
        wafer = fa.input
        name = v.node[f]
        out[f"lots {name}"] = v.col(a, ("p", fo)) / fa.cap0
        out[f"wafers at {name}"] = v.stock(a, f, fa.input) / fa.cap0
        out[f"wafers sent to {name}"] = v.into(a, f, fa.input) / fa.cap0
        out[f"raw chips at {name}"] = v.stock(a, f, fa.product) / fa.cap0
        out[f"raw chips out of {name}"] = v.out_of(a, f, fa.product) / fa.cap0
    for go, g in enumerate(inst.grids):
        ga = inst.nodes[g].grid
        name = v.node[g]
        fabs = [fo for fo in inst.grid_fabs[go] if inst.nodes[inst.fabs[fo]].fab.e > 0]
        draw = sum(inst.nodes[inst.fabs[fo]].fab.e * inst.nodes[inst.fabs[fo]].fab.cap0 for fo in fabs)
        if fabs:
            out[f"fab energy {name}"] = sum(v.col(a, ("E", fo)) for fo in fabs) / draw
        out[f"shed {name}"] = v.col(a, ("ysh", go)) / ga.base_load
        for k in ga.fuels:
            burn = ga.shares[k] * ga.deliverable
            fuel = v.K[k]
            out[f"{fuel} burnt {name}"] = v.col(a, ("G", go, k)) / burn
            out[f"{fuel} at {name}"] = v.stock(a, g, k) / burn
            tl, e = v.terminal(go, k)
            if tl is not None:
                out[f"{fuel} at {v.node[tl]}"] = v.stock(a, tl, k) / burn
                out[f"{fuel} delivered {name}"] = v.flow(a, e, k) / burn
                out[f"{fuel} sent to {v.node[tl]}"] = v.into(a, tl, k) / burn
                out[f"{fuel} arriving {v.node[tl]}"] = v.into(a, tl, k, arrive=True) / burn
            else:
                out[f"{fuel} sent to {name}"] = v.into(a, g, k) / burn
    for k in v.fuels + [wafer]:
        lifted = np.zeros(a.shape[0])
        for s, slot in enumerate(inst.stock_slots):
            if slot.k == k and ("lift", s) in v.tmpl:
                lifted += v.col(a, ("lift", s))
        out[f"lifted {v.K[k]}"] = lifted
    for oo, o in enumerate(inst.osats):
        for _kr, kp in inst.nodes[o].osat.packages.items():
            out[f"packaged {v.K[kp]} {v.node[o]}"] = v.col(a, ("xi", oo, kp))
    dem = np.asarray(marks.demand)
    for good in ("chip_le", "chip_mat"):
        served, want = np.zeros(a.shape[0]), np.zeros(a.shape[0])
        for do, dm in enumerate(inst.demands):
            if v.K[dm.k] == good:
                served += v.col(a, ("D", do))
                want += dem[:, do]
        out[f"served {good}"] = served / np.maximum(want, 1.0)
    disp = defaultdict(lambda: np.zeros(a.shape[0]))
    for s, slot in enumerate(inst.stock_slots):
        if ("O", s) in v.tmpl:
            disp[v.K[slot.k]] += v.col(a, ("O", s))
    for good, x in disp.items():
        out[f"disposed {good}"] = x
    return out
