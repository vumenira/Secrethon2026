"""The chip chain of one episode in blocks of weeks, played | optimised: fabs, OSATs, their outlets, sinks.

For every edge that carries chips: the flow against the edge's true capacity; for every OSAT: starts against
throughput, packaged stock against storage, disposal; for every sink: served against demand.

uv run python chain.py --task=small --n=8 --name=cal --block=4
"""

import fire
import numpy as np

import common as C
import describe as Dsc


def main(task="small", n=0, name="cal", block=4, goods="chip"):
    cal = C.Cal(task, int(n))
    inst, marks, T = cal.inst, cal.marks, cal.T
    d = C.load_traj(task, int(n))
    a = d["zpi"].reshape(T, cal.nc)
    b, _keys, _z = Dsc.load_sol(task, int(n), name)
    v = Dsc.View(cal)
    K = inst.commodities
    sel = {k for k, c in enumerate(K) if c.id.startswith(goods)}
    B = [(s, min(s + block, T)) for s in range(0, T, block)]
    u = np.asarray(marks.u)
    pr = np.asarray(marks.prohibited)
    o = np.asarray(marks.o)
    chk = set(inst.chokepoints)

    def row(label, xa, xb, cap=None, scale=1e3):
        line = f"  {label:46s}"
        for s, e in B:
            line += f" {xa[s:e].sum() / scale:6.0f}|{xb[s:e].sum() / scale:<6.0f}"
            if cap is not None:
                line += f"/{cap[s:e].sum() / scale:<6.0f}"
        print(line)

    print(f"{task} episode {n}: thousands of units per block of {block} weeks, played|optimised ({name}); blocks start at weeks " + " ".join(str(s + 1) for s, _e in B))
    print("fabs: lots started / capacity; raw chips at the fab at the block's end; disposed")
    for fo, f in enumerate(inst.fabs):
        fa = inst.nodes[f].fab
        cap = np.asarray(marks.alpha_bar)[:, fo] * np.asarray(marks.R)[:, fo] * fa.cap0
        row(f"{v.node[f]} lots/cap", v.col(a, ("p", fo)), v.col(b, ("p", fo)), cap)
        s = inst.slot_index[(f, fa.product)]
        sa, sb = v.col(a, ("I", s)), v.col(b, ("I", s))
        print(f"  {v.node[f] + ' raw stock at block end (storage ' + format(inst.stock_slots[s].storage / 1e3, '.0f') + ')':46s}" + "".join(f" {sa[e - 1] / 1e3:6.0f}|{sb[e - 1] / 1e3:<6.0f}" for _s, e in B))
        row(f"{v.node[f]} raw disposed", v.col(a, ("O", s)), v.col(b, ("O", s)))
        sw = inst.slot_index[(f, fa.input)]
        wa, wb = v.col(a, ("I", sw)), v.col(b, ("I", sw))
        print(f"  {v.node[f] + ' wafers at block end':46s}" + "".join(f" {wa[e - 1] / 1e3:6.0f}|{wb[e - 1] / 1e3:<6.0f}" for _s, e in B))
    print("osats: starts / throughput; packaged stock at the block's end; disposed")
    thr = None
    from shockbench_flow.marks import osat_throughput

    thr = osat_throughput(inst, marks.R_osat)
    for oo, o_ in enumerate(inst.osats):
        oa = inst.nodes[o_].osat
        xa = sum(v.col(a, ("xi", oo, kp)) for kp in oa.packages.values())
        xb = sum(v.col(b, ("xi", oo, kp)) for kp in oa.packages.values())
        row(f"{v.node[o_]} starts/throughput", xa, xb, thr[:, oo])
        for kr, kp in oa.packages.items():
            s = inst.slot_index[(o_, kp)]
            sa, sb = v.col(a, ("I", s)), v.col(b, ("I", s))
            print(f"  {v.node[o_] + ' ' + K[kp].id + ' stock at block end (storage ' + format(inst.stock_slots[s].storage / 1e3, '.0f') + ')':46s}" + "".join(f" {sa[e - 1] / 1e3:6.0f}|{sb[e - 1] / 1e3:<6.0f}" for _s, e in B))
            row(f"{v.node[o_]} {K[kp].id} disposed", v.col(a, ("O", s)), v.col(b, ("O", s)))
    print("edges that carry chips: flow / the edge's capacity (all goods); weeks prohibited for the good; a strait at the head: its lowest open share")
    for (e, k), _cols in sorted(v.flows.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if k not in sel:
            continue
        ed = inst.edges[e]
        fa_, fb_ = v.flow(a, e, k), v.flow(b, e, k)
        if fa_.sum() + fb_.sum() < 1:
            continue
        note = ""
        if pr[:, e, k].any():
            note += f" prohibited {Dsc.spans(pr[:, e, k])}"
        for end_ in (ed.tail, ed.head):
            if end_ in chk:
                ci = inst.chokepoint_ordinal[end_]
                if (o[:, ci] < 1 - 1e-9).any():
                    note += f" {v.node[end_]} open<1 in {Dsc.spans(o[:, ci] < 1 - 1e-9)} (lowest {o[:, ci].min():.2f})"
        row(f"{v.edge_name(e)} {K[k].id} (lead {ed.tau})", fa_, fb_, np.where(np.isfinite(u[:, e]), u[:, e], 0.0))
        if note:
            print(f"      {note}")
    print("sinks: served / demand")
    for do, dm in enumerate(inst.demands):
        dem = np.asarray(marks.demand)[:, do]
        if dem.sum() > 0 and dm.k in sel:
            row(f"{v.node[dm.node]} {K[dm.k].id} served/demand", v.col(a, ("D", do)), v.col(b, ("D", do)), dem)


if __name__ == "__main__":
    fire.Fire(main)
