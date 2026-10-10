"""Closure spells, military threats and what the warning scores track (Small, 120 episodes of root 12345).

    uv run python experiments/eda/signals.py

Follow-up to env_eda.py: how long a strait stays impaired, whether every militarised closure is announced by a MID
threat (and how early), and the warning scores against the latent risk, the conflict state and closure onsets.
Prints only; about 2 minutes with 4 workers.
"""

from multiprocessing import Pool

import numpy as np
from env_eda import auroc  # same folder
from shockbench_flow.hosting.tasks import scenario, task_generator
from shockbench_flow.information.theta import resolve_regime
from shockbench_flow.information.view import build_view
from shockbench_flow.marks import compute_marks, read_events
from shockbench_flow.omega import codes


def ep(n):
    inst, _ = task_generator("small")
    om = scenario("small", n, entropy=12345)
    mk = compute_marks(inst, om)
    view = build_view(inst, om, resolve_regime("standard"), marks=mk)
    T = inst.T
    B = int(om.arrays["meta_burn_in"])
    ev = read_events(inst, om.arrays)
    o = np.asarray(mk.o)
    spells = []
    for i, c in enumerate(inst.chokepoints):
        imp = o[:, i] < 0.999
        t = 0
        while t < T:
            if imp[t]:
                s = t
                while t < T and imp[t]:
                    t += 1
                spells.append(
                    dict(
                        chk=inst.nodes[c].id,
                        start=s + 1,
                        length=t - s,
                        censored=t == T,
                        minopen=float(o[s:t, i].min()),
                        mean_open=float(o[s:t, i].mean()),
                    )
                )
            else:
                t += 1
    # militarised closure onsets in episode and mid threats
    mids = [
        a
        for a in view.messages
        if codes.CHANNELS[a.channel] == "mid_threat" and codes.MESSAGE_KINDS[a.kind] == "threat"
    ]
    mil = []
    for e in ev:
        if e.type == 3 and 0 <= e.onset < T:
            pre = [a for a in mids if a.target == e.target and not a.decoy and abs(a.effect - e.onset) < 1e-9]
            mil.append(
                dict(
                    chk=inst.nodes[e.target].id,
                    onset=e.onset,
                    dur=e.duration,
                    sev=e.severity,
                    threat_lead=(e.onset - pre[0].instant) if pre else None,
                )
            )
    # warnings: chokepoint vs militarised onset only, region vs own conflict state next week
    S = np.asarray(view.scores)
    X = np.asarray(om.arrays["X"])
    zown = np.asarray(om.arrays["z_c_own"])
    wc, wr, wx = [], [], []
    for ui, u in enumerate(view.units):
        for t in range(1, T + 1):
            if u.kind == "chokepoint":
                lab = any(e.type == 3 and e.target == u.unit and t <= int(np.floor(e.onset)) + 1 < t + 8 for e in ev)
                wc.append((S[t - 1, ui], lab))
            if u.kind == "region":
                wr.append((S[t - 1, ui], zown[u.unit, t + B] > 0))
            wx.append((S[t - 1, ui], X[u.row, t - 1 + B]))
    return dict(spells=spells, mil=mil, wc=wc, wr=wr, wx=wx)


if __name__ == "__main__":
    with Pool(4) as p:
        R = p.map(ep, range(120), chunksize=3)
    sp = [s for r in R for s in r["spells"]]
    print("closure spells:", len(sp), "per episode", len(sp) / 120)
    L = np.array([s["length"] for s in sp])
    st = np.array([s["start"] for s in sp])
    cen = np.array([s["censored"] for s in sp])
    print("spell length quantiles 25/50/75/90:", np.quantile(L, [0.25, 0.5, 0.75, 0.9]))
    print("share starting week 1:", (st == 1).mean(), " share lasting to episode end:", cen.mean())
    print("length 1-2 weeks share:", (L <= 2).mean())
    for c in sorted({s["chk"] for s in sp}):
        ss = [s for s in sp if s["chk"] == c]
        lens = [s["length"] for s in ss]
        print(
            f"  {c:12s} spells {len(ss):3d} median len {np.median(lens):5.1f} long(>8w) {sum(x > 8 for x in lens):3d}"
            f" mean open in spell {np.mean([s['mean_open'] for s in ss]):.2f}"
        )
    mil = [m for r in R for m in r["mil"]]
    print(
        "in-episode militarised closures:",
        len(mil),
        "with a real MID threat before:",
        sum(m["threat_lead"] is not None for m in mil),
    )
    tl = [m["threat_lead"] for m in mil if m["threat_lead"] is not None]
    if tl:
        print(
            "  threat lead weeks quantiles 10/50/90:",
            np.quantile(tl, [0.1, 0.5, 0.9]),
            " share lead>=1 week:",
            np.mean(np.array(tl) >= 1),
        )
    print(
        "  durations (weeks) of in-episode militarised closures q25/50/75:",
        np.quantile([m["dur"] for m in mil], [0.25, 0.5, 0.75]),
    )
    s, lab = zip(*[x for r in R for x in r["wc"]])
    print("chokepoint warning -> militarised onset within 8w AUROC", auroc(s, lab), "base", np.mean(lab))
    s, lab = zip(*[x for r in R for x in r["wr"]])
    print("region warning -> region in own conflict (minor/war) AUROC", auroc(s, lab), "base", np.mean(lab))
    s, x = zip(*[x for r in R for x in r["wx"]])
    print("corr(warning, latent X of week t-1)", np.corrcoef(s, x)[0, 1])
