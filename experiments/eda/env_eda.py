"""EDA of the ShockBench-Flow scenario generator: what disruptions happen, and what the agent can see coming.

    uv run python experiments/eda/env_eda.py                              # Small, 120 episodes of root 12345
    uv run python experiments/eda/env_eda.py --task=full --episodes=30    # Full

No agent is played: each episode's scenario (omega), its weekly marks (what the network really is each week) and the
`standard` information view (warnings, messages, forecast, as the agent sees them) are built straight from the
package, so the run is cheap (about 3 s per Small episode). It reads the hidden truth (which messages are decoys,
when events start) to measure how informative the visible signals are.

Files: outputs/eda/<task>_<date_time>/summary.json and events.csv.
"""

import csv
import json
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import fire
import numpy as np
from shockbench_flow.disruption import harm
from shockbench_flow.hosting.tasks import scenario, task_generator
from shockbench_flow.information.theta import resolve_regime
from shockbench_flow.information.view import build_view
from shockbench_flow.marks import compute_marks, read_events
from shockbench_flow.omega import codes


ROOT = Path(__file__).resolve().parents[2]
HORIZONS = (1, 2, 4, 8)
MILITARY = (3, 4, 5, 6)  # militarised closure, regional conflict, piracy, energy shock
CLOSURES = (codes.EVENT_TYPES.index("militarised_closure"), codes.EVENT_TYPES.index("weather_closure"))


def auroc(score, label):
    """Area under the ROC curve (rank formula); None when a class is empty."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return None
    order = score.argsort(kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    for v in np.unique(score[np.r_[False, np.diff(score[order]) == 0]] if len(score) > 1 else []):
        tie = score == v
        ranks[tie] = ranks[tie].mean()
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def target_name(inst, kind, target):
    name = codes.TARGET_KINDS[kind]
    if name in ("chokepoint", "node"):
        return inst.nodes[target].id
    if name == "edge":
        return inst.edges[target].id
    return inst.regions[target]


def one_episode(args):
    task, entropy, n = args
    inst, params = task_generator(task)
    theta = resolve_regime("standard")
    om = scenario(task, n, entropy=entropy)
    mk = compute_marks(inst, om)
    view = build_view(inst, om, theta, marks=mk)
    T = inst.T
    events = read_events(inst, om.arrays)
    h_ev, H, h_type = harm.totals(inst, om)
    lead = np.asarray(om.arrays["ev_lead"])

    ev_rows = []
    for e, h in zip(events, h_ev):
        chans = [codes.CHANNELS[c] for c in range(lead.shape[1]) if not np.isnan(lead[e.row, c])]
        leads = [float(lead[e.row, c]) for c in range(lead.shape[1]) if not np.isnan(lead[e.row, c])]
        ev_rows.append(
            dict(
                episode=n,
                type=codes.EVENT_TYPES[e.type],
                target_kind=codes.TARGET_KINDS[e.target_kind],
                target=target_name(inst, e.target_kind, e.target),
                region=e.region,
                commodity=-1 if e.commodity < 0 else inst.commodities[e.commodity].id,
                onset=e.onset,
                duration=e.duration,
                severity=e.severity,
                rate=e.rate,
                harm=float(h),
                channels="|".join(chans),
                max_lead=max(leads) if leads else None,
            )
        )

    # messages as shown (decoy flag is the hidden truth)
    msgs = []
    for a in view.messages:
        msgs.append(
            dict(
                thread=a.thread,
                channel=codes.CHANNELS[a.channel],
                kind=codes.MESSAGE_KINDS[a.kind],
                decoy=bool(a.decoy),
                instant=a.instant,
                effect=a.effect,
                stated=a.stated_effective_week,
                target_kind=codes.TARGET_KINDS[a.target_kind],
            )
        )

    # warnings against what happens next, per signal unit
    S = np.asarray(view.scores)  # (T, U)
    units = view.units
    chk_nodes = list(inst.chokepoints)
    warn = defaultdict(list)  # (unit kind, h) -> list of (score, label)
    onset_week = lambda e: int(np.floor(e.onset)) + 1  # noqa: E731
    for ui, u in enumerate(units):
        kind = u.kind
        if kind == "chokepoint":
            weeks = [onset_week(e) for e in events if e.type in CLOSURES and e.target == u.unit and e.onset >= 0]
        elif kind == "region":
            weeks = [onset_week(e) for e in events if e.region == u.unit and e.onset >= 0]
        else:  # dyad: any militarised event in either of its regions
            a, b = view.dyads[u.unit]
            weeks = [onset_week(e) for e in events if e.type in MILITARY and e.region in (a, b) and e.onset >= 0]
        for t in range(1, T + 1):
            for h in HORIZONS:
                lab = any(t <= w < t + h for w in weeks)
                warn[(kind, h)].append((float(S[t - 1, ui]), lab))

    # forecast error by horizon (relative)
    F = np.asarray(view.forecast)  # (T, D, 8)
    d = np.asarray(mk.demand)  # (T, D)
    ferr = {}
    for h in range(8):
        f = F[: T - h, :, h]
        r = d[h:, :]
        ok = np.isfinite(f) & (r > 0)
        ferr[h] = float(np.median(np.abs(f[ok] / r[ok] - 1))) if ok.any() else None
    dbar = np.array([dm.dbar for dm in inst.demands])
    demand_ratio = (d[:, dbar > 0] / dbar[dbar > 0]).ravel().tolist()

    # how the network really looks week by week
    o = np.asarray(mk.o)  # (T, C)
    u0 = np.array([np.nan if e.u0 is None else float(e.u0) for e in inst.edges])
    finite = np.isfinite(u0) & (u0 > 0)
    ucut = (np.asarray(mk.u)[:, finite] < 0.999 * u0[finite]).mean()
    prohib = np.asarray(mk.prohibited).sum(axis=(1, 2))
    prohib = (prohib - prohib.min()).tolist()  # pairs above the episode's calmest week
    tariff = (np.asarray(mk.tariff) > 0).sum(axis=(1, 2)).tolist()
    R = np.asarray(mk.R)
    Gb = np.asarray(mk.G_bar)
    Gb0 = np.array([inst.nodes[g].grid.deliverable for g in inst.grids])
    return dict(
        episode=n,
        harm=float(H),
        harm_by_type={codes.EVENT_TYPES[k]: float(v) for k, v in h_type.items()},
        events=ev_rows,
        messages=msgs,
        warn={f"{k}|{h}": v for (k, h), v in warn.items()},
        ferr=ferr,
        demand_ratio_q=np.quantile(demand_ratio, [0.05, 0.5, 0.95]).tolist(),
        open_closed_weeks={inst.nodes[c].id: int((o[:, i] < 0.999).sum()) for i, c in enumerate(chk_nodes)},
        open_min={inst.nodes[c].id: float(o[:, i].min()) for i, c in enumerate(chk_nodes)},
        edge_cut_share=float(ucut),
        prohib_pairs_mean=float(np.mean(prohib)),
        tariff_pairs_mean=float(np.mean(tariff)),
        fab_R_min=R.min(axis=0).tolist(),
        grid_Gbar_min=(Gb.min(axis=0) / Gb0).tolist(),
        blackout=[int(x) for x in np.asarray(om.arrays["blk_start"])],
    )


def summarise(rows, inst):
    T = inst.T
    out = {}
    N = len(rows)
    ev = [e for r in rows for e in r["events"]]
    out["episodes"] = N
    # events by type
    by = defaultdict(list)
    for e in ev:
        by[e["type"]].append(e)
    tab = {}
    for ty, es in sorted(by.items(), key=lambda kv: -sum(e["harm"] for e in kv[1])):
        inep = [e for e in es if e["onset"] >= 0]
        tab[ty] = dict(
            per_episode=len(es) / N,
            started_before_episode=sum(e["onset"] < 0 for e in es) / len(es),
            duration_weeks_median=float(np.median([e["duration"] for e in es])),
            duration_weeks_p90=float(np.quantile([e["duration"] for e in es], 0.9)),
            severity_median=float(np.median([e["severity"] for e in es])),
            harm_share=sum(e["harm"] for e in es) / sum(r["harm"] for r in rows),
            announced_share=sum(bool(e["channels"]) for e in inep) / max(1, len(inep)),
            lead_weeks_median=float(np.median([e["max_lead"] for e in inep if e["max_lead"] is not None]))
            if any(e["max_lead"] is not None for e in inep)
            else None,
            top_targets=Counter(e["target"] for e in es).most_common(6),
        )
    out["event_types"] = tab
    # clustering (Hawkes): P(another event within 4 weeks after an event) vs base rate
    clus = []
    for r in rows:
        on = sorted(e["onset"] for e in r["events"] if 0 <= e["onset"] < T - 4)
        allon = np.array(sorted(e["onset"] for e in r["events"] if e["onset"] >= 0))
        for x in on:
            clus.append(((allon > x) & (allon <= x + 4)).sum())
    weekly_rate = np.mean([sum(0 <= e["onset"] < T for e in r["events"]) / T for r in rows])
    out["clustering"] = dict(
        events_in_next_4_weeks_after_an_event=float(np.mean(clus)) if clus else None,
        events_in_any_4_weeks=float(4 * weekly_rate),
    )
    # harm distribution
    hs = np.array([r["harm"] for r in rows])
    out["harm_usd_quantiles"] = dict(
        zip(["p10", "p50", "p75", "p90", "p95", "p99"], np.quantile(hs, [0.1, 0.5, 0.75, 0.9, 0.95, 0.99]).tolist())
    )
    out["harm_top10pct_share"] = float(np.sort(hs)[::-1][: max(1, N // 10)].sum() / hs.sum())
    # messages
    m = [x for r in rows for x in r["messages"]]
    first = {}
    for r in rows:
        for x in r["messages"]:
            key = (r["episode"], x["thread"])
            if key not in first or x["instant"] < first[key]["instant"]:
                first[key] = x
    chan = defaultdict(lambda: [0, 0])
    for x in first.values():
        chan[x["channel"]][0] += 1
        chan[x["channel"]][1] += x["decoy"]
    out["message_threads"] = {
        c: dict(threads_per_episode=a / N, decoy_share=b / a) for c, (a, b) in sorted(chan.items())
    }
    kinds = defaultdict(list)
    for x in m:
        if x["kind"] != "withdrawal":
            kinds[(x["channel"], x["kind"])].append(x["effect"] - x["instant"])
    out["message_lead_weeks"] = {
        f"{c}/{k}": dict(
            n=len(v), median=float(np.median(v)), p10=float(np.quantile(v, 0.1)), p90=float(np.quantile(v, 0.9))
        )
        for (c, k), v in sorted(kinds.items())
    }
    # second notice: does a thread with a final notice / legal publication ever turn out to be a decoy?
    thr = defaultdict(list)
    for r in rows:
        for x in r["messages"]:
            thr[(r["episode"], x["thread"])].append(x)
    conf = defaultdict(lambda: [0, 0])
    for msgs in thr.values():
        kinds_ = {x["kind"] for x in msgs}
        dec = msgs[0]["decoy"]
        tag = "with second notice" if ("final_notice" in kinds_ or "publication" in kinds_) else "first notice only"
        conf[(msgs[0]["channel"] if msgs[0]["kind"] != "withdrawal" else "?", tag)][0] += 1
        conf[(msgs[0]["channel"] if msgs[0]["kind"] != "withdrawal" else "?", tag)][1] += dec
    out["decoy_share_by_second_notice"] = {
        f"{c}/{t}": dict(threads=a, decoy_share=b / a) for (c, t), (a, b) in sorted(conf.items())
    }
    # warnings
    wk = defaultdict(list)
    for r in rows:
        for key, v in r["warn"].items():
            wk[key] += v
    out["warning_auroc"] = {}
    for key, v in sorted(wk.items()):
        s, lab = zip(*v)
        out["warning_auroc"][key] = dict(auroc=auroc(s, lab), base_rate=float(np.mean(lab)), n=len(lab))
    # forecast
    out["forecast_median_abs_rel_error_by_h"] = {
        h: float(np.median([r["ferr"][h] for r in rows if r["ferr"][h] is not None])) for h in range(8)
    }
    out["demand_over_dbar_q05_q50_q95"] = np.median([r["demand_ratio_q"] for r in rows], axis=0).tolist()
    # network state
    chk = rows[0]["open_closed_weeks"].keys()
    out["chokepoints"] = {
        c: dict(
            episodes_with_closure=float(np.mean([r["open_closed_weeks"][c] > 0 for r in rows])),
            mean_weeks_impaired=float(np.mean([r["open_closed_weeks"][c] for r in rows])),
            min_open_median_when_hit=float(np.median([r["open_min"][c] for r in rows if r["open_closed_weeks"][c] > 0]))
            if any(r["open_closed_weeks"][c] > 0 for r in rows)
            else None,
        )
        for c in chk
    }
    out["edge_week_share_with_cut_capacity"] = float(np.mean([r["edge_cut_share"] for r in rows]))
    out["extra_prohibited_pairs_per_week"] = float(np.mean([r["prohib_pairs_mean"] for r in rows]))
    out["tariffed_pairs_per_week"] = float(np.mean([r["tariff_pairs_mean"] for r in rows]))
    fr = np.array([r["fab_R_min"] for r in rows])
    out["fab_episodes_with_outage"] = (fr < 0.999).mean(axis=0).tolist() if fr.size else []
    gr = np.array([r["grid_Gbar_min"] for r in rows])
    out["grid_episodes_with_energy_shock"] = (gr < 0.999).mean(axis=0).tolist() if gr.size else []
    out["grids"] = [inst.nodes[g].id for g in inst.grids]
    out["fabs"] = [inst.nodes[f].id for f in inst.fabs]
    return out


def main(task: str = "small", episodes: int = 120, entropy: int = 12345, workers: int = 4):
    inst, _ = task_generator(task)
    t0 = time.time()
    with Pool(workers) as pool:
        rows = pool.map(one_episode, [(task, entropy, n) for n in range(episodes)], chunksize=2)
    print(f"{episodes} episodes in {time.time() - t0:.0f} s")
    out_dir = ROOT / "outputs" / "eda" / f"{task}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarise(rows, inst)
    summary.update(task=task, entropy=entropy)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    with open(out_dir / "events.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]["events"][0].keys()))
        w.writeheader()
        for r in rows:
            w.writerows(r["events"])
    print(json.dumps(summary, indent=1, default=str))
    print(f"written to {out_dir}")


if __name__ == "__main__":
    fire.Fire(main)
