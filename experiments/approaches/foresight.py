"""What each kind of knowledge about the future would be worth to the rolling LP: ceilings, not candidates.

uv run python experiments/approaches/foresight.py run --names=fs_all,fs_all_h24      # play the named variants
uv run python experiments/approaches/foresight.py run                                # every variant not played yet
uv run python experiments/approaches/foresight.py report                             # the fs_ rows against the base
uv run python experiments/approaches/foresight.py same --a=fs_k0 --b=mpc_det_h20     # equal episode for episode?

Each variant plays the probe ``fs_probe`` (``mpc_det`` with a hook; the probes and the planners named below are folders
of ``experiments/pulse_agent/versions/``) on the episodes ``tune.py`` uses, with the planning window of every
week filled from the scenario's own marks, the arrays the simulator itself reads. A variant names the
families fed (``fields``), how far ahead (``lead``, weeks; absent: the whole window), the window (``h``, weeks) and
whether demand (``demand``) or the decoys among the pending sanctions (``decoys``) are known. Everything not fed is
planned as ``mpc_det`` plans it: the network as observed this week, held for the whole window.

Timing. The observation of week t shows the network at the instant t - 1; week t's own average is already the future
(an event may start inside the week). So ``lead`` k means: the true week averages of weeks t .. t + k - 1, and from
week t + k on the network as it stands at the instant t - 1 + k, held (what the observation of week t + k will show).
k = 0 is ``mpc_det`` itself, and ``fs_k0`` checks that the hook then changes nothing.

``mode`` "recover" feeds no onsets: an element that is off its nominal value now follows its true values until it is
back to nominal and stays nominal after, as ``tune.truth_series`` does; ``fs_rec_grid`` reproduces ``truth_grid``.
``mode`` "onset" is the opposite, a perfect early warning: of what ``lead`` would tell, only what is worse than the
network as observed now is kept, so nothing seen now is told to end (an event that starts in the known weeks is
told whole, its own end included, when that falls in the known weeks). An event that outlasts the known weeks is
then held to the window's end, as ``mpc_det`` holds whatever it sees. ``mode`` "warn" is the early warning for a
planner that knows durations: as "recover", and every element that goes off nominal within ``lead`` weeks follows
its truth too, until it is back; its difference to "recover" is what the warning adds.

``base_first`` is not foresight. The clairvoyant plan (score 1) may shed a grid's base load to power its fabs; the
simulator serves base load first, and ``mpc_det`` prices that rule in the first window week only, so its later weeks
plan lots the simulator will not start. The switch prices it in every window week (``BASE_FIRST_EVERY_WEEK`` in the
probe's ``sbflow/oracle/lp.py``), to see how much of what full foresight leaves is this one rule.

Results are cached by name beside ``tune.py``'s, in the same layout, so ``tune.py report`` lists them too.
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import fire
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
import tune  # noqa: E402 - the harness beside this file: its run directory, episode set and memory guard


PROBE = tune.VERSIONS / "fs_probe"  # agents/mpc_det_h20 with the hook
PROBE_EV = tune.VERSIONS / "fs_probe_ev"  # ft_buffer with the hook (variants with "probe": "ev")
AGENT_V2 = tune.VERSIONS / "ft_buffer_v2"  # the pulse planner as it is, never edited (variants "probe": "v2")
MODULE = "submission_foresight"
# family -> the window arrays it owns (every array the LP reads from the window, demand apart)
FAMILIES = {
    "closure": ("o", "kappa", "wr_class", "h_queue", "c_wr"),  # a strait's open fraction, throughput, war-risk costs
    "capacity": ("u",),  # port strikes, a war's cuts, a sanction's friendly fire
    "grid": ("G_bar", "y_bar"),
    "sanction": ("prohibited",),
    "tariff": ("tariff",),
    "supply": ("supply",),
    "fab": ("R", "alpha_bar", "R_osat", "sigma_scr"),  # fab and OSAT restoration, the fab ceiling, scrap and its hits
    "freight": ("c",),  # piracy surcharges
}
# the array of the marks that holds a window array's value at the instant t - 1 (row t - 1); absent: the week's own row
INSTANT = {
    "u": "u_now",
    "o": "o_now",
    "kappa": "kappa_now",
    "supply": "supply_now",
    "G_bar": "G_bar_now",
    "y_bar": "y_bar_now",
    "R": "R_now",
    "alpha_bar": "alpha_now",
}
WORSE_UP = ("tariff", "c", "wr_class", "h_queue", "c_wr")  # arrays where a larger value is the worse one
VARIANTS = {
    # ----- wiring checks
    "fs_k0": {"fields": "all", "lead": 0, "told": "nothing (the hook with a lead of 0 weeks): must equal mpc_det_h20"},
    "fs_rec_grid": {
        "fields": ["grid"],
        "mode": "recover",
        "told": "true recovery of visible grid shortfalls: must equal truth_grid",
    },
    "fs_all_dem_h52": {
        "fields": "all",
        "demand": True,
        "h": 52,
        "told": "everything to the episode's end, demand too (the clairvoyant plan, replanned weekly)",
    },
    # ----- A: the whole window
    "fs_all": {"fields": "all", "told": "the true network in all 20 window weeks"},
    "fs_all_h24": {"fields": "all", "h": 24, "told": "the true network in all 24 window weeks"},
    "fs_rec_all": {
        "fields": "all",
        "mode": "recover",
        "told": "true recovery of everything off nominal now, no onsets (20 weeks)",
    },
    "fs_all_dem": {"fields": "all", "demand": True, "told": "the true network and the true demand, 20 weeks"},
    "fs_all_h52": {"fields": "all", "h": 52, "told": "the true network to the episode's end, forecast demand"},
    # ----- B: one family at a time, 20 weeks
    "fs_closure": {"fields": ["closure"], "told": "straits: open fraction, throughput, war-risk class"},
    "fs_capacity": {"fields": ["capacity"], "told": "edge capacity (strikes, wars, friendly fire)"},
    "fs_grid": {"fields": ["grid"], "told": "grid power"},
    "fs_sanction": {"fields": ["sanction"], "told": "prohibitions (sanctions), decoys resolved"},
    "fs_tariff": {"fields": ["tariff"], "told": "tariffs"},
    "fs_supply": {"fields": ["supply"], "told": "supply caps"},
    "fs_fab": {"fields": ["fab"], "told": "fab and OSAT restoration, fab ceiling, scrap"},
    "fs_freight": {"fields": ["freight"], "told": "freight rates (piracy)"},
    # ----- C: by lead time, every family
    "fs_all_k1": {"fields": "all", "lead": 1, "told": "the true network 1 week ahead, then held"},
    "fs_all_k2": {"fields": "all", "lead": 2, "told": "the true network 2 weeks ahead, then held"},
    "fs_all_k4": {"fields": "all", "lead": 4, "told": "the true network 4 weeks ahead, then held"},
    "fs_all_k8": {"fields": "all", "lead": 8, "told": "the true network 8 weeks ahead, then held"},
    # ----- D and E
    "fs_demand": {"fields": [], "demand": True, "told": "the true demand in the window, network as observed"},
    "fs_nodecoy": {"fields": [], "decoys": True, "told": "pending sanctions without the ones that never take effect"},
    # ----- not foresight: the simulator's rule "base load before fabs" priced in every window week, not only the first
    "fs_bf_k0": {"fields": [], "base_first": True, "told": "nothing; the plan never sheds base load to power a fab"},
    "fs_bf_all_dem_h52": {
        "fields": "all",
        "demand": True,
        "h": 52,
        "base_first": True,
        "told": "everything to the episode's end; the plan never sheds base load to power a fab",
    },
}
# ----- the same on top of other planners: fs_probe_ev (ft_buffer with the hook), "params" its PARAMS
EV = {"end_value": 1.0}  # mpc_buffer's end-value planner: the rows f_e10 (Full) and t3a_e10 (Small)
PQ = {  # the pulse planner of the row ft_q_e20_fd8 (Full)
    "end_value": 2.0,
    "reserve_weeks": 8.0,
    "reserve_feeders": True,
    "reserve_price": 0.001,
    "reserve_fuels": "all",
    "shed_ramp": 0.002,
    "pulse": True,
    "pulse_run": 4,
    "pulse_hold_weeks": 12,
    "pulse_partial": 0.5,
    "pulse_payback": 12,
    "short_premium": 1.0,
}
for _tag, _params, _on in (("ev", EV, "end-value planner"), ("pq", PQ, "pulse planner")):
    _base = {"probe": "ev", "params": _params}
    VARIANTS[f"fs_{_tag}_k0"] = _base | {"fields": "all", "lead": 0, "told": f"{_on}: nothing (must equal its row)"}
    VARIANTS[f"fs_{_tag}_all"] = _base | {"fields": "all", "told": f"{_on}: the true network, 20 window weeks"}
    VARIANTS[f"fs_{_tag}_rec_all"] = _base | {
        "fields": "all",
        "mode": "recover",
        "told": f"{_on}: true recovery of everything off nominal now, no onsets",
    }
    VARIANTS[f"fs_{_tag}_all_dem_h104"] = _base | {
        "fields": "all",
        "demand": True,
        "h": 104,
        "told": f"{_on}: everything to the episode's end, demand too",
    }
    for _h in (24, 32):
        VARIANTS[f"fs_{_tag}_h{_h}"] = _base | {"fields": [], "h": _h, "told": f"{_on}: nothing, {_h} window weeks"}
        VARIANTS[f"fs_{_tag}_all_h{_h}"] = _base | {
            "fields": "all",
            "h": _h,
            "told": f"{_on}: the true network, {_h} window weeks",
        }
# ----- ft_buffer_v2 as it is (its params.json), with one setting changed: quick tests of the pulse analysis
V2 = {"probe": "v2", "fields": [], "lead": 0}
VARIANTS["fs_v2_k0"] = V2 | {"params": {}, "told": "pulse planner v2 as it is (must equal ft_v2 / ft_s_v2)"}
VARIANTS["fs_v2_fill100"] = V2 | {"params": {"pulse_fill": 1.0}, "told": "v2, fuel held up to the whole tank"}
VARIANTS["fs_v2_run2"] = V2 | {"params": {"pulse_run": 2}, "told": "v2, a gas charge once 2 whole weeks are covered"}
VARIANTS["fs_v2_partial100"] = V2 | {"params": {"pulse_partial": 1.0}, "told": "v2, a pulse starts on any shortfall"}
VARIANTS["fs_v2_payback0"] = V2 | {"params": {"pulse_payback": 0}, "told": "v2, every fuel with a terminal may be held"}
VARIANTS["fs_v2_hold24"] = V2 | {"params": {"pulse_hold_weeks": 24}, "told": "v2, a tank may take 24 weeks to fill"}
VARIANTS["fs_v2_mem12"] = V2 | {"params": {"pulse_memory": 12}, "told": "v2, a grid counts as pulsed for 12 weeks"}
VARIANTS["fs_v2_resrat"] = V2 | {
    "params": {"reserve_fuels": "rationed"},
    "told": "v2, a reserve of gas only, not crude",
}
for _k in (None, 1, 2, 4, 8):  # C: a perfect early warning, every family (onsets only, no recovery of what is seen)
    VARIANTS["fs_on_all" + ("" if _k is None else f"_k{_k}")] = {
        "fields": "all",
        "mode": "onset",
        "lead": _k,
        "told": "every onset "
        + ("in the 20 window weeks" if _k is None else f"{_k} week(s) ahead")
        + ", no recoveries",
    }
for _family in ("closure", "capacity", "grid", "sanction"):  # a perfect early warning of one family's onsets
    for _k in (None, 2):
        VARIANTS[f"fs_on_{_family}" + ("" if _k is None else f"_k{_k}")] = {
            "fields": [_family],
            "mode": "onset",
            "lead": _k,
            "told": f"{_family}: every onset "
            + ("in the 20 window weeks" if _k is None else f"{_k} week(s) ahead")
            + ", no recoveries",
        }
for _family in ("all", "closure", "capacity"):  # a perfect early warning for a planner that knows every duration
    _fields = "all" if _family == "all" else [_family]
    if _family != "all":
        VARIANTS[f"fs_rec_{_family}"] = {
            "fields": _fields,
            "mode": "recover",
            "told": f"{_family}: true recovery of everything off nominal now, no onsets",
        }
    for _k in (1, 2, 4, 8):
        VARIANTS[f"fs_warn_{_family}_k{_k}"] = {
            "fields": _fields,
            "mode": "warn",
            "lead": _k,
            "told": f"{_family}: as fs_rec, and every event that starts within {_k} week(s), told whole",
        }
for _family in ("closure", "capacity", "grid", "sanction", "fab"):  # C: by lead time, one family
    for _k in (1, 2, 4, 8):
        VARIANTS[f"fs_{_family}_k{_k}"] = {
            "fields": [_family],
            "lead": _k,
            "told": f"{_family} {_k} week(s) ahead, then held",
        }


def arrays_of(fields):
    """The window arrays of a variant's ``fields`` ("all" or a list of families)."""
    families = list(FAMILIES) if fields == "all" else list(fields)
    return [name for family in families for name in FAMILIES[family]]


def truth_of(inst, omega, marks):
    """The scenario's ground truth as the windows need it: week averages, values at each instant, nominal values."""
    from shockbench_flow.marks import event_free_marks, mark_params_from_json, osat_restoration_now, read_events

    names = [name for name in arrays_of("all") if name != "sigma_scr"]
    events = read_events(inst, omega.arrays)
    osat_now = osat_restoration_now(inst, events, mark_params_from_json(str(omega.arrays["meta_mark_params"])))
    calm = event_free_marks(inst)
    instant = {name: np.asarray(getattr(marks, INSTANT.get(name, name))) for name in names}
    instant["R_osat"] = np.asarray(osat_now)  # the observation's graph_now.osat.R: not a field of the marks
    return SimpleNamespace(
        T=int(marks.T),
        avg={name: np.asarray(getattr(marks, name)) for name in names},
        instant=instant,
        nominal={name: np.asarray(getattr(calm, name))[0] for name in names},
        demand=np.asarray(marks.demand),
        hits=[(int(h.fab), float(h.onset), float(h.severity), int(h.onset_week)) for h in marks.fab_hits],
        w_scr=tuple(marks.w_scr),
    )


def same(a, b):
    """Elementwise: a equals b up to rounding (an infinite capacity equals itself)."""
    if a.dtype == bool:
        return a == b
    with np.errstate(invalid="ignore"):
        return (a == b) | (np.abs(a - b) <= 1e-9 * np.maximum(1.0, np.abs(b)))


def scrap(hits, w_scr, H):
    """(H, F) the scrap share of lots started in each window week under ``hits`` (fab, onset in window weeks, share)."""
    keep = np.ones((H, len(w_scr)))
    for fab, onset, severity in hits:
        week = math.floor(onset) + 1
        for h in range(H):  # lots started in window week h + 1 are hit in (h + 1, h + 1 + w_scr]
            if h + 1 < week <= h + 1 + w_scr[fab]:
                keep[h, fab] *= 1.0 - severity
    return 1.0 - keep


class Window:
    """The probe's hook of one episode: the planner's window arrays of week t under a variant."""

    def __init__(self, truth, spec):
        self.truth, self.spec = truth, spec

    def __call__(self, planner, obs, H, L):
        tr, spec = self.truth, self.spec
        inst, memory, t = planner._inst, planner._memory, int(obs["week"])
        if spec.get("decoys"):  # E: a pending sanction that never takes effect is dropped as soon as it is listed
            real = [(e, k, w) for e, k, w in memory.pending if w <= tr.T and tr.avg["prohibited"][w - 1, e, k]]
            planner.stats["pending_real"] = planner.stats.get("pending_real", 0) + len(real)
            planner.stats["pending_decoy"] = planner.stats.get("pending_decoy", 0) + len(memory.pending) - len(real)
            memory.pending = real
        out = {name: a.copy() for name, a in L.persistence_arrays(inst, obs, memory, H).items()}
        names = arrays_of(spec.get("fields", []))
        hits = []
        if spec.get("mode") in ("recover", "warn"):
            n = min(int(spec.get("lead") or 0), H) if spec["mode"] == "warn" else 0  # weeks of warning of an onset
            for name in names:
                if name == "sigma_scr":
                    if n:
                        hits = [(f, on - (t - 1), s) for f, on, s, week in tr.hits if on > t - 1 and week <= t + n - 1]
                        out[name] = scrap(hits, tr.w_scr, H)
                elif n:
                    self.warn(name, out[name], t, H, n)
                else:
                    self.recover(name, out[name], t, H)
        else:
            n = H if spec.get("lead") is None else min(int(spec["lead"]), H)  # window weeks known exactly
            for name in names:
                if name == "sigma_scr":
                    hits = [(f, on - (t - 1), sev) for f, on, sev, week in tr.hits if on > t - 1 and week <= t + n - 1]
                    out[name] = scrap(hits, tr.w_scr, H)
                    continue
                a, held = out[name], out[name].copy()
                a[:n] = tr.avg[name][t - 1 : t - 1 + n]
                if n < H:
                    a[n:] = tr.instant[name][t - 1 + n]  # as it stands at the end of the known weeks, held
                    if name == "prohibited":  # and the announced sanctions due later, as mpc_det plans them
                        for e, k, w in memory.pending:
                            if n <= w - t < H:
                                a[w - t :, e, k] = True
                if spec.get("mode") == "onset" and name != "alpha_bar":  # no recovery of what is seen now is told
                    a[:] = (held | a) if a.dtype == bool else (np.maximum if name in WORSE_UP else np.minimum)(held, a)
        if spec.get("demand"):
            out["demand"] = tr.demand[t - 1 : t - 1 + H].copy()
        return out, hits

    def recover(self, name, a, t, H):
        """In place: every element off nominal now follows its truth until it is back, nominal after; no onsets."""
        tr = self.truth
        nominal = tr.nominal[name]
        off = ~same(tr.instant[name][t - 1], nominal)
        back = np.zeros(nominal.shape, dtype=bool)
        for h in range(H):
            a[h][off] = np.where(back, nominal, tr.avg[name][t - 1 + h])[off]
            week = t + h + 1  # the week whose observation shows the instant at the end of window week h
            end = tr.instant[name][week - 1] if week <= tr.T else tr.avg[name][t - 1 + h]
            back |= same(end, nominal)

    def warn(self, name, a, t, H, n):
        """In place, as ``recover``, for every element off nominal now or at some time in the next ``n`` weeks.

        Such an element follows its truth from now until it is back to nominal after having been off, nominal after:
        an event that starts within ``n`` weeks is told whole, its end included; a later one is not told.
        """
        tr = self.truth
        nominal, avg, instant = tr.nominal[name], tr.avg[name], tr.instant[name]
        seen = ~same(instant[t - 1], nominal)  # the element has been off nominal by now
        known = seen.copy()
        for h in range(n):
            known |= ~same(avg[t - 1 + h], nominal)
        if t - 1 + n < tr.T:
            known |= ~same(instant[t - 1 + n], nominal)
        back = np.zeros(nominal.shape, dtype=bool)
        for h in range(H):
            seen |= ~same(avg[t - 1 + h], nominal)
            a[h][known] = np.where(back, nominal, avg[t - 1 + h])[known]
            week = t + h + 1
            end = instant[week - 1] if week <= tr.T else avg[t - 1 + h]
            back |= seen & same(end, nominal)


def play(root, spec, episodes, variant):
    """As scoring._play, on the probe with its hook set from the scenario's marks."""
    from shockbench_flow.dynamics.env import rollout, took_fallback
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed, _world
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    task, entropy, regime, fq, cache = spec
    rows = []
    for n in episodes:
        inst, omega, marks, fallback = _world(task, entropy, n, fq, cache)
        truth = truth_of(inst, omega, marks)
        with without_secret_like():
            factory = load_agent_class(root, MODULE)
            module = sys.modules[factory.__module__]
            if variant.get("probe") in ("ev", "v2"):  # their own settings; any failure is raised (strict)
                module.PARAMS.update({**variant.get("params", {}), "h_weeks": variant.get("h", 20), "strict": True})
            else:
                module.Planner.h_weeks = variant.get("h", 20)
            module.Planner.foresight = Window(truth, variant)
            sys.modules["sbflow.oracle.lp"].BASE_FIRST_EVERY_WEEK = bool(variant.get("base_first"))
            shim = _metered_shim(factory, None)
            start = time.perf_counter()
            seed = _policy_seed(entropy, n, NO_ZIP_SHA256)
            traj = rollout(inst, shim, omega, regime, seed, marks=marks, fallback=fallback)
            seconds = time.perf_counter() - start
        planner = getattr(shim.agent, "planner", None)
        unload_agent()
        rows.append(
            {
                "episode": n,
                "J_policy_cents": traj.J_cents,
                "fallback_weeks": sum(took_fallback(r) for r in traj.records),
                "first_error": f"week {shim.errors[0][0]}: {shim.errors[0][1]}" if shim.errors else None,
                "forecast_failed": getattr(planner, "failed", None),  # weeks the hook raised: must be 0
                "planner_errors": getattr(shim.agent, "act_errors", None),  # weeks the maximum was sent: must be 0
                "pulse_failed": getattr(shim.agent, "pulse_failed", None),  # fs_probe_ev: weeks the pulse's code failed
                "lp_fallback_weeks": sum(s.fallback for s in planner.telemetry) if planner is not None else None,
                "stats": getattr(planner, "stats", None),
                # where the cost went (USD, (23)'s components over the episode) and what the fabs and grids did
                "parts": {
                    k: math.fsum(getattr(r.costs, k) for r in traj.records) for k in traj.records[0].costs.as_dict()
                },
                "salvage": traj.salvage,
                "lots": np.sum([r.lots_started for r in traj.records], axis=0).tolist(),  # per fab
                "shed": np.sum([r.shed for r in traj.records], axis=0).tolist(),  # per grid, energy
                "seconds": round(seconds, 3),
                "omega_hash": omega.hash,
            }
        )
    return rows


def chosen(names):
    if names is None:
        return list(VARIANTS)
    names = [names] if isinstance(names, str) else list(names)
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; the variants are {list(VARIANTS)}")
    return names


def run(names=None, task="small", entropy=12345, episodes=64, n_jobs=2, redo=False):
    """Play the variants ``names`` (a name or several, comma-separated; default: all) not played yet."""
    from joblib import Parallel, delayed

    es = tune.episode_set(task, entropy, episodes, n_jobs)
    out = tune.run_dir(task, entropy, episodes)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "references.json").exists():
        (out / "references.json").write_text(json.dumps(list(es.references), indent=1))
    refs = list(es.references)
    for name in chosen(names):
        target = out / f"{name}.json"
        if target.exists() and not redo:
            print(f"{name}: cached", flush=True)
            continue
        tune.wait_for_memory()
        start = time.perf_counter()
        spec = VARIANTS[name]
        ns = list(es.episodes)
        chunks = [c for c in np.array_split(np.array(ns), min(len(ns), n_jobs)) if c.size]
        root = str({"ev": PROBE_EV, "v2": AGENT_V2}.get(spec.get("probe"), PROBE))
        parts = Parallel(n_jobs=n_jobs)(delayed(play)(root, es._spec, [int(n) for n in c], spec) for c in chunks)
        rows = [r for part in parts for r in part]
        for ref, row in zip(refs, rows):
            if ref["omega_hash"] != row["omega_hash"]:
                raise RuntimeError(f"episode {ref['episode']}: the cached reference was computed on another scenario")
        tmp = out / f"{name}.tmp"
        tmp.write_text(json.dumps({"spec": spec, "rows": rows}, indent=1))
        os.replace(tmp, target)
        table = es.rss([r["J_policy_cents"] for r in rows])
        keys = ("fallback_weeks", "forecast_failed", "planner_errors", "pulse_failed", "lp_fallback_weeks")
        bad = {key: sum(r[key] or 0 for r in rows) for key in keys}
        bad["errors"] = [r["first_error"] for r in rows if r["first_error"]][:2]
        stats = {}
        for r in rows:
            for key, value in (r["stats"] or {}).items():
                stats[key] = stats.get(key, 0) + value
        print(
            f"{name}: score {table['rss']:.4f} by level {table['rss_by_stratum']} failures {bad} stats {stats}"
            f" slowest episode {max(r['seconds'] for r in rows):.0f} s ({time.perf_counter() - start:.0f} s)",
            flush=True,
        )
        if episodes <= 8:  # a small check run: each episode's own score
            for ref, r in zip(refs, rows):
                gap = ref["J_naive_cents"] - ref["J_oracle_cents"]
                share = (ref["J_naive_cents"] - r["J_policy_cents"]) / gap if gap else float("nan")
                print(f"   episode {r['episode']} level {ref['stratum']}: {share:.4f} of a gap of {gap / 1e11:.1f} bn")


def costs_of(out, name):
    return [r["J_policy_cents"] for r in json.loads((out / f"{name}.json").read_text())["rows"]]


def report(base="mpc_det_h20", also="t3b_h24,truth_all", task="small", entropy=12345, episodes=64, n_boot=4000):
    """The fs_ variants (and ``also``) against ``base``: score, paired difference, 90 % interval, episodes."""
    from shockbench_flow_agent.scoring import _boot_stats

    es = tune.episode_set(task, entropy, episodes, 1)
    out = tune.run_dir(task, entropy, episodes)
    refs = list(es.references)
    extra = [also] if isinstance(also, str) else list(also or [])
    extra = [n for part in extra for n in part.split(",") if (out / f"{n}.json").exists()]
    names = [base] + extra + [n for n in VARIANTS if (out / f"{n}.json").exists()]
    costs = {n: costs_of(out, n) for n in names}
    boots = dict(zip(names, _boot_stats(refs, [costs[n] for n in names], True, n_boot, 0)))
    base_rss = es.rss(costs[base])["rss"]
    strata = [r["stratum"] for r in refs]
    print(f"{task}, root {entropy}, {episodes} episodes; by harm level {[strata.count(s) for s in (1, 2, 3, 4)]}")
    head = ("Variant", "What the planner was told", "Score", "Diff to base", "90 % interval", "Excl. 0")
    head += ("Better / worse", "By level")
    print("| " + " | ".join(head) + " |")
    print("|" + " --- |" * len(head))
    for n in names:
        table = es.rss(costs[n])
        d = boots[n] - boots[base]
        lo, hi = np.quantile(d[np.isfinite(d)], [0.05, 0.95])
        delta = (np.array(costs[base], dtype=float) - np.array(costs[n], dtype=float)) / 1e11  # bn USD saved
        by = " ".join(f"{table['rss_by_stratum'].get(s, float('nan')):.3f}" for s in (1, 2, 3, 4))
        told = VARIANTS.get(n, {}).get("told", "")
        clear = "yes" if lo > 0 or hi < 0 else "no"
        print(
            f"| `{n}` | {told} | {table['rss']:.4f} | {table['rss'] - base_rss:+.4f} | {lo:+.4f} to {hi:+.4f} | {clear}"
            f" | {int((delta > 1e-6).sum())} / {int((delta < -1e-6).sum())} | {by} |"
        )


def same_costs(a, b, task="small", entropy=12345, episodes=64):
    """Whether two cached variants cost the same episode for episode (a wiring check)."""
    out = tune.run_dir(task, entropy, episodes)
    ja, jb = np.array(costs_of(out, a), dtype=float), np.array(costs_of(out, b), dtype=float)
    n = min(len(ja), len(jb))
    diff = np.abs(ja[:n] - jb[:n])
    print(
        f"{a} against {b}: {int((diff == 0).sum())} of {n} episodes cost the same to the cent;"
        f" largest difference {diff.max() / 100:,.2f} USD of a mean cost of {ja[:n].mean() / 100:,.0f} USD"
    )


if __name__ == "__main__":
    fire.Fire({"run": run, "report": report, "same": same_costs})
