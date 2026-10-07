"""The hindsight calendar: tell the window LP the weeks each fab grid really had power, read off a finished play.

uv run python <this> probe --task=small --episodes=60,25 --forms=lots,energy,energy+tight [--passes=2]
uv run python <this> run --task=small --form=energy --name=ft_s_v5_hc_en --passes=2 --n_jobs=2

Pass 0 plays ``agents/ft_buffer_v5`` with v4's settings and keeps the calendar (per grid and week, the share of the
fabs' full draw the grid had above its base load); it is cached per episode. Pass i plays with the calendar of pass
i - 1 told to the window (``reserve.window_calendar`` replaced in the loaded module), and keeps its own calendar. A
ceiling, not an agent: the calendar is the future. ``run`` writes each pass as ``<name>_p<i>.json`` in the harness's
layout, so ``tune.py report`` lists it.

A form is ``<plan_calendar>[+<calendar_shed>][+nowaf][+pulsed]``: "lots" or "energy"; "tight" or "package"; "nowaf"
switches the wafer top-up off; "pulsed" tells only the weeks of grids the pulse held fuel for in that play.
"""

import json
import os
import sys
import time
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lab  # noqa: E402


AGENT = "ft_buffer_v5"
CAL = HERE / "cal"


def form_params(form):
    parts = form.split("+")
    params = dict(lab.V4) | {"plan_calendar": parts[0]}
    for p in parts[1:]:
        if p in ("tight", "package"):
            params["calendar_shed"] = p
        elif p == "nowaf":
            params["pulse_wafers"] = 0.0
        elif p in ("pulsed", "pin", "pinwaf"):
            pass
        elif p in ("idle", "short"):
            params["calendar_wafers"] = p
        elif p == "load":
            params["calendar_load"] = True
        elif p.startswith("look"):
            pass
        elif p.startswith("wafers"):  # wafers0.5: the top-up's level
            params["pulse_wafers"] = float(p[6:])
        else:
            raise ValueError(form)
    return params, "pulsed" in parts[1:], ("wafers" if "pinwaf" in parts[1:] else True) if "pin" in parts[1:] else False


def pulsed_weeks(inst, agent):
    """(T, G) bool: the grid counted as pulsed in that week of the play (a fuel was held within pulse_memory weeks)."""
    out = np.zeros((inst.T, len(inst.grids)), dtype=bool)
    pulse = getattr(agent, "pulse", None)
    if pulse is None:
        return out
    for t, go, mode, _moved in pulse.log:
        if mode == "off":
            out[t - 1 : int(t + pulse.memory), go] = True
    return out


def executed_flows(inst, traj):
    """(T, action slots) the flows the simulator executed."""
    out = np.zeros((inst.T, len(inst.action_slots)))
    for ti, r in enumerate(traj.records):
        for s, q in r.executed.items():
            out[ti, int(s)] = q
    return out


LOOK = {}  # form -> (weeks of the window told after its first, what later weeks get: "on" or "mean")


def look_of(form):
    for p in form.split("+"):
        if p.startswith("look"):
            return int(p[4:].rstrip("m")), ("mean" if p.endswith("m") else "on")
    return None


def feed(calendar, only=None, pin=None, wafers=False, look=None):
    """The patch that makes ``reserve.window_calendar`` return the true calendar's window ((T, G) shares).

    ``pin`` ((T, action slots), the flows executed in the play the calendar was read from): the terminal-to-grid
    flows of every grid with fabs are replayed from that play after the pulse has run, so the grid's fuel, and with
    it the calendar, stays that play's as long as the terminal holds the fuel.
    """

    def patch(mods):
        if pin is not None:
            P = mods["pulse"].Pulse
            orig = P.adjust

            def adjust(self, obs, values, prohibited, action):
                out = orig(self, obs, values, prohibited, action)
                t = int(obs["week"])
                if not out.get("flows") or t > pin.shape[0]:
                    return out
                slots = self.__dict__.get("_pin_slots")
                if slots is None:
                    slots = {s for _go, _g, _fabs, fuels, _sh in self.grids for fuel in fuels.values() for _e, _tail, _st, ss in fuel[2] for s in ss}
                    if wafers:  # and the wafer flows into every fab, so the lots stay that play's too
                        inst = self.inst
                        need = {f: inst.nodes[f].fab.input for f in inst.fabs}
                        slots |= {s for s, (e, k, _lane) in enumerate(inst.action_slots) if need.get(inst.edges[e].head) == k}
                    slots = sorted(slots)
                    self._pin_slots = slots
                sent = dict(zip(out["flows"]["slot"], out["flows"]["qty"]))
                for s in slots:
                    if s in sent:
                        sent[s] = float(pin[t - 1, s])
                out = dict(out)
                out["flows"] = {"slot": list(out["flows"]["slot"]), "qty": [sent[s] for s in out["flows"]["slot"]]}
                return out

            P.adjust = adjust

        def window_calendar(inst, obs, arrays, H, params):
            t = int(obs["week"])
            out = {}
            for go in lab.fab_grids(inst):
                share = calendar[t - 1 : t - 1 + H, go].copy()
                if only is not None:  # a grid not pulsed in a week of the play: nothing told for that week
                    share = np.where(only[t - 1 : t - 1 + H, go], share, 1.0)
                if look is not None and len(share) > look[0] + 1:  # only the next weeks are known; later ones are not
                    rest = share[look[0] + 1 :]
                    share[look[0] + 1 :] = 1.0 if look[1] == "on" else float(np.mean(rest))
                out[go] = share
            return out

        mods["reserve"].window_calendar = window_calendar

    return patch


def snapshot(name):
    """A copy of ``agents/ft_buffer_v5`` under the harness's agents folder, so a run does not read a folder in edit."""
    import shutil

    out = lab.OUT / "agents" / name
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(lab.ROOT / "agents" / AGENT, out, ignore=shutil.ignore_patterns("__pycache__"))
    return name


def base_pass(task, n, w=None):
    """Pass 0 of episode n, cached: (J cents, calendar (T, G), pulsed (T, G), executed flows (T, slots))."""
    CAL.mkdir(exist_ok=True)
    f = CAL / f"{task}_base_{n}.npz"
    if f.exists():
        d = np.load(f)
        if "flows" in d:
            return int(d["J"]), d["share"], d["pulsed"], d["flows"]
    w = w or lab.world(task, n)
    res = lab.play(task, n, "ft_buffer_v4", dict(lab.V4), world_=w, tag="hc0")  # v4 as played: the untouched folder
    share, _room, _draw = lab.calendar_of(w[0], w[2], res["traj"])
    pulsed = pulsed_weeks(w[0], res["agent"])
    if res["errors"]:
        raise RuntimeError(f"pass 0 of episode {n}: {res['errors'][:2]}")
    flows = executed_flows(w[0], res["traj"])
    np.savez(f, J=np.int64(res["traj"].J_cents), share=share, pulsed=pulsed, flows=flows)
    return int(res["traj"].J_cents), share, pulsed, flows


def episode(task, n, form, passes=1, save=None, agent=AGENT):
    """Passes 1..passes of one episode under ``form``. Returns the rows of each pass (pass 0 first)."""
    w = lab.world(task, n)
    inst, marks = w[0], w[2]
    j0, share, pulsed, flows = base_pass(task, n, w)
    params, only_pulsed, pin = form_params(form)
    grids = lab.fab_grids(inst)
    rows = [{"episode": n, "J_policy_cents": j0, "fallback_weeks": 0, "first_error": None, "forecast_failed": 0, "seconds": 0.0}]
    for i in range(1, passes + 1):
        start = time.perf_counter()
        res = lab.play(task, n, agent, params, patch=feed(share, pulsed if only_pulsed else None, flows if pin else None, pin == "wafers", look_of(form)), world_=w, tag=f"hc{i}")
        new, _room, _draw = lab.calendar_of(inst, marks, res["traj"])
        new_pulsed = pulsed_weeks(inst, res["agent"])
        a, b = share[:, grids], new[:, grids]
        rows.append(
            {
                "episode": n,
                "J_policy_cents": int(res["traj"].J_cents),
                "fallback_weeks": 0,
                "first_error": f"week {res['errors'][0][0]}: {res['errors'][0][1]}" if res["errors"] else None,
                "forecast_failed": getattr(getattr(res["agent"], "planner", None), "failed", None),
                "seconds": round(time.perf_counter() - start, 3),
                # how far the calendar moved once the plan knew it: mean absolute change of the share, grid-weeks that
                # switched between whole and not, and the grid-weeks whole before and after
                "calendar_moved": float(np.mean(np.abs(a - b))),
                "weeks_switched": int(((a >= 0.999) != (b >= 0.999)).sum()),
                "weeks_whole": [int((a >= 0.999).sum()), int((b >= 0.999).sum())],
                "mean_share": [float(a.mean()), float(b.mean())],
            }
        )
        share, pulsed, flows = new, new_pulsed, executed_flows(inst, res["traj"])
        if save:
            CAL.mkdir(exist_ok=True)
            np.savez(CAL / f"{task}_{save}_p{i}_{n}.npz", J=np.int64(res["traj"].J_cents), share=share, pulsed=pulsed)
    return rows


def probe(episodes, forms="lots,energy,energy+tight", task="small", passes=1):
    ns = [int(n) for n in (episodes if isinstance(episodes, (list, tuple)) else [episodes])]
    forms = list(forms) if isinstance(forms, (list, tuple)) else [f for f in str(forms).split(",") if f]
    refs = lab.references(task)
    total = {(f, i): 0.0 for f in forms for i in range(1, passes + 1)}
    gaps = 0.0
    print(f"{task}: difference to v4 as played (pass 0), bn USD and share of the episode's gap; positive: the calendar helps")
    for n in ns:
        gap = (refs[n]["J_naive_cents"] - refs[n]["J_oracle_cents"]) / 1e11
        gaps += gap
        line = f"  episode {n:3d} (level {refs[n]['stratum']}, gap {gap:7.1f})"
        for f in forms:
            rows = episode(task, n, f, passes)
            j0 = rows[0]["J_policy_cents"]
            for i in range(1, passes + 1):
                d = (j0 - rows[i]["J_policy_cents"]) / 1e11
                total[(f, i)] += d
                line += f" | {f} p{i} {d:+7.2f} ({d / gap:+.4f}) moved {rows[i]['calendar_moved']:.3f} sw {rows[i]['weeks_switched']:3d} whole {rows[i]['weeks_whole'][0]}->{rows[i]['weeks_whole'][1]}"
                if rows[i]["first_error"]:
                    line += " ERROR " + rows[i]["first_error"]
        print(line, flush=True)
    print("  sum: " + " | ".join(f"{f} p{i} {total[(f, i)]:+.1f} bn ({total[(f, i)] / gaps:+.4f} of the probed gaps)" for f in forms for i in range(1, passes + 1)))


def _chunk(task, ns, form, passes, save, agent):
    return [episode(task, int(n), form, passes, save, agent) for n in ns]


def run(form, name, task="small", passes=1, n_jobs=2, episodes=None, redo=False):
    from joblib import Parallel, delayed

    sys.path.insert(0, str(lab.ROOT / "experiments" / "approaches"))
    import tune

    total = 64 if task == "small" else 16
    out = tune.run_dir(task, 12345, total)
    targets = [out / f"{name}_p{i}.json" for i in range(1, passes + 1)]
    if all(t.exists() for t in targets) and not redo:
        print(f"{name}: cached", flush=True)
        return
    tune.wait_for_memory()
    start = time.perf_counter()
    ns = list(range(total)) if episodes is None else [int(n) for n in episodes]
    chunks = [c for c in np.array_split(np.array(ns), min(len(ns), n_jobs)) if c.size]
    agent = snapshot(name)
    parts = Parallel(n_jobs=n_jobs)(delayed(_chunk)(task, [int(n) for n in c], form, passes, name, agent) for c in chunks)
    per_episode = [rows for part in parts for rows in part]
    es = tune.episode_set(task, 12345, total, 1)
    base = lab.cached(task, "ft_s_v4" if task == "small" else "ft_v4")
    same = sum(int(rows[0]["J_policy_cents"] == base[rows[0]["episode"]]) for rows in per_episode)
    print(f"pass 0 against the cached v4: {same} of {len(per_episode)} episodes cost the same to the cent", flush=True)
    if len(ns) != total:
        print("a partial run: nothing written")
    for i in range(passes + 1):
        rows = [r[i] for r in per_episode]
        if len(ns) == total:
            if i > 0:
                tmp = out / f"{name}_p{i}.tmp"
                tmp.write_text(json.dumps({"spec": {"base": AGENT, "form": form, "pass": i, "hindsight": True}, "rows": rows}, indent=1))
                os.replace(tmp, out / f"{name}_p{i}.json")
            table = es.rss([r["J_policy_cents"] for r in rows])
            line = f"{name} pass {i}: score {table['rss']:.4f} by level {table['rss_by_stratum']}"
        else:
            line = f"{name} pass {i}: J {sum(r['J_policy_cents'] for r in rows) / 1e11:.1f} bn"
        if i > 0:
            errors = [r["first_error"] for r in rows if r["first_error"]]
            line += (f" errors {errors[:2]} failed {sum(r['forecast_failed'] or 0 for r in rows)}; calendar moved {np.mean([r['calendar_moved'] for r in rows]):.4f},"
                     f" grid-weeks switched {sum(r['weeks_switched'] for r in rows)}, whole {sum(r['weeks_whole'][0] for r in rows)} -> {sum(r['weeks_whole'][1] for r in rows)},"
                     f" mean share {np.mean([r['mean_share'][0] for r in rows]):.4f} -> {np.mean([r['mean_share'][1] for r in rows]):.4f}")
        print(line, flush=True)
    print(f"({time.perf_counter() - start:.0f} s)", flush=True)


if __name__ == "__main__":
    fire.Fire({"probe": probe, "run": run})
