"""The pulse's decision at one grid week by week, with the numbers it weighs (a copy of the gates of Pulse.adjust).

uv run python trace56.py --task=small --n=56 --grid=grid_kr --first=30 --last=40 --params="ration_trim=true"
"""

import math
import sys
from pathlib import Path

import fire

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "v6"))
import common as C  # noqa: E402
import probe as P  # noqa: E402

TOL = 1e-9


def main(task="small", n=56, grid="grid_kr", first=30, last=40, folder="mpc_pulse_rec_v6", params="", entropy=12345):
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    inst0, omega, marks, fallback = C.world(task, n, entropy)
    lines = []
    with without_secret_like():
        factory = load_agent_class(str(C.ROOT / "agents" / folder), "submission_trace")
        module = sys.modules[factory.__module__]
        module.PARAMS.update(P.parse(params) | {"strict": True})
        pulse = sys.modules["pulse"]
        orig = pulse.Pulse.adjust

        def adjust(self, obs, values, prohibited, action):
            inst, t = self.inst, int(obs["week"])
            name = [nd.id for nd in inst.nodes]
            if first <= t <= last:
                stock = {(int(a), int(b)): float(q) for a, b, q in zip(obs["stock"]["node"], obs["stock"]["k"], obs["stock"]["qty"])}
                arriving = {}
                pp = obs["pipeline"]
                for e, k, q, week in zip(pp["edge"], pp["k"], pp["qty"], pp["arrival_week"]):
                    if int(week) == t:
                        key = (inst.edges[int(e)].head, int(k))
                        arriving[key] = arriving.get(key, 0.0) + float(q)
                sent = dict(zip(action["flows"]["slot"], action["flows"]["qty"]))
                for go, g, fabs, fuels, shares in self.grids:
                    if name[g] != grid:
                        continue
                    G, y = float(values["grid_G"][go]), float(values["grid_y"][go])
                    want = 0.0
                    for fo in fabs:
                        f = inst.fabs[fo]
                        fa = inst.nodes[f].fab
                        R, alpha = float(values["fab_R"][fo]), float(values["fab_alpha"][fo])
                        if self.tail is not None and t + self.lead[fo] + self.tail > inst.T:
                            continue
                        if R > 0:
                            wafers = stock.get((f, fa.input), 0.0) + arriving.get((f, fa.input), 0.0)
                            want += fa.e * min(alpha * R * fa.cap0, wafers) / R
                    target = min(want, G * shares - y)
                    able = G * (shares - sum(fuel[0] for fuel in fuels.values()))
                    text = []
                    for k, (share, threshold, feeders, worth_it) in fuels.items():
                        here = stock.get((g, k), 0.0)
                        ration = min(1.0, here / threshold) if threshold > 0 else 1.0
                        on_hand = here + arriving.get((g, k), 0.0)
                        tank = inflow = 0.0
                        feed = planned = 0.0
                        for e, tail, _storage, ss in feeders:
                            live = [s for s in ss if s in sent]
                            held = stock.get((tail, k), 0.0)
                            tank += held
                            inflow += arriving.get((tail, k), 0.0)
                            if live and not bool(prohibited[e, k]):
                                feed += min(held, float(values["u"][e]))
                                planned += sum(sent[s] for s in live)
                        burn = share * G
                        able += min(burn * ration, on_hand + feed)
                        lack = burn + threshold - tank - on_hand
                        text.append(f"{inst.commodities[k].id}: grid {here:.0f} ration {ration:.2f} tank {tank:.0f} inflow {inflow:.0f} rate {self.rate.get((go, k), float('nan')):.0f} feed {feed:.0f} planned {planned:.0f} lack {lack:.0f} worth {worth_it}")
                    pulsed = t - self.held.get(go, -10**9) <= self.memory
                    ahead = max(self.last[go] - t, 0) if self.rundown else math.inf
                    lines.append(f"wk {t}: G {G:.0f} y {y:.0f} want {want:.0f} target {target:.0f} able {able:.0f} whole {able >= y + target - TOL * G} pulsed {pulsed} (held {self.held.get(go)}) trimmed {self.trimmed.get(go)} last sellable {self.last[go]} ahead {ahead}; leads {[self.lead[fo] for fo in fabs]}")
                    lines.extend("      " + s for s in text)
            before = len(self.log)
            out = orig(self, obs, values, prohibited, action)
            if first <= t <= last:
                lines.append("      log: " + "; ".join(f"{name[inst.grids[go]]} {what} {dict((inst.commodities[k].id, round(q)) for k, q in moved.items())}" for _w, go, what, moved in self.log[before:]))
            return out

        pulse.Pulse.adjust = adjust
        shim = _metered_shim(factory, None)
        traj = rollout(inst0, shim, omega, "standard", _policy_seed(entropy, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
    unload_agent()
    print(f"{task} episode {n} {grid}, {params or 'defaults'}: J {traj.J_cents / 1e11:,.2f} bn")
    inst = inst0.at_digest(marks.instance_digest)
    go = next(i for i, g in enumerate(inst.grids) if inst.nodes[g].id == grid)
    for line in lines:
        print(line)
        if line.startswith("wk "):
            t = int(line.split()[1].rstrip(":"))
            r = traj.records[t - 1]
            print(f"      realised: shed {float(r.shed[go]):.0f} fab energy {sum(float(r.energy[fo]) for fo in inst.grid_fabs[go]):.0f}")


if __name__ == "__main__":
    fire.Fire(main)
