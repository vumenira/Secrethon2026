"""Two agents on one episode: base load shed per grid, lots per fab, unserved demand, in bn USD and by weeks."""

import json
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent
REFS = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026\outputs\recovery_tune\full-e12345-n16\references.json")


def main(a="ft_pulse_r4", b="f_e10_w4", episode=14, block=13):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator("full")
    N = inst.nodes
    A = np.load(HERE / "gap" / f"{a}_{episode}.npz", allow_pickle=True)
    B = np.load(HERE / "gap" / f"{b}_{episode}.npz", allow_pickle=True)
    O = np.load(HERE / "gap" / f"oracle_{episode}.npz", allow_pickle=True)
    ref = json.loads(REFS.read_text())[episode]
    gap = (ref["J_naive_cents"] - ref["J_oracle_cents"]) / 100
    print(f"episode {episode} level {ref['stratum']}: gap {gap / 1e9:.0f} bn; {a} J {float(int(A['J_cents'])) / 1e11:.1f}, {b} J {float(int(B['J_cents'])) / 1e11:.1f}, clairvoyant {float(int(O['J_cents'])) / 1e11:.1f}")
    T = A["shed"].shape[0]
    blocks = [(lo, min(lo + block, T)) for lo in range(0, T, block)]
    print("base load shed, bn USD per block of weeks:", " ".join(f"{lo + 1}-{hi}" for lo, hi in blocks))
    for go, g in enumerate(inst.grids):
        ra = [A["shed"][lo:hi, go].sum() / 1e9 for lo, hi in blocks]
        rb = [B["shed"][lo:hi, go].sum() / 1e9 for lo, hi in blocks]
        ro = [O["shed"][lo:hi, go].sum() / 1e9 for lo, hi in blocks]
        if max(ra + rb) < 0.5:
            continue
        print(f"  {N[g].id:9s} {a}: " + " ".join(f"{x:6.1f}" for x in ra) + f"  = {sum(ra):7.1f}")
        print(f"  {'':9s} {b}: " + " ".join(f"{x:6.1f}" for x in rb) + f"  = {sum(rb):7.1f}")
        print(f"  {'':9s} clairvoyant: " + " ".join(f"{x:6.1f}" for x in ro) + f"  = {sum(ro):7.1f}")
    print("lots started, share of capacity per block:")
    for fo, f in enumerate(inst.fabs):
        cap = N[f].fab.cap0
        ra = [A["lots_started"][lo:hi, fo].mean() / cap for lo, hi in blocks]
        rb = [B["lots_started"][lo:hi, fo].mean() / cap for lo, hi in blocks]
        print(f"  {N[f].id:18s} " + " ".join(f"{x:4.2f}" for x in ra) + "  |  " + " ".join(f"{x:4.2f}" for x in rb))
    print(f"unserved demand bn: {a} {A['short'].sum() / 1e9:.1f}, {b} {B['short'].sum() / 1e9:.1f}, clairvoyant {O['short'].sum() / 1e9:.1f}")
    print(f"shed bn: {a} {A['shed'].sum() / 1e9:.1f}, {b} {B['shed'].sum() / 1e9:.1f}, clairvoyant {O['shed'].sum() / 1e9:.1f}")


if __name__ == "__main__":
    fire.Fire(main)
