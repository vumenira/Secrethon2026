"""One episode of diff.py, printed (a debugging run)."""

import sys

import common as C
import diff

n = int(sys.argv[1]) if len(sys.argv) > 1 else 41
out = diff.one("small", n)
D = out["D"]
naive = C.references("small")[n]["J_naive_cents"] / 100
sc = lambda J: (naive - J) / D  # noqa: E731
print(f"episode {n} ({out['src']}): played {sc(out['J_played']):.4f} calendar {sc(out['J_cal']):.4f} plan {sc(out['J_plan']):.4f} final {sc(out['J_final']):.4f}")
print(f"grid-weeks that differ {out.get('weeks_differ_raw')} -> {out.get('weeks_differ_physical')} -> {out.get('weeks_differ_final')}; {out.get('solves')} LPs; {out.get('seconds')} s")
for c in out.get("clusters", []):
    print(f"  {c['grid']} weeks {c['a']}-{c['b']}: on {c['on']} off {c['off']}; loss when put back {c['loss_back'] / D:.4f}, gain alone {c['gain_alone'] / D:.4f};"
          f" fab energy ours {c['ours']['E']:.0f} plan {c['plan']['E']:.0f} (ceiling in its on weeks {c['emax_on']:.0f}, partial {c['partial_on']:.0f}); shed ours {c['ours']['shed']:.0f} plan {c['plan']['shed']:.0f}")
    for k, f in c["fuel"].items():
        o, p = f["ours"], f["plan"]
        print(f"      {k:8s} burnt in the span ours {o['G']:.0f} plan {p['G']:.0f}; burnt before {o['G_before']:.0f} | {p['G_before']:.0f}; arrived by the end {o['arr_to_b']:.0f} | {p['arr_to_b']:.0f}; stock before {o['stock_before']:.0f} | {p['stock_before']:.0f}; stock at the end {o['stock_end']:.0f} | {p['stock_end']:.0f}; sent more from week {f['sent_from']} ({f['sent_more_to_b']:.0f}); all grids' arrivals, plan minus ours {c['arr_all'][k]:.0f}")
