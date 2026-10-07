"""Counterfactual worlds of one episode: the marks with a subset of its events, or with persistence from a week.

Demand is always the episode's own, so only the network differs.
"""

import dataclasses

import numpy as np


def events(inst, omega):
    from shockbench_flow.marks import read_events

    return read_events(inst, omega.arrays)


def describe(inst, q):
    """A short text for one event: type, target, onset, end, severity."""
    from shockbench_flow.omega import codes

    kind = codes.TARGET_KINDS[q.target_kind]
    if kind in ("chokepoint", "node"):
        target = inst.nodes[q.target].id
    elif kind == "edge":
        ed = inst.edges[q.target]
        target = f"{inst.nodes[ed.tail].id}->{inst.nodes[ed.head].id}"
    else:
        target = inst.regions[q.target]
    extra = ""
    if q.counterpart >= 0:
        extra += f" vs {inst.regions[q.counterpart]}"
    if q.commodity >= 0:
        extra += f" {inst.commodities[q.commodity].id}"
    return f"{codes.EVENT_TYPES[q.type]} {target}{extra} [{q.onset:.1f}, {q.end:.1f}) sev {q.severity:.2f}"


def subset_marks(inst, omega, keep, until=None):
    """The marks of the episode with only the events whose rows are in ``keep`` (true demand kept).

    ``until``: a week; every kept event then lasts at least to the end of that week (persistence of what is seen).
    """
    from shockbench_flow.marks import compute_marks, mark_params_from_json, stored_mark_arrays
    from shockbench_flow.omega.container import EVENT_FIELDS

    arrays = omega.arrays
    idx = np.array(sorted(int(i) for i in keep), dtype=int)
    new = {}
    for name, _dt in EVENT_FIELDS:
        new[f"ev_{name}"] = np.ascontiguousarray(np.asarray(arrays[f"ev_{name}"])[idx])
    if until is not None and len(idx):
        onset = new["ev_onset"]
        new["ev_duration"] = np.maximum(new["ev_duration"], float(until) + 1.0 - onset)
    ptr, key = np.asarray(arrays["ev_key_ptr"]), np.asarray(arrays["ev_key"])
    ks, p = [], [0]
    for i in idx:
        ks += [int(v) for v in key[ptr[i] : ptr[i + 1]]]
        p.append(len(ks))
    new["ev_key_ptr"], new["ev_key"] = np.array(p, dtype=np.int64), np.array(ks, dtype=np.int64)
    if "V_lead" in arrays and np.asarray(arrays["V_lead"]).shape[0] == np.asarray(arrays["ev_type"]).shape[0]:
        new["V_lead"] = np.ascontiguousarray(np.asarray(arrays["V_lead"])[idx])
    params = mark_params_from_json(str(arrays["meta_mark_params"]))
    new.update(stored_mark_arrays(inst, new, params))
    return compute_marks(inst, omega.replace(**new))


def started_by(inst, omega, t):
    """Rows of the events whose onset is at or before the instant t - 1 (what week t's observation can show)."""
    return [q.row for q in events(inst, omega) if q.onset <= t - 1.0]


def persist_marks(marks, t):
    """The marks with the network of week t's observation kept from week t on (weeks before t as they were).

    Week averages take the instantaneous values at t - 1; marks without an instantaneous twin keep week t's value;
    no fab is hit from week t on.
    """
    ti = t - 1
    new = {}
    for name in ("u", "o", "kappa", "supply", "G_bar", "y_bar", "R"):
        a = np.array(getattr(marks, name))
        now = np.array(getattr(marks, f"{name}_now"))
        a[ti:] = now[ti]
        now[ti:] = now[ti]
        new[name], new[f"{name}_now"] = a, now
    alpha, alpha_now = np.array(marks.alpha_bar), np.array(marks.alpha_now)
    alpha[ti:] = alpha_now[ti]
    alpha_now[ti:] = alpha_now[ti]
    new["alpha_bar"], new["alpha_now"] = alpha, alpha_now
    for name in ("c", "R_osat", "prohibited", "tariff", "wr_class", "h_queue", "c_wr"):
        a = np.array(getattr(marks, name))
        a[ti:] = a[ti]
        new[name] = a
    scr = np.array(marks.sigma_scr)
    hits = tuple(h for h in marks.fab_hits if h.onset_week < t)
    # the scrap share of lots started in week s: hits with onset week in (s, s + w_scr]; only the kept hits count
    for fo in range(scr.shape[1]):
        for s in range(scr.shape[0]):
            keep = 1.0
            for h in hits:
                if h.fab == fo and (s + 1) < h.onset_week <= (s + 1) + marks.w_scr[fo]:
                    keep *= 1.0 - h.severity
            scr[s, fo] = 1.0 - keep
    new["sigma_scr"], new["fab_hits"] = scr, hits
    return dataclasses.replace(marks, **new)
