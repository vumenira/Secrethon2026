"""Write the y_ variants files of the third merge step (agents/mpc_pulse_rec), every setting spelled out."""

import json
from pathlib import Path


OUT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026\outputs\recovery_tune\variants")
SNAPSHOT = json.loads((Path(__file__).parent / "ft_buffer_v1" / "params.json").read_text())  # as uploaded, strict on
ALL = ["closure", "capacity", "grid"]
OLD = {  # agents/mpc_combined without its reserve (x_e10_rec): no pulse, none of the knobs agents/ft_buffer added
    "h_weeks": 20,
    "end_value": 1.0,
    "reserve_weeks": 0.0,
    "reserve_feeders": False,
    "shed_ramp": 0.0,
    "pulse": False,
    "short_premium": 0.0,
    "kinds": ALL,
    "strict": True,
}


def pulse(kinds, **more):
    return {"base": "mpc_pulse_rec", "params": SNAPSHOT | {"kinds": kinds} | more}


files = {
    "y_small1": {"y_reg_pulse": pulse([]), "y_reg_rec": {"base": "mpc_pulse_rec", "params": OLD}, "y_all": pulse(ALL)},
    "y_full1": {"y_reg_pulse": pulse([]), "y_all": pulse(ALL)},
    "y_kinds": {"y_closure": pulse(["closure"]), "y_capacity": pulse(["capacity"]), "y_grid": pulse(["grid"])},
    "y_cc": {"y_cc": pulse(["closure", "capacity"])},
}
assert SNAPSHOT["strict"] is True
for name, specs in files.items():
    (OUT / f"{name}.json").write_text(json.dumps(specs, indent=1))
    print(name, {k: v["params"].get("kinds") for k, v in specs.items()})
print("snapshot params:", json.dumps(SNAPSHOT))
