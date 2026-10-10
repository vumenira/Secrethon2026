"""HiGHS as SciPy bundles it, under the names the planner uses from highspy (scripts/vendor_planner.py)."""

from scipy.optimize._highspy import _core


globals().update({name: getattr(_core, name) for name in dir(_core) if not name.startswith("_")})
Highs = _core._Highs
