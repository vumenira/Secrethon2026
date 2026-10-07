"""Why the first validate run returned "Not Set": the return codes of each HiGHS call on episode 0."""

import numpy as np

import common as C


d = C.load_traj("small", 0)
cal = C.Cal("small", 0)
zpi = np.concatenate([d["zpi"], np.zeros(cal.nz)])
z, both = cal.calendar(d["zpi"])
zpi[cal.nn :] = z.ravel()
from scipy.optimize._highspy import _core as hs

A = cal.A
print("A", type(A), A.shape, A.format, "indptr", A.indptr.dtype, len(A.indptr), "nnz", A.nnz)
print("lb>ub", int((cal.lb > cal.ub).sum()), "nan cost", int(np.isnan(cal.cost).sum()), "nan A", int(np.isnan(A.data).sum()))
print("lo>up", int((cal.lo > cal.up).sum()), "nan lo/up", int(np.isnan(cal.lo).sum()), int(np.isnan(cal.up).sum()))
inf = hs.kHighsInf
lp = hs.HighsLp()
lp.num_col_, lp.num_row_ = A.shape[1], A.shape[0]
lp.col_cost_ = cal.cost
lp.col_lower_ = np.where(np.isinf(cal.lb), -inf, cal.lb)
lp.col_upper_ = np.where(np.isinf(cal.ub), inf, cal.ub)
lp.row_lower_ = np.where(np.isinf(cal.lo), -inf, cal.lo)
lp.row_upper_ = np.where(np.isinf(cal.up), inf, cal.up)
lp.offset_ = float(cal.offset)
lp.a_matrix_.format_ = hs.MatrixFormat.kColwise
lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = A.shape[1], A.shape[0]
lp.a_matrix_.start_ = A.indptr.astype(np.int32)
lp.a_matrix_.index_ = A.indices.astype(np.int32)
lp.a_matrix_.value_ = A.data.astype(np.float64)
h = hs._Highs()
h.setOptionValue("output_flag", True)
print("threads", h.setOptionValue("threads", 1))
print("passModel", h.passModel(lp))
print("cols", h.getNumCol(), "rows", h.getNumRow())
lb, ub = cal.lb.copy(), cal.ub.copy()
lb[cal.nn :], ub[cal.nn :] = z.ravel(), z.ravel()
n = len(lb)
st = h.changeColsBounds(n, np.arange(n, dtype=np.int32), np.where(np.isinf(lb), -inf, lb), np.where(np.isinf(ub), inf, ub))
print("changeColsBounds", st)
print("time_limit", h.setOptionValue("time_limit", 900.0))
print("run", h.run())
print("status", h.modelStatusToString(h.getModelStatus()), "objective", h.getInfo().objective_function_value)
