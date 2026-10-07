"""Does HiGHS accept a thread count before its first run, and after a run with another count?"""

import sys

import numpy as np
from scipy.optimize._highspy import _core as hs


def tiny(threads=None):
    lp = hs.HighsLp()
    lp.num_col_, lp.num_row_ = 2, 1
    lp.col_cost_ = np.array([1.0, 1.0])
    lp.col_lower_ = np.array([0.0, 0.0])
    lp.col_upper_ = np.array([4.0, 4.0])
    lp.row_lower_ = np.array([1.5])
    lp.row_upper_ = np.array([hs.kHighsInf])
    lp.a_matrix_.format_ = hs.MatrixFormat.kColwise
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = 2, 1
    lp.a_matrix_.start_ = np.array([0, 1, 2], dtype=np.int32)
    lp.a_matrix_.index_ = np.array([0, 0], dtype=np.int32)
    lp.a_matrix_.value_ = np.array([1.0, 1.0])
    lp.integrality_ = [hs.HighsVarType.kInteger, hs.HighsVarType.kContinuous]
    h = hs._Highs()
    h.setOptionValue("output_flag", False)
    if threads is not None:
        print("set threads", threads, h.setOptionValue("threads", int(threads)))
    h.passModel(lp)
    st = h.run()
    print("run", st, h.modelStatusToString(h.getModelStatus()), h.getInfo().objective_function_value)


order = sys.argv[1] if len(sys.argv) > 1 else "1,1"
for t in order.split(","):
    tiny(None if t == "d" else int(t))
