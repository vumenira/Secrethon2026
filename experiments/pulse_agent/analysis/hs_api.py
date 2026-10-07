from scipy.optimize._highspy import _core
names = [n for n in dir(_core) if not n.startswith("__")]
print([n for n in names if "Highs" in n or "VarType" in n or "Solution" in n][:60])
h = _core._Highs()
print([m for m in dir(h) if "olution" in m or "Info" in m or "ption" in m or "ntegrality" in m or "passModel" in m])
print([m for m in dir(_core.HighsLp()) if not m.startswith("__")])
print([m for m in dir(_core.HighsSolution()) if not m.startswith("__")])
info = h.getInfo()
print([m for m in dir(info) if "mip" in m or "objective" in m or "gap" in m])
print(list(_core.HighsVarType.__members__))
