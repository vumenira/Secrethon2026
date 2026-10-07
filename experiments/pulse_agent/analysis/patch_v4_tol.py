"""Patch pulse.py of a v4 working copy with the tolerance for short fuels that cannot be held."""

import sys

p = sys.argv[1]
s = open(p, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"


def rep(a, b, count=1):
    global s
    a, b = a.replace("\n", nl), b.replace("\n", nl)
    assert s.count(a) == count, (s.count(a), a[:70])
    s = s.replace(a, b)


rep(
    """        small = bool(params.get("pulse_small", False))
""",
    """        small = bool(params.get("pulse_small", False))
        # pulse_tolerate: a short fuel that cannot be held (no terminal, not worth a tank, a tank that fills too
        # slowly) makes the pulse stand down only if its shortfall leaves the fabs less than this share of what they
        # could get; otherwise the fabs lose that shortfall and the pulse works on the other fuels. 0: it always does
        self.tolerate = float(params.get("pulse_tolerate", 0.0))
""",
)
rep(
    """            target = min(target, G * shares - y - lost)  # the most the fabs can get this week
            if target <= TOL * max(y, 1.0):
                continue  # no fab to power this week: the planner's flows stand
            whole = able >= y + target - TOL * G
            whole_now[go] = whole
            pulsed = t - self.held.get(go, -10**9) <= self.memory
            ahead = max(self.last[go] - t, 0) if self.rundown else math.inf  # sellable weeks after this one
""",
    """            target = min(target, G * shares - y - lost)  # the most the fabs can get this week
            if target <= TOL * max(y, 1.0):
                continue  # no fab to power this week: the planner's flows stand
            pulsed = t - self.held.get(go, -10**9) <= self.memory
            ahead = max(self.last[go] - t, 0) if self.rundown else math.inf  # sellable weeks after this one
            given = set()  # short fuels that cannot be held and are given up: the fabs run on what they leave
            if self.tolerate > 0:
                spared = 0.0
                for k, (burn, threshold, ration, on_hand, tank, inflow, room, reach, feed) in state.items():
                    miss = burn - min(burn * ration, on_hand + feed)
                    if k in small or miss <= burn * TOL:
                        continue
                    lack = burn + threshold - tank - on_hand
                    if not reach or k in unheld or lack > min(self.hold_weeks, ahead) * max(inflow, self.rate[(go, k)]):
                        given.add(k)
                        spared += miss
                if given and target - spared >= self.tolerate * target and target - spared > TOL * max(y, 1.0):
                    target -= spared
                else:
                    given = set()
            whole = able >= y + target - TOL * G
            whole_now[go] = whole
""",
)
rep(
    """                    if k in small:
                        if reach and k not in unheld:
                            short.add(k)  # kept for the next ON week, where each unit of it is fab energy
                        continue
                    short.add(k)
""",
    """                    if k in small:
                        if reach and k not in unheld:
                            short.add(k)  # kept for the next ON week, where each unit of it is fab energy
                        continue
                    if k in given:
                        continue  # cannot be held and leaves the fabs enough: the planner's flow stands
                    short.add(k)
""",
)
open(p, "w", encoding="utf-8", newline="").write(s)
print("pulse patched:", p)
