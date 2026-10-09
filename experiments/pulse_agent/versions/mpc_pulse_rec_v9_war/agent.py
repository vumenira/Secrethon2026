"""``mpc_det`` with a recovery forecast, the LP changes of ``reserve.py`` and fuel pulsed at grids short of it.

Each week it plans ``h_weeks`` weeks and plays the first. It is ``agents/mpc_pulse_rec_v8_on`` (the agent uploaded on
2026-10-07) with ``open_hold`` on in ``PARAMS`` itself (the folder has no ``params.json``) and one rule added to
``pulse.py``, switched by ``schedule`` (two more entries of ``PARAMS``): ``reserve.py``, ``recovery.py`` and
``sbflow/`` are that agent's files, byte for byte, and with ``schedule`` false the folder plays as
``agents/mpc_pulse_rec_v8_on`` does, to the cent. That agent is ``agents/mpc_pulse_rec_v7`` with one rule added to
``pulse.py``, switched by ``open_hold`` (seven more entries of ``PARAMS``, all ``open_hold...``); with ``open_hold``
false besides the folder plays as ``agents/mpc_pulse_rec_v7`` does. That agent is ``agents/mpc_pulse_rec_v6``
with three rules added to ``pulse.py`` and five more entries of ``PARAMS`` (``ration_trim_release``, ``fuel_trim``
with ``fuel_trim_cover`` and ``fuel_trim_margin``, ``osat_fit``); with those three switches false besides it plays
as ``agents/mpc_pulse_rec_v6`` does, to the cent. That agent is ``agents/mpc_pulse_rec_v4`` with three rules of its own
in ``pulse.py`` (``fuel_cancel``, ``ration_trim`` with ``ration_trim_margin`` and ``ration_trim_cover``,
``wafers_fit``), and that one ``agents/ft_buffer_v4`` (the fourth version of the uploaded ``agents/ft_buffer``) with
the recovery forecast of ``agents/mpc_combined`` (``recovery.py``) in the planner's window.

The benchmark's planner assumes the network stays as observed for the whole window (a closed strait stays closed, an
announced sanction starts on its week) and demand follows the forecast, solves a linear program for the cheapest flows
over the window, and sends this week's. Five things change that, the first four in this order each week:

- the forecast (``recovery.py``; ``kinds``, the rule ``mode``, ``quantile``, ``first``, ``caution``, and ``rules`` per
  kind or event type): an impairment of a kind in ``kinds`` ends in the window instead of lasting it. A militarised
  closure fades as the published duration law of its type says, given how long it has been seen; a storm's closure is
  planned as over from this week on; the year-long closure a CN-TW conflict puts on the Taiwan Strait is kept; a port
  stoppage fades as its law says, with this week's shipments asked for as if it were over; a port slowdown and a war's
  or a sanction's cut of an edge are kept;
- the LP (``reserve.py``): a soft minimum on the fuel a grid and its terminal hold, a price on shed base load that falls
  over the window, a share of the customs value for goods in stock, in transit or in process when the window ends
  before the episode does, and a price on a fuel's shortfall at a grid with fabs (``short_premium``; per unit of the
  fabs' draw for a fuel whose segment is smaller than that draw, ``premium_small``; ``premium_limit``, left off, would
  keep it in full only for the grid's scarcest fuel, read off the window's start state and its first week's grid
  power);
- the pulse (``pulse.py``), after the planner has chosen the week's flows: the simulator serves a grid's base load
  before its fabs, so a fuel short by a few per cent takes all the fabs' energy; the pulse holds the short fuel at the
  terminal for some weeks and delivers a full week's burn in others, and the fabs run in the whole weeks. A fab stops
  counting toward a pulse once a lot started now could not reach a sink before the episode ends (``pulse_tail``); near
  that week held gas is spent on the whole weeks that are left (``pulse_rundown``); the wafers of the fabs on a
  pulsed grid are topped up to a week of capacity, so a whole week finds them (``pulse_wafers``: it raises
  material-to-fab flows the planner already sends); and a fuel whose segment is smaller than the fabs' draw never
  makes the pulse stand down (``pulse_small``);
- three cuts to the week's flows, also in ``pulse.py`` and after the pulse (they only lower flows): fuel sent from a
  source straight into a grid that the grid could never burn in the weeks left is not sent (``fuel_cancel``: the
  planner sends it for its window-end value); in a week a grid sheds nothing, the rationed fuel delivered is cut to
  the week's burn plus the stock that rations next week's segment to what next week needs, so the segment that needs
  no fuel runs in full and the fuel saved waits at the terminal for a week that would shed (``ration_trim``; its
  forecast of next week's fab draw counts the wafers a strait will pass on by the simulator's release rule,
  ``ration_trim_release``); and no wafers are sent toward a fab of a grid that sheds base load and is not pulsed
  beyond what its store can take (``wafers_fit``);
- two more rules of ``pulse.py``, added in ``agents/mpc_pulse_rec_v7``: a fuel that is not rationed is delivered from
  its terminal as the week needs it and no more, so in a week with slack it gives way to the segment that needs no
  fuel (``fuel_trim``); and an OSAT whose store of a packaged chip is full is sent no more raw chips than the store
  ships out (``osat_fit``).

One rule came with ``agents/mpc_pulse_rec_v8`` (``open_hold``; ``pulse.py``, with the pulse's own rules): where the
episode's first window plan already leaves a grid's gas segment short in its weeks 4 to 12, the gas stays at the
terminal in weeks 2 to 4 of the episode, so that it is not burnt for the few lots the fabs start in the opening weeks,
and the pulse then delivers it in whole weeks. It is the one rule that reads the planner's solution:
``Planner._record`` keeps the week's window model and solution, and ``Agent.pulsed`` hands them to ``Pulse.adjust``.
On the tuning roots it gains on Small and not on Full (the numbers are at ``open_hold`` in ``PARAMS``); it is on
here because the uploaded agent plays with it.

One rule is added here (``schedule``; ``pulse.py``): at a pulsed grid the last whole week of a run is delivered the
rationed fuel it burns and no more, and the charge that starts a run is made as soon as one whole week can follow.
Both tests are the simulator's own rule for the rationed segment applied to what the week's observation shows (the
stock at the grid and at the terminal, the week's arrivals): no forecast and no setting. It replaces the wait of
``pulse_run`` and the refill of the threshold in a whole week the next cannot follow. Where it comes from: the best
schedule of terminal-to-grid deliveries, found with hindsight in an exact model of the grid rule, ran its whole weeks
this way; re-solving that schedule every week from forecasts did not beat the hand rules, and this part of it needs
none (the numbers are at ``schedule`` in ``PARAMS``).

The rest is ``agents/mpc_det_h20``: the planner is shockbench-flow's code (MIT) in ``sbflow/``
(``scripts/vendor_planner.py``, unchanged), solved with the HiGHS that SciPy bundles, and this file translates between
the agent's arrays and the observation the planner reads. It reads no warnings and no announcements.

``PARAMS`` holds the settings, as measured best. A ``params.json`` beside this file replaces entries of it (``rules``
as a whole); a name ``PARAMS`` does not have, a value of another kind than its default, or one the forecast or the
reserve refuses is an error when the agent is loaded. With ``schedule`` false it plays as
``agents/mpc_pulse_rec_v8_on``; with ``open_hold`` false besides as ``agents/mpc_pulse_rec_v7``; with
``ration_trim_release``, ``fuel_trim`` and ``osat_fit``
false besides as ``agents/mpc_pulse_rec_v6``; with ``fuel_cancel``, ``ration_trim`` and ``wafers_fit`` false besides
as ``agents/mpc_pulse_rec_v4``; with ``kinds`` empty besides as ``agents/ft_buffer_v4`` does (on the same settings);
with ``pulse_small`` and ``premium_small`` false as ``agents/mpc_pulse_rec_v3``; with ``pulse_tail`` 0,
``pulse_rundown`` false and ``pulse_wafers`` 0 besides as ``agents/mpc_pulse_rec_v2``; with ``pulse`` false and the
knobs ``agents/mpc_buffer`` does not have at their off values, as ``agents/mpc_combined``.

What the modules are called with (all a newer ``pulse.py`` or ``reserve.py`` has to keep working with):

- ``pulse.Pulse(inst, PARAMS)``, once per episode: the planner's instance and this whole dict. A knob it reads must be
  an entry of ``PARAMS`` to be set from ``params.json`` (a new knob is one more entry here);
- ``Pulse.adjust(obs, values, prohibited, action, plan)``, once per week after the planner: the wire observation, the
  planner's memory of the network as observed (``ObservedGraph.values``: never the forecast's window), the
  prohibitions in force this week and the planner's wire action; it returns the wire action to send. ``plan`` is the
  week's window model and its solution, or None (``open_hold`` off, or a week the planner fell back); a caller that
  leaves it out plays without ``open_hold``;
- ``reserve.Settings.of(PARAMS, week, T)`` (also once at load, as the check of the reserve's settings),
  ``reserve.rolled_lp(inst, obs, arrays, H, PARAMS, planning_rules=...)`` with the forecast's window as ``arrays``, and
  ``reserve.changes(model)``.

A week the forecast fails is planned on what is observed; a week the reserve's code fails is planned with the
benchmark's own LP (both counted, ``Planner.failed``); a week the pulse's code fails sends the planner's flows as they
were, and an episode whose pulse cannot be built is played without it (``Agent.pulse_failed``); a week the planner
fails sends the maximum.
"""

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pulse
import reserve
from recovery import Recovery, check
from sbflow.instance.schema import WAR_RISK_CLASSES
from sbflow.policies import lp_common
from sbflow.policies.base import StepTelemetry
from sbflow.policies.mpc_det import MpcDet, MpcDetParams
from sbflow.policies.registry import PolicyContext


PARAMS = {
    "h_weeks": 20,  # the planning window in weeks; None: the network's longest lead time L (24 on small and full)
    # ----- the recovery forecast (recovery.py) -----
    # the impairments forecast to end, of closure, capacity, grid, supply, fab, osat (the others are kept); []: none
    "kinds": ["closure", "capacity", "grid"],
    # the rule an impairment ends by: these four entries, unless ``rules`` says otherwise for its kind or event type
    "mode": "mean",  # "mean": the expected share of each week still impaired; "quantile": impaired until a quantile
    "quantile": 0.5,  # of its remaining life (0: planned as over from this week on); "keep": impaired all the window
    "first": None,  # this week alone: "mean" (its expected share), "open" (as if over); None: as the mode says
    "caution": 1.0,  # the power on the probability an impairment is still on: below 1, recoveries come later
    # {a kind or an event type: the entries of the four above that differ for it}; an event type's entry goes before
    # its kind's. The event types: weather_closure and militarised_closure (closure), port_stoppage and port_slowdown
    # (capacity), energy_shock (grid), material_outage (supply), regional_conflict (fab, osat)
    "rules": {
        "weather_closure": {"mode": "quantile", "quantile": 0.0},
        "grid": {"mode": "quantile", "quantile": 0.0},
        "port_stoppage": {"first": "open"},
        "port_slowdown": {"mode": "keep"},
    },
    # added in agents/mpc_pulse_rec_v9_war (recovery.WarEnds). True: the 0.27 of capacity that a war's second window
    # leaves the edges between its two regions lasts 52 weeks exactly (the generator's war profile), so the window has
    # that capacity back from the 52nd week after the cut was first shown, and half of it in the week before where
    # the cut's start was seen (it ends inside that week); False: a war's cut is kept for the whole window, as
    # agents/mpc_pulse_rec_v9 plays. No setting of its own. It can act on Full only: on Small no such cut ends inside
    # the 52 weeks. Worth about +0.0001 on Full, all of it the half week; the plays behind that number were made with
    # the first version of recovery.WarEnds, before its reading of shares was made independent of the table's size
    # (the replay with this version was stopped for low memory and not repeated)
    "war_end": True,
    # ----- the window's LP (reserve.py) -----
    # the fuel a grid should hold, in weeks of its full-rate burn of that fuel (the fuel's share of the grid times its
    # nominal deliverable power), capped at the slot's storage; 0: no reserve
    "reserve_weeks": 8.0,
    "reserve_price": 0.001,  # the price of a unit-week of fuel missing below that level, as a share of the grid's VOLL
    "reserve_fuels": "all",  # "all": every fuel a grid burns; "rationed": only its rationed fuel (LNG)
    # True: the cover a grid's storage cannot hold is asked of the node feeding it over a zero-lead edge (its terminal)
    "reserve_feeders": True,
    # shed base load costs this share of VOLL more for every window week still to come (a ramp down to VOLL in the
    # window's last week), so a plan sheds later rather than earlier; with a reserve, keep it above reserve_price, or
    # the reserve pays for shedding early; 0: no ramp
    "shed_ramp": 0.002,
    # while the window ends before the episode: goods in stock, in transit or in process at the window's end are worth
    # this share of their customs value where that is above the salvage price; 0: the salvage price
    "end_value": 2.0,
    "end_value_fuel": None,  # the same share for the goods a grid burns (fuels) when it should differ; None: end_value
    # False: fuel held at grids and flows into grids keep the salvage price at the window's end (no end value there)
    "end_value_grids": True,
    # what a lot in process at the window's end is worth: "input" (its wafer) or "product" (its raw chip)
    "end_value_wip": "input",
    # the window weeks (from the first) in which shed base load at a grid with fabs carries the base-first price, so a
    # plan runs a fab only on energy above the base load, as the simulator does; 1: the planner's own (the first week)
    "base_first_weeks": 1,
    # the price it carries: "package", the planner's first-week price; "tight", just above a fab's energy value
    "base_first_price": "package",
    # a fuel segment's shortfall at a grid with fabs costs this share of the chips it loses under the pulse (the fabs'
    # weekly chips less VOLL times their energy, per unit of the fuel's weekly burn), so the plan sends a scarce fuel
    # to the grid that loses most; 0: every shortfall costs VOLL alike, as the planner has it
    "short_premium": 1.0,
    # above 0: chips kept at their fab cost the plan nothing to hold, and this share of the holding cost less, while
    # the end value applies, so a fab whose outlets are cut builds its chip stock now, not in the window's last weeks
    "stock_early": 0.0,
    # True: the shortfall premium of a fuel whose segment is smaller than the fabs' draw is per unit of that draw, not
    # of its burn (it costs the fabs a unit of energy per unit short; it cannot stop them)
    "premium_small": True,
    # True: the shortfall premium is taken off the part of a fuel's shortfall that the grid's scarcest fuel makes
    # useless (beyond the share of its burn that fuel covers), so a plentiful fuel stops outbidding the scarce one.
    # Off as measured: next to pulse_small and premium_small it gave +0.0014 on Small and +0.0001 on Full (tuning
    # roots), +0.0004 and -0.0003 on fresh ones, every interval holding 0
    "premium_limit": False,
    # True: from the window's second week the lots of a grid's fabs are bounded at the share of capacity its fuel
    # cover allows (the duty the pulse gives them); False: the planner runs them at capacity on shed base load
    "plan_duty": False,
    # the weeks over which stock and arrivals under way are counted as a fuel's cover (read by premium_limit and
    # plan_duty only)
    "cover_weeks": 4.0,
    # ----- the pulse (pulse.py) -----
    # True: a fuel that a grid with fabs is short of is held at its terminal until the grid can run a whole week, and
    # delivered then, so the fabs run in some weeks instead of never
    "pulse": True,
    "pulse_hold_weeks": 12.0,  # a fuel whose tank would take longer than this to fill for one whole week is not held
    "pulse_fill": 0.9,  # fuel is held up to this share of the terminal's storage
    "pulse_margin": 1.0,  # a whole week delivers this many times the week's burn of a fuel the terminal runs short of
    # the rationed fuel (its grid stock must first be lifted to the rationing threshold, which the off weeks burn) is
    # delivered again when the terminal holds enough for this many whole weeks in a row, or is full
    "pulse_run": 4.0,
    # a grid not yet pulsed starts only when its fabs would get less than this share of their draw as things are
    # (a mild shortage leaves them most of it every week, which a pulse does not beat); 1: any shortfall starts one
    "pulse_partial": 0.5,
    "pulse_memory": 4.0,  # a grid counts as pulsed for this many weeks after the last week a fuel was held for it
    # a fuel is held only if this many whole weeks of the grid's fabs pay for the energy a full tank defers (lost
    # when the shortage ends with the tank full); 0: every fuel with a terminal may be held
    "pulse_payback": 12.0,
    # a fab asks for no energy once a lot started now would reach no sink before the episode ends with this many weeks
    # to spare (its chips are then worth nothing, and a pulse would shed base load for them); None: the rule is off
    "pulse_tail": 2,
    # True: near the last week a lot can still be sold, the rationed fuel is delivered for as many whole weeks as are
    # left instead of pulse_run, and no fuel is held that cannot give a whole week by then
    "pulse_rundown": True,
    # the wafers of a fab on a pulsed grid (at the fab, on their way, sent this week) are topped up to this many weeks
    # of its capacity over the direct edges from material nodes, so a whole week finds a full week's wafers (the
    # planner sends an idle fab one week of its own plan and then stops); 0: the planner's wafer flows stand
    "pulse_wafers": 1.0,
    "pulse_wafers_room": 0.5,  # no top-up while the fab's chip stock is above this share of its storage
    # True: the top-up's target grows by the planner's own wafers at hand for the fab (one week of its plan) with
    # every idle week, up to pulse_wafers weeks of capacity: the lots the plan wanted and power denied, no more
    "pulse_wafers_catchup": False,
    # True: a fuel whose whole segment is smaller than the fabs' full draw never keeps a week from being ON and never
    # makes the pulse stand down (it can only take part of the fabs' energy); it is still held in OFF weeks
    "pulse_small": True,
    # ----- added in agents/mpc_pulse_rec_v6 (pulse.py; they need "pulse" on and only lower flows) -----
    # Measured against agents/mpc_pulse_rec_v4 on the tuning roots (Small 64 episodes, Full 16; 90 % paired intervals)
    # True: a fuel sent from a source straight into a grid is cut to what the grid can still burn (its stock, what is
    # on its way and the dispatch never exceed the fuel's full-rate burn of the weeks left); the rationed fuel is left
    # out. The planner sends such fuel for its window-end value; it would sit at the grid to the end. Small +0.0025
    # (+0.0020 to +0.0031), Full +0.0035 (+0.0028 to +0.0045), no episode worse: only freight, tariff and holding change
    "fuel_cancel": True,
    # True: in a week a grid sheds nothing, the rationed fuel delivered from its terminal is cut to the week's burn
    # plus the stock that rations next week's segment to what next week needs, so the segment that needs no fuel runs
    # in full and the fuel saved waits at the terminal for a week that would shed. Small +0.0086 (+0.0069 to +0.0105,
    # 60 episodes better / 4 worse), Full +0.0061 (+0.0048 to +0.0074, 16 / 0)
    "ration_trim": True,
    # the room left above next week's forecast need, as a share of the energy the grid's fabs draw at full capacity
    # (0.1 gave a quarter less than 0 on 16 Small episodes: the forecast is rarely too low)
    "ration_trim_margin": 0.0,
    # the trim applies while the grid and its terminal hold less than this many weeks of the fuel's full-rate burn
    # (with plenty, fuel saved is never needed; without this gate the scores are the same to four places)
    "ration_trim_cover": 6.0,
    # True: in a week a grid sheds base load (its fabs start no lot) and is not being pulsed, the wafers sent toward
    # each of its fabs are cut to what the fab's store can still take, counting what is on its way there. Small
    # -0.0000 (-0.0002 to +0.0001), Full +0.0006 (+0.0003 to +0.0008, 14 / 2). At pulsed grids too it lost 0.0029 on
    # Small: their fabs need the wafers gathered in OFF weeks
    "wafers_fit": True,
    # ----- added in agents/mpc_pulse_rec_v7 (pulse.py; they need "pulse" on) -----
    # Measured against agents/mpc_pulse_rec_v6 on the tuning roots (Small 64 episodes, Full 16; 90 % paired intervals)
    # True: ration_trim's forecast of next week's need counts, of the goods waiting at a strait, what the simulator's
    # release rule passes on this week (throughput by pool, arrival cohorts in order, the next edge's capacity);
    # False: the whole queue of a strait that is not shut, as agents/mpc_pulse_rec_v6 did (the fabs' draw was then
    # forecast too high for as long as wafers queued at a throttled strait, and the gas burnt in the slack).
    # Small +0.0018 (+0.0013 to +0.0022, 56 episodes better / 5 worse), Full +0.0007 (+0.0003 to +0.0011, 12 / 2)
    "ration_trim_release": True,
    # True: in a week a grid can serve its base load and its fabs' whole draw, a fuel that is not rationed is delivered
    # from its terminal so that it has on hand what its segment burns once the week's slack is taken off it (a segment
    # gives no more than the fuel on hand, and a zero-lead delivery arrives before the grid runs): the rest waits at
    # the terminal, and a delivery the planner left under that is raised to it. Small +0.0011 (+0.0007 to +0.0015,
    # 49 / 15), Full +0.0018 (+0.0007 to +0.0033, 15 / 1, one episode carrying half of it). With
    # ration_trim_release: Small +0.0029 (+0.0023 to +0.0036, 58 / 6), Full +0.0027 (+0.0013 to +0.0042, 15 / 1)
    "fuel_trim": True,
    # fuel_trim applies while the grid and its terminal hold less than this many weeks of the fuel's full-rate burn
    # (a plentiful fuel is left to burn: cutting it burns more of the others). Without the gate Small is 0.0001
    # lower (43 episodes better / 20 worse, the losses larger); not played on Full
    "fuel_trim_cover": 6.0,
    # the room fuel_trim leaves above the week's need, as a share of the energy the grid's fabs draw at full capacity
    # (not varied)
    "fuel_trim_margin": 0.0,
    # True: in a week an OSAT's store of a packaged chip is full when it begins, the raw chips sent toward the OSAT
    # are cut to what the planner's own flows take out of that store this week (the rest would be packaged and
    # disposed of); False: the planner's flows stand. Off as measured, alone: Small -0.0001 (-0.0002 to +0.0000,
    # 7 / 35), Full -0.0001 (-0.0001 to -0.0000, 2 / 14)
    "osat_fit": False,
    # ----- added in agents/mpc_pulse_rec_v8 (pulse.py; it needs "pulse" on) -----
    # Measured against agents/mpc_pulse_rec_v7 on the tuning roots (Small 64 episodes, Full 16; 90 % paired intervals)
    # True: where the episode's first window plan leaves a grid's rationed fuel's segment short by open_hold_short or
    # more of its capacity over window weeks 4 to 12 (pulse.OPEN_SPAN), that fuel stays at the terminal in weeks
    # open_hold_first..open_hold_last of the episode (until the terminal is full), and the grid counts as pulsed: the
    # gas the grid would burn for fabs that start few lots in the opening weeks waits for whole weeks. False: the
    # pulse starts once the grid is short, as agents/mpc_pulse_rec_v7 plays. As measured, with the values below:
    # Small +0.0053 (+0.0031 to +0.0076, 35 episodes better / 17 worse), Full -0.0000 (-0.0011 to +0.0011, 8 / 6).
    # A gain on Small and none on Full, where the hold mostly shifts the pulse's cycle by a few weeks. On here: the
    # uploaded agent (agents/mpc_pulse_rec_v8_on) plays with it
    "open_hold": True,
    # the first and the last week of the episode in which open_hold holds a fuel (week 1's lots run on wafers that
    # were on their way when the episode began; both tasks start alike). Weeks 2 to 8, with open_hold_recheck:
    # Small +0.0054 against +0.0050 for 2 to 4, Full -0.0006 for both
    "open_hold_first": 2,
    "open_hold_last": 4,
    "open_hold_short": 0.1,  # the planned shortfall of the fuel's segment, as a share of its capacity (not varied)
    # True: not at a grid where the plan also leaves another fuel short (by open_hold_short or more) that can take all
    # the fabs' energy, one whose segment is no smaller than their draw: gas saved there buys no whole week. False
    # (with open_hold_recheck): Small +0.0045 (36 / 24) against +0.0050 (35 / 17), Full -0.0006 for both
    "open_hold_alone": True,
    # True: every week's plan decides for its own week, so a hold can start in any of the weeks and ends when the plan
    # stops expecting the shortage; False: the episode's first plan decides for all of them. Off as measured:
    # Small +0.0050 (+0.0029 to +0.0074, 35 / 17), Full -0.0006 (-0.0031 to +0.0014, 8 / 7); the holds it starts
    # late rest on small shortfalls and lose
    "open_hold_recheck": False,
    # open_hold holds only in a week the grid's fabs would draw less than this share of their full draw (the wafers on
    # hand against capacity), so a week they would use in full runs; above 1: any week. 0.75 with open_hold_recheck:
    # Small +0.0044 (+0.0026 to +0.0066, 32 / 15), Full +0.0001 (-0.0011 to +0.0012, 6 / 8): smaller losses, and
    # smaller gains
    "open_hold_draw": 2.0,
    # ----- added in agents/mpc_pulse_rec_v9 (pulse.py; it needs "pulse" on) -----
    # Measured against agents/mpc_pulse_rec_v8_on, the folder with this switch off (90 % paired intervals)
    # True: at a pulsed grid, a whole week is delivered the rationed fuel it burns plus the threshold's stock only if
    # the next week can be whole too (this week can close on the threshold and the terminal then holds the rest of a
    # week's burn: the simulator's rule for the rationed segment, on the week's own observation); if not, it is
    # delivered its burn alone and the threshold's worth waits at the terminal instead of burning next week for base
    # load. And a charge is made as soon as one whole week can follow it, without the wait of pulse_run. False: the
    # rules of agents/mpc_pulse_rec_v8_on (every whole week refills the threshold as far as the terminal can; a charge
    # waits for pulse_run whole weeks). No setting of its own: the threshold, the burn and the stocks are the
    # instance's and the observation's. Tuning roots (12345), played once: Full +0.0050 (+0.0024 to +0.0075,
    # 11 episodes better / 5 worse), Small +0.0044 (+0.0022 to +0.0065, 34 / 20). Fresh roots, played once after:
    # Full 2026 +0.0078 (+0.0032 to +0.0131, 14 / 2, one episode carrying half), Full 777 (32 episodes) +0.0046
    # (+0.0020 to +0.0073, 18 / 11), Small 2026 +0.0025 (+0.0007 to +0.0043, 28 / 25)
    "schedule": True,
    # True (with schedule): the charge still waits until the terminal can follow it with pulse_run whole weeks, so
    # only the last week of a run changes. Built, not played
    "schedule_wait": False,
    # debugging: True raises any failure (of the forecast, of the reserve's code, of the pulse's or of the planner) out
    # of act, so a local run counts the week as an error instead of taking the fallback
    "strict": False,
}


NULLABLE = ("h_weeks", "pulse_tail")  # numbers that may be None: the network's own window; the tail rule off
SIGNED = ("pulse_tail",)  # numbers that may be below 0


def same_kind(name, value):
    """Whether a value of ``params.json`` is of its default's kind (a default of None: the module reading it checks)."""
    default = PARAMS[name]
    if default is None or (value is None and name in NULLABLE):
        return True
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, (int, float)):
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        return number and bool(np.isfinite(value)) and (value >= 0 or name in SIGNED)
    return isinstance(value, type(default))


if (Path(__file__).parent / "params.json").is_file():
    given = json.loads((Path(__file__).parent / "params.json").read_text())
    if set(given) - set(PARAMS):  # a name that would be ignored (the two scales of agents/ft_buffer are not kept)
        raise ValueError(f"params.json: no parameter {sorted(set(given) - set(PARAMS))} among {sorted(PARAMS)}")
    odd = sorted(k for k, v in given.items() if not same_kind(k, v))
    if odd:
        defaults = {k: PARAMS[k] for k in odd}
        raise ValueError(f"params.json: {odd} must be of the kind their defaults are: {defaults}")
    PARAMS |= given
check(PARAMS)  # the forecast's kinds and rules
reserve.Settings.of(PARAMS, 1, 1)  # the reserve's, the ramp's, the end value's and the premium's settings
if PARAMS["h_weeks"] is not None and int(PARAMS["h_weeks"]) < 1:
    raise ValueError(f"params.json: h_weeks {PARAMS['h_weeks']!r} is not a number of weeks")


class Planner(MpcDet):
    def reset(self, static, obs, policy_seed):
        super().reset(static, obs, policy_seed)
        self.recovery = Recovery(self._inst, PARAMS) if PARAMS["kinds"] else None
        # weeks planned on what is observed, after a failure of the forecast; weeks planned with the benchmark's own
        # LP, after a failure of the reserve's code
        self.forecast_failed, self.reserve_failed = 0, 0
        self.window = None  # the week's window model and its solution, kept for the pulse's open_hold; None: no plan

    def _record(self, model, res):
        """Keep the week's window plan for ``pulse.Pulse.adjust`` (``open_hold`` reads the fuels it leaves short)."""
        self.window = (model, res.x) if PARAMS["open_hold"] and res.ok else None

    @property
    def failed(self):
        """The fallbacks taken so far, the forecast's and the reserve's."""
        return self.forecast_failed + self.reserve_failed

    def _horizon(self, inst, plan):
        return super()._horizon(inst, plan) if PARAMS["h_weeks"] is None else int(PARAMS["h_weeks"])

    def _window_arrays(self, inst, obs, H_t):
        """The window with the recoveries of ``recovery.py`` in it; a week that fails is planned on what is observed."""
        if self.recovery is None:
            return super()._window_arrays(inst, obs, H_t)
        try:
            self.recovery.update(obs, self._memory)
            return self.recovery.arrays(obs, self._memory, H_t)
        except Exception:  # noqa: BLE001 - any failure of the forecast: this week is planned on persistence
            if PARAMS["strict"]:
                raise
            traceback.print_exc()
            self.forecast_failed += 1
            return super()._window_arrays(inst, obs, H_t)

    def act(self, obs):
        """``MpcDet.act`` with the window's model built by ``reserve.rolled_lp``; a week that fails is planned plain."""
        start = time.perf_counter()
        inst, t = self._inst, int(obs["week"])
        self._memory.update(inst, obs)
        H_t = lp_common.window_length(self._H, t, inst.T)
        arrays = self._window_arrays(inst, obs, H_t)
        rules = self.params.planning_rules
        model = res = None
        try:
            model = reserve.rolled_lp(inst, obs, arrays, H_t, PARAMS, planning_rules=rules)
            res = self._session.solve(lp_common.to_highs_lp(model), lp_common.WindowShape.of(model))
            if not res.ok and reserve.changes(model):
                raise RuntimeError(f"no solution of the window with the reserve: {res.trail}")
        except Exception:  # noqa: BLE001 - any failure of the reserve's code: this week is planned without it
            if PARAMS["strict"]:
                raise
            traceback.print_exc()
            self.reserve_failed += 1
            model = lp_common.rolled_lp(inst, obs, arrays, H_t, planning_rules=rules)
            res = self._session.solve(lp_common.to_highs_lp(model), lp_common.WindowShape.of(model))
        self._record(model, res)
        if res.ok:
            action = lp_common.week1_action(inst, model, res.x, obs, lp_common.prohibited_now(self._memory, t))
        else:
            action = self._fallback.act(obs)
        self.telemetry.append(
            StepTelemetry(t, time.perf_counter() - start, res.iterations, res.trail, fallback=not res.ok)
        )
        return action


def nullable(obs, key, cast=float):
    """A column as a list, None where it is not observed."""
    return [cast(v) if seen else None for v, seen in zip(obs[key].tolist(), obs[f"{key}.observed"].tolist())]


def rows(obs, block, fields):
    """The live entries of a padded list, as columns."""
    live = np.flatnonzero(obs[f"{block}.{fields[0]}.observed"])
    return {f: obs[f"{block}.{f}"][live].tolist() for f in fields}, live


class Agent:
    def __init__(self, config=None):
        static, layout = config["static"], config["layout"]
        self.static, self.seed = static, config["policy_seed"]
        self.planner, self.broken = None, False  # the planner starts on the first observation
        # the episode's pulse (None: off); weeks its code failed and the planner's flows were sent as they were
        self.pulse, self.pulse_failed = None, 0
        self.stock_slots, self.supply_slots = layout["stock_slots"], layout["supply_slots"]
        self.demands = layout["demands"]
        self.fabs, self.grids, self.osats = layout["fabs"], layout["grids"], layout["osats"]
        self.lot_keys = layout.get("lot_keys")  # small and full: the rows of the dense queue_lots block
        edges, lanes, slots = static["edges"], static["lanes"], static["action_slots"]
        if self.lot_keys is not None:  # the edge each lot entered its strait by: the lane's edge before its next edge
            self.lot_entry = []
            for _c, _k, lane, nxt in self.lot_keys:
                along = lanes["edges"][lane]
                self.lot_entry.append(along[along.index(nxt) - 1] if nxt in along[1:] else nxt)
            self.lot_transit = [edges["tau0"][e] for e in self.lot_entry]
        pairs = {tuple(pair): i for i, pair in enumerate(layout["release_pairs"])}
        override = static["override_slots"]
        self.pairs = pairs
        self.override_pair = [pairs[(c, k)] for c, k in zip(override["chokepoint"], override["k"])]
        # the fallback: each slot's nominal capacity and the straits its lane passes
        self.capacity = np.array([edges["u0"][e] for e in slots["edge"]], dtype=float)
        position = {node: i for i, node in enumerate(layout["chokepoints"])}
        passes = lanes["chokepoints"]
        self.through = [[position[c] for c in passes[lane]] if lane is not None else [] for lane in slots["lane"]]

    def act(self, observation):
        if not self.broken:
            try:
                obs = self.wire(observation)
                if self.planner is None:
                    planner = Planner(MpcDetParams(), PolicyContext())
                    planner.reset(self.static, obs, self.seed)
                    self.planner = planner
                    self.pulse = self.new_pulse(planner._inst)
                return self.action(self.pulsed(obs, self.planner.act(obs)))
            except Exception:  # noqa: BLE001 - any failure of the planner: this week sends the maximum
                if PARAMS["strict"]:
                    raise
                traceback.print_exc()
                self.broken = self.planner is None  # a planner that never started is not tried again
        return self.send_max(observation)

    def new_pulse(self, inst):
        """The episode's ``pulse.Pulse``; None when it is off or cannot be built (the planner's flows then stand)."""
        if not PARAMS["pulse"]:
            return None
        try:
            return pulse.Pulse(inst, PARAMS)
        except Exception:  # noqa: BLE001 - any failure of the pulse's code: the episode is played without it
            if PARAMS["strict"]:
                raise
            traceback.print_exc()
            self.pulse_failed += 1
            return None

    def pulsed(self, obs, action):
        """The planner's action with short grids' fuel held or delivered by ``pulse.Pulse``; as it was on failure."""
        if self.pulse is None:
            return action
        try:
            memory = self.planner._memory
            banned = lp_common.prohibited_now(memory, int(obs["week"]))
            return self.pulse.adjust(obs, memory.values, banned, action, self.planner.window)
        except Exception:  # noqa: BLE001 - any failure of the pulse's code: the planner's flows stand
            if PARAMS["strict"]:
                raise
            traceback.print_exc()
            self.pulse_failed += 1
            return action

    def wire(self, o):
        """The observation the planner reads (the scorer's own layout), rebuilt from the agent's arrays."""
        blackout = o["action_mask.observed"][0] == 0
        keep = np.flatnonzero(o["stock.qty.observed"])
        stock = {
            "node": [self.stock_slots[i][0] for i in keep],
            "k": [self.stock_slots[i][1] for i in keep],
            "qty": o["stock.qty"][keep].tolist(),
        }
        keep = np.flatnonzero(o["backlog.qty.observed"])
        backlog = {
            "node": [self.demands[i][0] for i in keep],
            "k": [self.demands[i][1] for i in keep],
            "qty": o["backlog.qty"][keep].tolist(),
        }
        pipeline, live = rows(o, "pipeline", ("qty", "edge", "k", "arrival_week"))
        pipeline["lane"] = nullable(o, "pipeline.lane", int)
        pipeline["lane"] = [pipeline["lane"][i] for i in live]
        wip, _live = rows(o, "wip", ("qty", "node", "k", "out_week"))
        pending, live = rows(o, "pending_prohibitions", ("edge", "k", "effective_week"))
        d, h = np.nonzero(o["demand_forecast.qty.observed"])
        forecast = {
            "node": [self.demands[i][0] for i in d],
            "k": [self.demands[i][1] for i in d],
            "h": h.tolist(),
            "qty": o["demand_forecast.qty"][d, h].tolist(),
        }
        return {
            "week": int(o["week"][0]),
            "stock": stock,
            "backlog": backlog,
            "pipeline": pipeline,
            "queue_lots": self.lots(o),
            "wip": wip,
            "graph_now": self.graph(o),
            "demand_forecast": forecast if d.size else None,
            "pending_prohibitions": None if blackout and not live.size else pending,
        }

    def lots(self, o):
        fields = (
            "qty",
            "lot_id",
            "chokepoint",
            "k",
            "lane",
            "next_edge",
            "arrival_week",
            "dispatch_week",
            "entry_edge",
        )
        if self.lot_keys is None:  # tiny: the lots as a list
            return rows(o, "queue_lots", fields)[0]
        key, week = np.nonzero(o["queue_lots.qty.observed"])  # row: a (strait, k, lane, next edge); column: week - 1
        order = np.lexsort((key, week))
        key, week = key[order].tolist(), (week[order] + 1).tolist()
        out = {f: [] for f in fields}
        for i, (r, w) in enumerate(zip(key, week)):
            c, k, lane, nxt = self.lot_keys[r]
            lot = (
                float(o["queue_lots.qty"][r, w - 1]),
                i,
                c,
                k,
                lane,
                nxt,
                w,
                w - self.lot_transit[r],
                self.lot_entry[r],
            )
            for f, v in zip(fields, lot):
                out[f].append(v)
        return out

    def graph(self, o):
        if o["override_mask.observed"][0] == 0:  # a blackout week: the planner keeps what it saw last
            return None
        keep = np.flatnonzero(o["graph_now.supply.avail.observed"])
        e, k = np.nonzero(o["graph_now.prohibited"])
        te, tk = np.nonzero(o["graph_now.tariff"])
        war = zip(o["graph_now.war_risk"].tolist(), o["graph_now.war_risk.observed"].tolist())
        return {
            "u": nullable(o, "graph_now.u"),
            "c": nullable(o, "graph_now.c"),
            "tau": nullable(o, "graph_now.tau", int),
            "open": nullable(o, "graph_now.open"),
            "kappa": {pool: nullable(o, f"graph_now.kappa.{pool}") for pool in ("tb", "ct")},
            "war_risk": [WAR_RISK_CLASSES[code] if seen else None for code, seen in war],
            "supply": {
                "node": [self.supply_slots[i][0] for i in keep],
                "k": [self.supply_slots[i][1] for i in keep],
                "avail": o["graph_now.supply.avail"][keep].tolist(),
            },
            "fab": {"node": self.fabs} | {f: nullable(o, f"graph_now.fab.{f}") for f in ("R", "alpha_bar", "cap_eff")},
            "grid": {"node": self.grids} | {f: nullable(o, f"graph_now.grid.{f}") for f in ("G_bar", "y_bar")},
            "osat": {"node": self.osats} | {f: nullable(o, f"graph_now.osat.{f}") for f in ("R", "thr_eff")},
            "prohibited": {"edge": e.tolist(), "k": k.tolist()},
            "tariff": {"edge": te.tolist(), "k": tk.tolist(), "rate": o["graph_now.tariff"][te, tk].tolist()},
        }

    def action(self, wire):
        """The planner's action as the agent's arrays."""
        flows = np.zeros(len(self.capacity))
        sent = wire.get("flows") or {"slot": [], "qty": []}
        flows[sent["slot"]] = sent["qty"]
        override, mode = np.zeros(len(self.override_pair)), np.zeros(len(self.pairs), dtype=np.int64)
        released = wire.get("overrides") or {"slot": [], "qty": []}
        for slot, qty in zip(released["slot"], released["qty"]):
            override[slot], mode[self.override_pair[slot]] = qty, 1
        held = wire.get("hold") or {"chokepoint": [], "k": []}
        for c, k in zip(held["chokepoint"], held["k"]):
            mode[self.pairs[(c, k)]] = 2
        return {"flows": flows, "override_qty": override, "release_mode": mode}

    def send_max(self, o):
        """The kit heuristic: every allowed route at capacity, a lane through a strait times its open fraction."""
        flows = self.capacity * o["action_mask"]
        open_now, seen = o["graph_now.open"], o["graph_now.open.observed"] == 1
        for s, straits in enumerate(self.through):
            for c in straits:
                if seen[c]:
                    flows[s] *= max(float(open_now[c]), 0.0)
        return {"flows": flows}
