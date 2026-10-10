"""``mpc_det`` with a recovery forecast, the LP changes of ``reserve.py`` and fuel pulsed at grids short of it.

Each week it plans ``h_weeks`` weeks and plays the first. It is ``agents/mpc_pulse_rec_v12`` with four additions,
built separately on that agent and merged here (the numbers are at the switches in ``PARAMS``; with the new switches
false and the guard's limit not reached the folder plays as that agent does, to the cent):

- ``strait_reserve`` (``reserve.py``), on: the fuel reserve goes on beyond what a grid and its terminal can store, as
  tanker cargo held at the last strait before the terminals, so that a strait, a route or a source that closes
  upstream is bridged for months and not for weeks. From a comparison of the worst Full episodes with the
  clairvoyant plan, which parks fuel there before a long closure. ``reserve_sources``, built with it, is off.
- ``mild_start`` (``pulse.py``), on: a grid whose fabs get half their draw or more is pulsed all the same once what
  they have missed is worth the dark week a start costs. From a census of every week the pulse leaves a grid as it
  is (``pulse.Pulse.census``, a record no rule reads). ``mild_ends``, built with it, is off.
- three rules about a run of whole weeks (``pulse.py``), all off as measured: ``ends_needed`` (a run at a grid that
  is not pulsed is ended only where the end buys back more than its dark week costs), ``runs_reach`` (a charge waits
  only for a run the arrivals can fill) and ``wafers_plan`` (the wafer top-up aims at the lots the window plan
  itself starts, not at a week of capacity; it gains on Small and not on Full).
- two changes to this file that concern the week's CPU and nothing else. ``guard``: a week over the CPU budget is
  played by the naive rule, and one such week costs about 0.007 to 0.011 of the score. The week's solve therefore
  stops at a share of the budget, and the week then plays the plan solved the week before (through the pulse), or,
  where there is none, the first week of a short window solved in what is left of the budget; the solve that was
  cut is taken up again in the weeks after, from the basis it reached. ``preload``: the instance and its naive plan
  are built when this module is imported (the container's start, which is not metered) from the public instances
  in ``instances/``, so the first week does not pay for them.

That agent is ``agents/mpc_pulse_rec_v11`` with three additions to
``pulse.py``, built separately on that agent and merged here; the numbers are at the switches in ``PARAMS``
(``reserve.py``, ``recovery.py`` and ``sbflow/`` are that agent's files, byte for byte):

- ``pipe_pulse`` and ``lean_hold``, both on: two rules for where the pulse stood down. A grid short of a fuel that
  "cannot be held" was not pulsed at all, and its fabs stood dark for as long as the shortage lasted.
  ``pipe_pulse``: a fuel that comes straight from a supply node can be held after all, at the supply node, which
  keeps what is not shipped up to its storage; the fuel is then shipped in lumps, decided a week ahead.
  ``lean_hold``: a fuel that is not worth a full tank at its terminal (``pulse_payback``) is held up to one charge
  instead.
- ``hold_checks``, the list of the checks that are on ("together" and "shock" here): two rules of
  ``agents/mpc_pulse_rec_v11`` held fuel for a whole week that did not need it (``pulse.py``, "Three checks added in
  agents/mpc_pulse_rec_v12b"). They came from the episodes of a Small set of our own (root 6006, 128 episodes) in
  which that agent lost 17 to 39 bn USD against ``agents/mpc_pulse_rec_v10``.
- ``fuel_cancel_tariff``, on: fuel sent from a source straight into a grid on a route that pays a tariff this week is
  cut to what the grid would lack if its largest route were cut (``Pulse.cancel_unburnable``).

With ``pipe_pulse``, ``lean_hold`` and ``fuel_cancel_tariff`` false and ``hold_checks`` empty the folder plays as
``agents/mpc_pulse_rec_v11`` does, to the cent. That agent is ``agents/mpc_pulse_rec_v10`` with five rules added
to ``pulse.py``, each behind its own switch (``hold_together``, ``pulse_hold_sales``, ``shock_hold``,
``schedule_ends`` and ``schedule_runs``: five more entries of ``PARAMS``) and, in a second stage, a pair of rules
from ``agents/mpc_pulse_rec_v10_kr``: ``premium_whole`` in ``reserve.py`` (that agent's file; it is
``agents/mpc_pulse_rec_v10``'s with that rule added) and ``hold_small`` in ``pulse.py``. All seven are on.
``recovery.py`` and ``sbflow/`` are ``agents/mpc_pulse_rec_v10``'s files, byte for byte, and with the seven switches
false the folder plays as ``agents/mpc_pulse_rec_v10`` does, to the cent. That agent is ``agents/mpc_pulse_rec_v9``
(the agent uploaded on 2026-10-08) with four rules added to ``pulse.py``, each behind its own switch
(``schedule_fuels`` and ``schedule_hold``, on; ``schedule_left`` and ``schedule_wafers``, off: four more entries of
``PARAMS``); with those four switches false besides the folder plays as ``agents/mpc_pulse_rec_v9`` does, to the cent.
That agent is ``agents/mpc_pulse_rec_v8_on`` (the agent uploaded on 2026-10-07) with ``open_hold`` on in ``PARAMS``
itself (the folder has no ``params.json``) and one rule added to ``pulse.py``, switched by ``schedule`` (two more
entries of ``PARAMS``): with ``schedule`` false besides the folder plays as ``agents/mpc_pulse_rec_v8_on`` does, to
the cent. That agent is ``agents/mpc_pulse_rec_v7`` with one rule added to
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

One rule came with ``agents/mpc_pulse_rec_v9`` (``schedule``; ``pulse.py``): at a pulsed grid the last whole week of
a run is delivered the rationed fuel it burns and no more, and the charge that starts a run is made as soon as one
whole week can follow. Both tests are the simulator's own rule for the rationed segment applied to what the week's
observation shows (the stock at the grid and at the terminal, the week's arrivals): no forecast and no setting. It
replaces the wait of
``pulse_run`` and the refill of the threshold in a whole week the next cannot follow. Where it comes from: the best
schedule of terminal-to-grid deliveries, found with hindsight in an exact model of the grid rule, ran its whole weeks
this way; re-solving that schedule every week from forecasts did not beat the hand rules, and this part of it needs
none (the numbers are at ``schedule`` in ``PARAMS``).

Four rules came with ``agents/mpc_pulse_rec_v10``, all to that one and all from its losing episodes (``pulse.py``; the
numbers are at the switches in ``PARAMS``). Two are on. ``schedule_fuels``: a week is whole only if every fuel's segment
gives its share, and the tests of ``schedule`` looked at the rationed fuel alone; now a charge is made, and a run goes
on, only if the rationed segment can stay full until the week the grid's other fuels next allow a whole week. At a grid
short of gas and of crude the gas was charged for weeks the crude then left dark. ``schedule_hold``: the test that makes
the pulse stand down where a terminal fills too slowly counts a whole week as ``schedule`` delivers it (the larger of
the threshold and the burn, not their sum), so a grid whose arrivals have slowed keeps its fuel for a whole week instead
of leaving it to the planner's flows. Two are off, as measured. ``schedule_left``: the charge waits for ``pulse_run``
whole weeks again where a run leaves stock at the grid (the threshold is above the week's burn). ``schedule_wafers``: a
charge is made, and a run goes on, only while the fabs will hold next week the wafers the top-up aims at, where the fuel
is the scarcer of the two. One more lead, a search over the deliveries of the last sellable weeks, was examined in the
grid model and not built: with hindsight and free from six weeks before the last sellable week it is worth +0.0014
against ``agents/mpc_pulse_rec_v9``, +0.0007 when re-solved each week from the plan's forecast and +0.0001 when only
shipments under way are counted as arrivals.

Five rules are added here (``pulse.py``; the numbers are at the switches in ``PARAMS``). They came from a comparison
of the worst Full episodes with the clairvoyant plan, grid by grid, and with a bound on what any agent can reach when
base load is served before fabs. Most of what those episodes lose is out of reach (one grid without gas for the
whole episode, which the plan powers by shedding base load); what is within reach is fuel the pulse had and burnt
for base load. ``hold_together``: while a pulsed grid waits for one fuel, its other scarce fuels wait at their
terminals too, so that the whole week finds them all (``schedule_fuels`` times the charge by what they then hold).
``pulse_hold_sales``: a tank may take as long to fill as lots can still be sold, where ``pulse_hold_weeks`` made the
pulse stand down and the trickle burn for base load. ``shock_hold``: through an energy shock at a pulsed grid the
scarce fuels stay at the terminal for the whole weeks after it. ``schedule_ends``: a run of whole weeks longer than
``pulse_memory`` ends as cleanly as a short one. ``schedule_runs``: where the rationing threshold is above a week's
burn, the charge waits for as long a run as the terminal's storage carries. A sixth rule of that analysis, the count
of what a whole week takes in the stand-down test, was already here in a more exact form (``schedule_hold``) and
was not added again. All five are on. Together, against ``agents/mpc_pulse_rec_v10`` on roots of our own (64 Full
episodes of three roots, 128 Small episodes of two; 90 % paired intervals): Full +0.0045 (+0.0030 to +0.0063), Small
+0.0030 (+0.0014 to +0.0046).

The second stage (``premium_whole`` in ``reserve.py``, ``hold_small`` in ``pulse.py``; one mechanism, switched on or
off together) is another agent's follow-up of a lead of that comparison: on an edge a terminal's small fuel shares
with its scarce one, the small fuel was priced per unit of the fabs' draw in every week and took the edge first, for
energy the fabs only see in a whole week. ``premium_whole`` frees a small fuel's shortfall of the premium up to the
share of its burn that the plan leaves the grid's other fuels short by; ``hold_small`` keeps a scarce small fuel at
the terminal in the weeks that are not whole, so that the whole week finds what the plan then sends. The pair is
on: on top of the five rules Full +0.0048 (+0.0031 to +0.0065), Small +0.0024 (+0.0010 to +0.0039); all seven against
``agents/mpc_pulse_rec_v10`` Full +0.0093 (+0.0067 to +0.0121), Small +0.0054 (+0.0037 to +0.0071).

The rest is ``agents/mpc_det_h20``: the planner is shockbench-flow's code (MIT) in ``sbflow/``
(``scripts/vendor_planner.py``, unchanged), solved with the HiGHS that SciPy bundles, and this file translates between
the agent's arrays and the observation the planner reads. It reads no warnings and no announcements.

``PARAMS`` holds the settings, as measured best. A ``params.json`` beside this file replaces entries of it (``rules``
as a whole); a name ``PARAMS`` does not have, a value of another kind than its default, or one the forecast or the
reserve refuses is an error when the agent is loaded. With ``pipe_pulse``, ``lean_hold`` and
``fuel_cancel_tariff`` false and ``hold_checks`` empty it plays as
``agents/mpc_pulse_rec_v11``; with ``premium_whole``, ``hold_small``, ``hold_together``,
``pulse_hold_sales``, ``shock_hold``, ``schedule_ends`` and ``schedule_runs`` false besides as
``agents/mpc_pulse_rec_v10``; with ``schedule_fuels``,
``schedule_hold``, ``schedule_left`` and ``schedule_wafers`` false besides as ``agents/mpc_pulse_rec_v9``; with
``schedule`` false besides as
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
from sbflow.policies.naive_parts import cached_plan
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
    # ----- added in agents/mpc_pulse_rec_v13 (reserve.py): the reserve at straits -----
    # True: the fuel reserve goes on beyond the storage of a grid and its terminal, as tanker cargo held at the last
    # strait before the terminals (a strait's queue has no limit, and cargo waits there until the agent releases
    # it). Where a fuel reaches a strait through another strait, the plan is asked to keep strait_weeks weeks of what
    # that strait can pass on to the grids behind it (each edge into a grid's terminal at its capacity in the window,
    # up to the burn it feeds, within the strait's throughput), at reserve_price, tapering to nothing at the
    # episode's end, and not in a week the queue is charged at a war-risk rate. It is filled with fuel no grid burns
    # in the window, after the terminals, and it bridges what closes upstream of it: a strait, the route to it or
    # the source. False: the reserve ends at the terminals, which hold a few weeks of burn; the clairvoyant plan
    # bridges a strait shut for a year with 30,000 GWh of gas and 34,000 of crude parked beyond it beforehand. It
    # names no strait, grid or fuel. Measured against agents/mpc_pulse_rec_v12 (this folder with the switch off, to
    # the cent), strict plays on roots of our own, 90 % paired intervals, episodes better / worse. With 26 weeks:
    # Full 12345 (16 episodes) +0.0012 (-0.0001 to +0.0025, 12 / 4), Full 2026 (16) +0.0010 (-0.0015 to +0.0041,
    # 9 / 7), Full 777 (32) +0.0078 (+0.0034 to +0.0136, 22 / 10), the 64 as one 0.8830 to 0.8875, +0.0046 (+0.0018
    # to +0.0080, 43 / 21; 937 bn USD saved); Small 12345 and 2026 (128) 0.8892 to 0.8946, +0.0054 (+0.0032 to
    # +0.0079, 91 / 37). With 8 weeks: Full +0.0036 (+0.0017 to +0.0060, 46 / 18), Small +0.0053 (+0.0037 to
    # +0.0070, 94 / 34). The gain sits where a strait is cut after the episode has begun: one Full episode (a strait
    # at 0.27 from week 24 to 87) gains 268 bn with 26 weeks and 183 with 8. What it costs: the queue's holding
    # (0.4 to 0.8 bn an episode with 8 weeks), and fuel that waits at a strait which then shuts under war risk pays
    # that rate until it gets through. Its largest loss, 58 bn in one Full episode, is not the reserve's own: the
    # plan cut another supply of a grid whose stores were full (see reserve_sources). On; its play on episodes
    # nothing was chosen on is in experiments/pulse_agent/README.md, at this version's entry
    "strait_reserve": True,
    "strait_weeks": 26.0,  # the weeks of cover asked at a strait (8, reserve_weeks' value, was the first played)
    # True: the cover that neither a grid nor its terminal can hold is also asked of the supply nodes with an edge
    # straight into the grid, each up to its storage (a supply node keeps what is not shipped, at no cost, and what
    # it cannot keep is lost). The plan then cuts, of two supplies of a grid whose stores are full, the one whose
    # fuel waits at its source, not the one that saves freight. False: stock at a supply node is worth nothing to
    # the plan. Off as measured: alone, Full 2026 -0.0031 (-0.0055 to -0.0015, 2 / 14); on top of strait_reserve
    # with 8 weeks, over the 64 Full episodes +0.0001 (-0.0006 to +0.0007, 29 / 34). It does take back the one
    # large loss it was built for (+60 bn in that episode; the largest loss of the pair is 14 bn, against 58)
    "reserve_sources": False,
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
    # ----- added in agents/mpc_pulse_rec_v10 (pulse.py; they need "schedule" on) -----
    # Measured against agents/mpc_pulse_rec_v9, the folder with these four switches off (90 % paired intervals;
    # episodes better / worse). None has a setting of its own. The two that are on, together: tuning roots (12345)
    # Full +0.0018 (+0.0007 to +0.0031, 8 / 2), Small +0.0039 (+0.0022 to +0.0057, 24 / 13); fresh roots, played once
    # after: Full 2026 +0.0037 (+0.0015 to +0.0067, 7 / 5), Full 777 (32 episodes) +0.0062 (+0.0030 to +0.0098,
    # 19 / 3), Small 2026 +0.0030 (+0.0016 to +0.0046, 17 / 13)
    # True: the charge is made at once only where a run leaves nothing at the grid, which is where the rationing
    # threshold is no more than the week's burn of the fuel. Where the threshold is above it, the stock a run's last
    # week leaves (threshold less burn) burns for base load alone, once per run: there the charge waits until the
    # terminal can follow it with pulse_run whole weeks, as agents/mpc_pulse_rec_v8_on did (in agents/
    # mpc_pulse_rec_v9 a grid with a threshold of 1.2 weeks of burn ran one-week runs and lost a sixth of its gas to
    # that stock). The threshold and the burn are the instance's and the observed grid power's. Off as measured:
    # alone Full +0.0001 (-0.0008 to +0.0013, 1 / 3), Small +0.0001 (-0.0007 to +0.0008, 6 / 8); on top of the two
    # that are on (with the first form of schedule_fuels) Full +0.0001 (-0.0001 to +0.0004), Small -0.0009 (-0.0014
    # to -0.0004)
    "schedule_left": False,
    # True: a charge is made, and a run goes on into the next week, only if the fabs will then hold the wafers the
    # top-up aims at (pulse_wafers weeks of capacity at every fab it serves: counted are the wafers left after this
    # week's lots, the shipments due next week and what is sent this week over an edge one week long, the grid's
    # top-up of the week included). A whole week burns nearly the same fuel whatever the fabs draw, so fuel that
    # is short is kept for a week that finds the wafers. Asked only where the fuel is the scarcer of the two: its
    # smoothed arrivals, as a share of the week's burn, are below the wafers' as a share of capacity (where the
    # wafers are the scarcer, every wafer is processed whatever the weeks hold). Not when the terminal is full, and
    # not when next week is the last a lot can be sold in. Off as measured: it acts in few episodes, Full -0.0001
    # (-0.0002 to +0.0000, 1 / 1), Small +0.0001 (-0.0001 to +0.0004, 3 / 3)
    "schedule_wafers": False,
    # True: the test that makes the pulse stand down where a terminal fills too slowly (pulse_hold_weeks) counts a
    # whole week as the schedule delivers it: the larger of the threshold and the week's burn (and what the grid's
    # stock burns meanwhile at the rationed rate), where it counted their sum. A grid whose arrivals have slowed
    # then keeps a whole week's fuel for a whole week; counted the old way it can stand down and leave that fuel
    # to the planner's flows, which burn it for base load. Alone: Full +0.0003 (-0.0000 to +0.0006, 4 / 4), Small
    # +0.0025 (+0.0015 to +0.0038, 18 / 13, the losses -0.0002 in all)
    "schedule_hold": True,
    # True: a charge is made, and a run goes on, only if the rationed segment can stay full until the week the
    # grid's other fuels next leave a whole week possible. That week is worked out from this week's flows: each
    # other fuel's stock at the grid and what its terminal can then send (filled by its forecast arrivals for the
    # weeks after the next), against its week's burn, by the test the pulse applies when the week comes (a small
    # fuel's shortfall lowers the fabs' draw, any other's beyond the grid's spare power makes the week not whole).
    # If that week is the next one the test is that of "schedule"; if it is later, the terminal must also hold what
    # the weeks between take to close on the threshold. A grid short of two fuels charged its gas for weeks the
    # other fuel then left dark: the charge burnt for base load and was gone when the other fuel came. Alone: Full
    # +0.0016 (+0.0004 to +0.0029, 5 / 1), Small +0.0011 (+0.0001 to +0.0022, 10 / 6). What it risks: fuel kept
    # back sheds base load that week, and is worth it only while the shortage lasts (at the grid it acts on most, a
    # week of that shed costs three whole weeks' chips)
    "schedule_fuels": True,
    # ----- added in agents/mpc_pulse_rec_v11 (pulse.py; they need "pulse" on, the last two "schedule" too) -----
    # None has a setting of its own. Measured on roots of our own (90 % paired intervals; episodes better / worse).
    # The five together against agents/mpc_pulse_rec_v10 (this folder with its seven new switches off): Full 12345
    # +0.0052 (+0.0006 to +0.0106, 10 / 6), Full 2026 +0.0030 (+0.0012 to +0.0048, 10 / 6), Full 777 (32 episodes)
    # +0.0049 (+0.0031 to +0.0068, 24 / 8), the three as one +0.0045 (+0.0030 to +0.0063); Small 12345 +0.0035
    # (+0.0017 to +0.0052, 34 / 21), Small 2026 +0.0027 (-0.0000 to +0.0057, 26 / 27), the two as one +0.0030 (+0.0014
    # to +0.0046). At each switch: what the five lose without it (Full: roots 12345 and 777 as one, 48 episodes;
    # Small: root 12345, 64 episodes) unless it names other sets
    # True: in a week a pulsed grid waits for one fuel, every other fuel that can take all the fabs' energy, has a
    # terminal and is worth a tank waits there too, if it was itself short or held in the last pulse_memory weeks, or
    # has no rationing threshold and reaches the terminal more slowly than the grid burns it. False: such a fuel is
    # delivered as the planner says whenever it can run in full, and burns for base load in the weeks the other fuel
    # is held (at a grid short of gas and of crude the two had their full weeks at different times). Full +0.0029
    # (+0.0014 to +0.0046, 15 / 6), Small +0.0013 (-0.0001 to +0.0027, 17 / 11). What it risks: the fuel that waits
    # sheds base load meanwhile (its largest losses: 35 bn USD in one Full episode, 33 in one Small episode)
    "hold_together": True,
    # True: the pulse stands down at a grid only if its tank cannot gather a whole week while a lot can still be sold;
    # False: also when it would take more than pulse_hold_weeks (the trickle then burns for base load every week).
    # It acts in few episodes. Next to the other four: Full +0.0001 (-0.0000 to +0.0004, 9 / 22), Small -0.0000 (four
    # episodes, 0.1 bn USD in all). Next to the other six (three Full roots, 64 episodes; two Small roots, 128): Full
    # +0.0004 (+0.0001 to +0.0008, 14 / 24; one episode gains 58 bn USD, no loss is above 8), Small +0.0000 (-0.0000
    # to +0.0002, 1 / 12)
    "pulse_hold_sales": True,
    # True: in a week a pulsed grid's own power does not reach past its base load while its fabs hold wafers (an
    # energy shock), the fuels the pulse can hold and that are scarce stay at the terminal, for the whole weeks after
    # the shock; False: the planner's flows stand, and what the pulse had gathered burns for base load. Full +0.0014
    # (+0.0005 to +0.0025, 18 / 7), Small +0.0009 (+0.0001 to +0.0019, 4 / 2); its largest loss 21 bn USD (Full)
    "shock_hold": True,
    # True: the last whole week of a run is delivered its burn alone also at a grid that no longer counts as pulsed
    # (pulse_memory weeks after its last held week), when its rationed fuel reaches the terminal more slowly than
    # the grid burns it and the terminal cannot cover the next week; False: such a run ends with a partly powered
    # week and a week in which the stock left at the grid burns for base load. Full +0.0023 (+0.0009 to +0.0040,
    # 26 / 16), Small +0.0028 (+0.0016 to +0.0041, 33 / 18)
    "schedule_ends": True,
    # True: where the rationing threshold is above the week's burn of the fuel (every run then leaves the difference
    # at the grid to burn for base load), the charge waits until the terminal carries as long a run of whole weeks
    # as its storage holds (threshold + (n - 1) burn within pulse_fill of it; no more than can still be sold; a
    # full terminal starts the run); False: it is made as soon as one whole week can follow (schedule_left, off,
    # waits for pulse_run weeks instead). Over three Full roots (64 episodes) +0.0005 (+0.0001 to +0.0009, 18 / 11),
    # over two Small roots (128) +0.0007 (+0.0000 to +0.0016, 15 / 17); on Full 12345 alone -0.0003 (-0.0011 to
    # +0.0003, 2 / 3)
    "schedule_runs": True,
    # ----- second stage of agents/mpc_pulse_rec_v11: a pair from agents/mpc_pulse_rec_v10_kr, on or off together -----
    # Measured as a pair on top of the five rules above (same roots, same intervals): Full 12345 +0.0065 (+0.0027 to
    # +0.0103, 8 / 8), Full 2026 +0.0045 (+0.0027 to +0.0067, 12 / 4), Full 777 +0.0039 (+0.0017 to +0.0061, 21 / 11),
    # the three as one +0.0048 (+0.0031 to +0.0065); Small 12345 +0.0018 (+0.0002 to +0.0035, 28 / 36), Small 2026
    # +0.0033 (+0.0008 to +0.0060, 28 / 36), the two as one +0.0024 (+0.0010 to +0.0039): on Small more episodes lose
    # than gain, by less. All seven against agents/mpc_pulse_rec_v10: Full +0.0093 (+0.0067 to +0.0121, 47 / 17),
    # Small +0.0054 (+0.0037 to +0.0071, 69 / 59)
    # True (reserve.py): a small fuel's premium (premium_small: per unit of the fabs' draw) counts only in the weeks
    # the grid can be whole. In every window week its shortfall is free of the premium up to the share of its burn
    # that the plan leaves the grid's other fuels short by in that week (a fuel that can take all the fabs' energy
    # and is short by a share s leaves about 1 - s of the weeks whole under the pulse, and in the others the small
    # fuel gives the fabs nothing). False: the premium on every unit; there a small fuel on an edge it shares with
    # the scarce fuel took its whole burn of the edge first, every week. No setting of its own
    "premium_whole": True,
    # True (pulse.py): in a week that is not whole at a pulsed grid, a small fuel (one whose segment is smaller than
    # the fabs' draw) that reaches its terminal more slowly than the grid burns it waits there too, also in a week
    # it could run in full. False: it is held only in a week it is short of a week's burn, and burns for base load
    # in the other weeks, the charge week before a whole week among them
    "hold_small": True,
    # ----- added in agents/mpc_pulse_rec_v12a (pulse.py; it needs "pulse" on) -----
    # True: a fuel that reaches a grid straight from a supply node over an edge one week long (a pipe) is pulsed as
    # a terminal's fuel is. A supply node keeps what is not shipped, up to its storage, so the fuel can wait there;
    # only its delivery is decided a week ahead, from the stock the grid closes the week on. At a grid whose fabs are
    # dark for want of such a fuel, the pipes hold back what the next charge takes, send the charge (the lump that
    # closes a week on the rationing threshold) once a whole week can follow, and then a week's burn at a time while
    # the supply nodes and the grid's other fuels carry the run; the grid counts as pulsed, so its other scarce
    # fuels wait at their terminals for the whole weeks. False: the pulse stands down at a grid short of a piped
    # fuel, and its fabs stay dark for as long as the pipe carries less than the week's burn. No setting of its
    # own (the test of whether whole weeks pay for the fuel held back is pulse_payback's, on what a charge holds
    # back at the supply nodes). Against agents/mpc_pulse_rec_v11, strict plays on the Full sets of roots 12345
    # (16 episodes), 2026 (16) and 777 (32), the 64 as one: +0.0070 (90 % interval +0.0045 to +0.0097), 28 episodes
    # better and 5 worse (the largest loss 9.0 bn), 1,509 bn saved. On Small no pipe pays or can carry a charge:
    # the 64 episodes of root 12345 cost what they did, to the cent
    "pipe_pulse": True,
    # True: a fuel that pulse_payback leaves out (the fabs' whole weeks do not pay for a full tank of it) is held at
    # its terminal all the same, but lean: the charge is made as soon as one whole week can follow it, and while the
    # charge waits for the grid's other fuels the terminal keeps what the next charge takes and sends the rest on.
    # What is then at risk is one charge (the larger of the threshold and a week's burn, not their sum), and the
    # fuel is held only if pulse_payback whole weeks pay for that. False: a grid short of such a fuel is not pulsed
    # at all, and its fabs stay dark while the shortage lasts. No setting of its own. With pipe_pulse on, against
    # pipe_pulse alone, on the same 64 Full episodes: +0.0021 (+0.0009 to +0.0034), 14 better and 5 worse, 542 bn
    # saved; the largest loss is 46.2 bn, an episode where the chips of the whole weeks it made were not short (the
    # planner then sent fewer wafers to a fab of the same product that sheds nothing). The two rules together:
    # 0.8730 to 0.8821, +0.0091 (+0.0066 to +0.0120), 39 better and 6 worse. On Small one charge pays at no grid,
    # so the rule does nothing there
    "lean_hold": True,
    # ----- added in agents/mpc_pulse_rec_v12c (pulse.py; it needs "fuel_cancel" on and only lowers flows) -----
    # True: a dispatch from a source straight into a grid over a route that pays a tariff this week is cut so that the
    # grid's stock, the fuel under way and the week's dispatches do not exceed what the grid would lack to the
    # episode's end if its largest route were cut from next week on: the full-rate burn of the weeks left, less what
    # its other routes can still deliver in time at this week's observed capacity. The cheapest route is kept first;
    # a route without a tariff is cut as fuel_cancel cuts it, and at a grid with one open route nothing changes.
    # False: fuel_cancel alone (fuel is bought on a tariffed route at full capacity until the stock covers the weeks
    # left). No setting of its own; it names no node and no fuel. Measured against agents/mpc_pulse_rec_v11, the
    # folder with this switch off (90 % paired intervals; episodes better / worse / equal to the cent): Full 12345
    # +0.0007 (+0.0002 to +0.0012, 8 / 0 / 8), Full 2026 +0.0005 (+0.0004 to +0.0006, 10 / 0 / 6), Full 777 (32
    # episodes) +0.0007 (+0.0004 to +0.0012, 16 / 0 / 16), the three as one +0.0007 (+0.0004 to +0.0009, 34 / 0 / 30;
    # 143 bn USD saved in all); Small 12345 and 2026 (128 episodes): every episode equal to the cent, as each grid
    # there has one such route. Every Full episode that changes gains (17 bn USD at most), each within 1 bn of what
    # a ledger of that fuel's freight, tariff and holding said beforehand. What it risks: fuel not bought on the
    # tariffed route is missing if the grid's routes are all cut later for long (no such episode in the 64 played,
    # none in the ledger's 464 scenarios)
    "fuel_cancel_tariff": True,
    # ----- added in agents/mpc_pulse_rec_v12b (pulse.py): one switch, a list of names as "kinds" is -----
    # The checks that are on, of "together", "ends" and "shock" (pulse.CHECKS); []: none, and the folder plays as
    # agents/mpc_pulse_rec_v11. Each makes one rule of that agent act only where the week it acts for needs it, by a
    # test of the pulse's own (no setting):
    # "together": hold_together holds the rationed fuel in a week it runs in full only if its run cannot last until
    #   the week the grid's other fuels next allow a whole week (the test of schedule_fuels);
    # "shock": shock_hold holds a fuel only while the grid and its terminal hold less than the weeks a lot can still
    #   be sold in can burn beyond their own arrivals;
    # "ends" (OFF, as measured): schedule_ends ends a run at a grid that is not pulsed only where what the fabs
    #   would lack next week as things are, taken to last every week a lot can still be sold in, is worth more than
    #   the dark week the end of a run costs. On the Small set it was built on (root 6006, 128 episodes, against
    #   agents/mpc_pulse_rec_v11; one play each, bn USD) it took 42.7 off the eight episodes it was designed on and
    #   added 97.0 in six others (12 to 25 each; the two traced were at one grid whose terminal was running down:
    #   the run ended anyway, later and with two dark weeks where the rule's end costs one, once in the last weeks
    #   a lot can be sold in). All three checks together on that set: +0.0006 (-0.0004 to +0.0016), 48 episodes
    #   better / 38 worse
    "hold_checks": ["together", "shock"],
    # ----- added in agents/mpc_pulse_rec_v13 (pulse.py; they need "pulse" on): mild shortages -----
    # Both are about the one stand-down the pulse makes by choice: pulse_partial leaves a grid alone while its fabs
    # get half their draw or more. A census of the 64 Full episodes of roots 12345, 2026 and 777 (pulse.Pulse.census,
    # read by experiments/pulse_agent/analysis/v13/stand_down/) counts 251 such grid-weeks of the 40,064 before
    # the end of sales, worth 0.0010 of the
    # score if every one were whole; every other stand-down is short of a fuel that is not there. Neither rule has a
    # setting of its own. Measured against agents/mpc_pulse_rec_v12, which this folder is with both switches false
    # (to the cent on those 64 episodes), in strict plays; 90 % paired intervals, episodes better / worse
    # True: a grid that is not pulsed and whose fabs get pulse_partial of their draw or more for want of a fuel
    # without a rationing threshold (a mild shortage: the fabs miss the fuel's whole shortfall every week it lasts)
    # is pulsed all the same once the chips its fabs have missed since the grid's last whole week are worth more
    # than the one dark week a start costs, and while the weeks a lot can still be sold in could miss as much again
    # (never worse than twice the best choice, whatever the shortage's length). False: pulse_partial alone decides.
    # Full 12345 (16 episodes): no start, every episode equal to the cent; Full 2026 (16) +0.0008 (+0.0000 to
    # +0.0017, 1 / 0: 20.5 bn USD in one episode); Full 777 (32) +0.0000 (+0.0000 to +0.0001, 3 / 0: 3.4 bn in
    # all); the 64 as one 0.8830 to 0.8831, +0.0002 (+0.0000 to +0.0004), 4 / 0 and 60 equal to the cent. Small
    # 12345 and 2026 (128 episodes): every episode equal to the cent. All four starts are of crude at one grid; in
    # the three that gained little the fab's chips were not short, and there the dark weeks' base load is all it
    # buys (in an episode whose chips are worth nothing a start would lose about 6 bn a cycle)
    "mild_start": True,
    # True: at such a grid a week short of a fuel without a rationing threshold is held, not delivered, when the
    # terminal's arrivals of the week say that the next week would leave the fabs under pulse_partial anyway (the
    # pulse then starts with a dark week whatever is done now) and that, held, the next week is whole: the dark week
    # comes first and a whole week replaces the partly powered one. False: the planner's flow stands in the partly
    # powered week. No forecast. Off as measured: Full 12345 +0.0004 (+0.0001 to +0.0009, 6 / 0), Full 2026 +0.0002
    # (-0.0000 to +0.0004, 5 / 2), Full 777 -0.0000 (3 / 4; from the play of both rules less mild_start's own
    # effect), the 64 as one +0.0001 (+0.0000 to +0.0003, 14 / 6, 25 bn saved, the largest loss 3.6); on Small (128
    # episodes) one episode gains 2.5 bn and one loses 2.3. It holds in 38 grid-weeks of the 64 episodes. The two
    # losses traced: an energy shock that began in the week after the held one (twice in one episode), and a week
    # held with the fabs at 94 % of their draw before weeks the rationed fuel left dark: the swap risks the partly
    # powered week's whole output for what that week lacks, and where the fab is short of wafers it only delays lots.
    # Both rules on: Full, the 64 as one, +0.0003 (+0.0001 to +0.0005, 18 / 5)
    "mild_ends": False,
    # ----- added in agents/mpc_pulse_rec_v13 (pulse.py): three rules about a run of whole weeks -----
    # Each mends a known misfire of a rule above, sized on recorded plays of agents/mpc_pulse_rec_v12 before it was
    # built (pulse.py, "Three rules about a run of whole weeks"); none has a setting of its own. All three
    # are OFF: none gains on the 64 Full episodes of roots 12345, 2026 and 777 taken as one, and one gains on Small.
    # Strict plays against that agent, which this folder plays as, to the cent, with the three off (Full 12345, 16
    # episodes; Small 12345, 64). 90 % paired intervals; episodes better / worse / equal to the cent; bn USD
    # True: schedule_ends ends a run at a grid that is not pulsed only if the end buys back more than it costs. It
    # buys the energy the fabs would lack each week once the terminal has run down (what the segment has to burn,
    # with the lots the plan itself starts, less what reaches the terminal), for as long as a lot can still be sold;
    # it costs the lots of the dark week that the weeks after cannot make up and the fuel the dark week saves beyond
    # what the weeks left of the episode could lack. False: the run is ended whenever the full segment cannot follow
    # and the fuel comes more slowly than the full-rate burn. OFF, as measured. The misfire is rare: of the runs
    # schedule_ends ended in recorded plays (67 in the 16 Full episodes of root 12345, 69 in those of root 2026, 234
    # in 128 Small episodes of root 6006) about 2, 2 and 9 are ends the rule does not make; all the others are grids
    # whose gas comes at 0.8 to 0.9 of what they burn, where the dark week is the pulse's own cycle. Full 12345
    # +0.0001 (+0.0000 to +0.0001, 1 / 0 / 15: the episode whose terminal held a week's burn less a thousandth,
    # 4.6), Full 2026 +0.0005 (-0.0000 to +0.0013, 1 / 1 / 14), Full 777 (32 episodes) +0.0000 (-0.0002 to +0.0002,
    # 2 / 3 / 27), the three as one +0.0001 (-0.0001 to +0.0003, 4 / 4 / 56; 20 bn saved, the largest gain 14.1, the
    # largest loss 8.5); Small 12345 -0.0001 (-0.0002 to +0.0000, 0 / 2 / 62), Small 2026 +0.0001 (+0.0000 to
    # +0.0003, 1 / 0 / 63), the two as one +0.0000 (-0.0001 to +0.0001); Small 6006 (128 episodes, the set the
    # misfire was read from, played in full) +0.0002 (-0.0000 to +0.0005, 5 / 1 / 122; 27 saved, the largest loss
    # 1.0). It does what it was built for and moves no score. (Outside Full 12345 and Small 6006 it was played
    # with runs_reach on every episode and alone on the episodes that play changed; an episode the two leave equal
    # to the cent, neither changes.) A first form took the fabs' draw from next week's wafers alone and counted the
    # whole dark week as lost: Full 12345 +0.0003 (+0.0000 to +0.0005, 4 / 1 / 11), Small 6006 +0.0001 (-0.0004 to
    # +0.0006, 18 / 9 / 101), with losses of 25.2, 15.7 and 12.1 traced to ends it should have made (pulse.py)
    "ends_needed": False,
    # True: schedule_runs waits only for the longest run whose fuel the arrivals bring within pulse_hold_weeks (or
    # the sellable weeks left); False: for the longest the terminal's storage carries, whatever arrives. OFF, as
    # measured: the long wait is not a misfire. It acts only where the gas is cut to a trickle (251 of the 549
    # weeks a charge waited on Small 6006, in 13 of 128 episodes; 18 of 263 on Full 2026, in one episode; none on
    # Full 12345, where all 16 episodes stay equal to the cent), and there the wait changes when the whole weeks
    # are run, not how many: the charge is made in the end, for the weeks that can still be sold, and a charge
    # made early spends on two whole weeks a tank that a later arrival would have stretched to three. Where it
    # acts (the difference between the plays with and without it next to ends_needed): 27 episodes of Small 6006
    # chosen for either rule, -9 in all (-27.8 and -7.4 against +9.2, +6.5, +4.5, +3.5 and +2.7), Small 2026 -17
    # (one episode -22.7), Small 12345 +16 (three episodes), Full 2026 -7 (one), Full 777 +9 (one)
    "runs_reach": False,
    # True: the top-up aims at the lots the window plan itself starts at the fab (the mean of its third to sixth
    # window weeks) that the fab has not started since its grid was pulsed, and next week's own; a week of capacity
    # at the most. False: at pulse_wafers weeks of capacity, whatever the plan wanted. OFF, as measured on Full; it
    # gains on Small, and whether that is worth a rule that does nothing on Full is the reader's call (open_hold is
    # on for the same reason). What it can take away on Full is small: on recorded plays the fabs of pulsed grids
    # start half the lots the plan wants, and what they start beyond it is 2 % of their lots (20 bn of base load at
    # VOLL in the 16 episodes of Full 12345, 39 in those of Full 2026). Full 12345 -0.0001 (-0.0006 to +0.0004,
    # 9 / 5 / 2; one episode loses 14.8: in agents/mpc_pulse_rec_v12's play of it the top-up brings one fab a week
    # of capacity four times while the plan starts no lot there for five weeks, and that fab's chips are short;
    # the loss itself was not traced), Full 2026 +0.0004 (+0.0000 to +0.0007, 8 / 5 / 3), Full 777
    # -0.0001 (-0.0003 to +0.0001, 16 / 9 / 7), the three as one +0.0000 (-0.0002 to +0.0002, 33 / 19 / 12; 18.5
    # saved, the largest loss 14.8); Small 12345 +0.0009 (+0.0003 to +0.0016, 27 / 12 / 25), Small 2026 +0.0003
    # (-0.0003 to +0.0010, 25 / 18 / 21), the two as one +0.0006 (+0.0002 to +0.0011, 52 / 30 / 46; 79 saved, the
    # largest loss 10.7). A first form forgot what the fab was owed in a week it ran: Full 12345 +0.0000 (-0.0007
    # to +0.0008, 7 / 8 / 1)
    "wafers_plan": False,
    # ----- added in agents/mpc_pulse_rec_v13 (this file): the week's CPU -----
    # The server meters the CPU of each week (2 s on the public board, 4 s on the private one). A week over it is
    # played by the naive rule, and an ``act`` still running is charged to the week after. What one such week costs
    # (the 16 Full episodes of root 12345, the week handed to the planner's own naive rule; 90 % paired intervals):
    # week 1 -0.0073 (-0.0089 to -0.0057), weeks 1 and 2 -0.0132, week 30 -0.0110, every episode worse each time.
    # How near the budget a week comes: the slowest is the first, a solve without a basis to start from (on Full
    # 10 to 15 thousand iterations where a later week takes 2 to 4 thousand). The scoring host is no slower than
    # the laptop it was timed on: the package's mpc_scen fits 1 s a week on small at 3 scenarios and not at 4 on
    # that host (shockbench_flow/policies/mpc_scen.py, S_BUDGET), and took 1.2 to 1.45 s and 1.3 to 1.8 s (the p95
    # of an episode's weeks) on the laptop, under the load in which this agent's first week took 1.3 to 3.3 s over
    # the 20 Full dev episodes. That puts the first week there at about 1.2 s for the median episode and 2.4 s at
    # most, of 4 s.
    # guard: insurance for the week that runs out all the same. The week's solve is given what is left of
    # ``guard_share`` of the budget (HiGHS's time_limit). Cut, the week plays, through the pulse: last week's plan
    # if there is one (``Planner.stale_action``), else the first week of a window of ``guard_window`` weeks solved
    # in what is left up to ``guard_reply`` of the budget (``Planner.short_action``: a fifth of the full window's
    # time), else an older plan, else the naive rule. A solve cut with presolve on leaves nothing to go on from,
    # so the weeks after it solve without presolve from the basis the last cut run reached, until one ends
    # (``GuardedSession``). What a cut costs, with the solve cut on purpose (same set): week 1 -0.0007, -0.0014,
    # -0.0010 and -0.0006 with a stand-in window of 5, 8, 12 and 16 weeks (no order in it, so the cheapest is
    # taken), weeks 1 and 2 -0.0012, week 30 -0.0009: a tenth to a fifth of what the naive week costs. The limit
    # is not reached in a local play on a quiet machine, and with it not reached the folder plays as
    # agents/mpc_pulse_rec_v12 does, to the cent (Full and Small, root 12345). A strict play raises when it is
    # reached, which a machine loaded with other plays can do in a hard first week: measure with guard false
    "guard": True,
    "guard_budget": None,  # the CPU seconds a week may take; None: the boards' 2 s, and 4 s for an episode over a year
    "guard_share": 0.8,  # the share of the budget at which the week's solve stops
    "guard_reply": 0.92,  # the share by which the stand-in for a cut solve must be solved
    "guard_window": 5,  # the weeks of the window that stands in for a cut solve when no plan of last week exists
    "guard_test": [],  # testing: the weeks whose solve is cut at once (a strict play does not raise for these)
    # preload: at import, build the instance and its naive plan for every instance JSON in instances/ (the public
    # small and full, as shockbench-flow packages them) into the planner's cache, which is keyed by the hash of the
    # instance: the first week then unpickles them instead of loading the instance and solving the plan (0.10 to
    # 0.15 s of the laptop's CPU on Full). An episode on another instance misses the cache and builds its own
    "preload": True,
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
if set(PARAMS["hold_checks"]) - set(pulse.CHECKS):  # a misspelt check would otherwise change nothing
    raise ValueError(f"params.json: hold_checks {PARAMS['hold_checks']!r} names no check of {list(pulse.CHECKS)}")
if PARAMS["h_weeks"] is not None and int(PARAMS["h_weeks"]) < 1:
    raise ValueError(f"params.json: h_weeks {PARAMS['h_weeks']!r} is not a number of weeks")
if not 0 < PARAMS["guard_share"] <= PARAMS["guard_reply"] <= 1 or int(PARAMS["guard_window"]) < 1:
    raise ValueError("params.json: guard_share and guard_reply are shares of the budget, guard_window a number of weeks")


def preload(folder=Path(__file__).parent / "instances"):
    """Fill the planner's cache of (instance, naive plan) from the instance JSONs beside this file; the number filled.

    ``sbflow.policies.naive._cached_plan`` keys its cache by the hash of the instance JSON and hands every reset
    its own unpickled copies, so an episode on one of these instances gets exactly what it would have built. Run at
    import, which the server does not meter. A file that fails is skipped (the first week then builds its own).
    """
    done = 0
    for path in sorted(folder.glob("*.json")):
        try:
            cached_plan({"instance": json.loads(path.read_text(encoding="utf-8"))}, PolicyContext().fq_quantile)
            done += 1
        except Exception:  # noqa: BLE001 - a preload, never a failure of the agent
            if PARAMS["strict"]:
                raise
            traceback.print_exc()
    return done


if PARAMS["preload"]:
    preload()


CUT = "Time limit reached"  # HiGHS's model status of a run stopped by its time_limit


class GuardedSession(lp_common.LPSession):
    """An ``LPSession`` whose solve stops at a deadline and is taken up again the week after (``guard``).

    ``deadline`` is a reading of ``time.process_time`` (the CPU of this process, which is what the server meters),
    or None for no limit: the session is then an ``LPSession``. With one, each rung of the ladder is given what is
    left of it as HiGHS's ``time_limit`` and a rung that would start after it is not run; ``timed_out`` says whether
    the last solve ended that way.

    A run cut with presolve on returns no basis, so the ladder's work is lost when it is cut. From then on
    (``catching_up``) a solve is one dual simplex run without presolve, started from the basis the session holds,
    moved to the week's window: the basis of the last solve that ended, or the one the last cut run reached, which
    HiGHS does return without presolve. Each week's run so goes on from the one before, and the first that ends
    puts the session back on the ladder. ``behind`` counts the weeks since the held basis was a week's own.
    """

    def __init__(self):
        super().__init__()
        self.deadline, self.timed_out, self._limit = None, False, None
        self.catching_up, self.behind = False, 0

    def _attempt(self, rung, lp, rung_options, basis, **more):
        if self._limit is None:
            return super()._attempt(rung, lp, rung_options, basis, **more)
        left = self._limit - time.process_time()
        if left <= 0:
            self.timed_out = True
            return lp_common._Attempt(f"{CUT} (not run)", 0, None, None, None)
        attempt = super()._attempt(rung, lp, {**rung_options, "time_limit": left}, basis, **more)
        if attempt.x is None and attempt.status.startswith(CUT):
            self.timed_out = True
        return attempt

    def solve(self, lp, shape=None, *, warm=True):
        self.timed_out = False
        if self.deadline is None:
            return super().solve(lp, shape, warm=warm)
        if self.catching_up:
            return self._catch_up(lp, shape)
        held = (self._col, self._row, self._shape)
        self._limit = self.deadline
        try:
            res = super().solve(lp, shape, warm=warm)
        finally:
            self._limit = None
        if not res.ok and self.timed_out:  # the ladder dropped the basis: keep it, a week older, to start from
            self._col, self._row, self._shape = held
            self.catching_up, self.behind = True, self.behind + 1 if held[0] is not None else 0
        elif res.ok:
            self.behind = 0
        return res

    def _moved(self, lp, shape):
        """The held basis moved to this week's window (``behind`` weeks further than a usual week); None: none fits."""
        if self._col is None or self._shape is None or shape is None:
            return None
        try:
            col, row = lp_common._shift_arrays(self._col, self._row, self._shape, shape, 1 + self.behind)
        except ValueError:
            return None
        if len(col) != lp.num_col_ or len(row) != lp.num_row_:
            return None
        return lp_common._to_basis(col, row, alien=True)

    def _catch_up(self, lp, shape):
        """One dual simplex run without presolve from the held basis, cut at the deadline (class docstring)."""
        start = time.perf_counter()

        def result(status, iterations=0, x=None, objective=None):
            trail = (("catch_up", status),)
            return lp_common.SolveResult(x, x is not None, status, trail, iterations, time.perf_counter() - start, objective)

        left = self.deadline - time.process_time()
        if left <= 0:
            self.timed_out = True
            self.behind += 1
            return result(f"{CUT} (not run)")
        hs = lp_common._highspy()
        h = hs.Highs()
        lp_common._set_options(h, {**self.options, "presolve": "off", "time_limit": left})
        h.passModel(lp)
        basis = self._moved(lp, shape)
        if basis is not None:
            h.setBasis(basis)  # a basis HiGHS refuses leaves the run cold
        h.run()
        status = h.modelStatusToString(h.getModelStatus())
        info = h.getInfo()
        iterations = lp_common._iterations(info)
        reached = h.getBasis()
        reached = lp_common._from_basis(reached) if reached.valid else None
        if status == lp_common.OPTIMAL:
            x = np.array(h.getSolution().col_value, dtype=np.float64)
            if x.shape == (lp.num_col_,) and bool(np.all(np.isfinite(x))):
                self._keep(reached, shape)
                self.catching_up, self.behind = False, 0
                return result(status, iterations, x, float(info.objective_function_value))
            status = f"{status} (no finite point)"
        if status.startswith(CUT):
            self.timed_out = True
            if reached is not None:  # what the cut run reached is this week's: next week goes on from it
                self._col, self._row = reached
                self._shape, self.behind = shape, 0
            else:
                self.behind += 1
            return result(status, iterations)
        self._keep(None, None)  # failed otherwise: back to the ladder, without a basis
        self.catching_up, self.behind = False, 0
        return result(status, iterations)


class Planner(MpcDet):
    def reset(self, static, obs, policy_seed):
        super().reset(static, obs, policy_seed)
        self.recovery = Recovery(self._inst, PARAMS) if PARAMS["kinds"] else None
        # weeks planned on what is observed, after a failure of the forecast; weeks planned with the benchmark's own
        # LP, after a failure of the reserve's code
        self.forecast_failed, self.reserve_failed = 0, 0
        self.window = None  # the week's window model and its solution, kept for the pulse's open_hold; None: no plan
        # the guard: the session that stops at ``deadline``, and ``reply``, by which a stand-in must be solved (both
        # set by ``Agent.act`` each week; None: no limit), the last plan solved as (model, solution, its week), and
        # the weeks played from a stand-in after a cut solve
        self._session = GuardedSession()
        self.deadline, self.reply, self.last, self.guard_weeks = None, None, None, 0

    def _record(self, model, res):
        """Keep the week's window plan for ``pulse.Pulse.adjust`` (``open_hold`` reads the fuels it leaves short)."""
        self.window = (model, res.x) if (PARAMS["open_hold"] or PARAMS["wafers_plan"]) and res.ok else None

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
        forced = t in PARAMS["guard_test"]  # a test: this week's solve is cut at once
        self._session.deadline = time.process_time() if forced else self.deadline
        model = res = None
        try:
            model = reserve.rolled_lp(inst, obs, arrays, H_t, PARAMS, planning_rules=rules)
            res = self._session.solve(lp_common.to_highs_lp(model), lp_common.WindowShape.of(model))
            if not res.ok and not self._session.timed_out and reserve.changes(model):
                raise RuntimeError(f"no solution of the window with the reserve: {res.trail}")
        except Exception:  # noqa: BLE001 - any failure of the reserve's code: this week is planned without it
            if PARAMS["strict"]:
                raise
            traceback.print_exc()
            self.reserve_failed += 1
            model = lp_common.rolled_lp(inst, obs, arrays, H_t, planning_rules=rules)
            res = self._session.solve(lp_common.to_highs_lp(model), lp_common.WindowShape.of(model))
        self._record(model, res)
        action = None
        if res.ok:
            action = lp_common.week1_action(inst, model, res.x, obs, lp_common.prohibited_now(self._memory, t))
            self.last = (model, res.x, t)
        elif self._session.timed_out:
            if PARAMS["strict"] and not forced:
                raise RuntimeError(f"guard: the solve of week {t} was cut at the week's CPU limit: {res.trail}")
            action = self.stand_in(obs, t, arrays, H_t, None if forced else self.reply)
            self.guard_weeks += action is not None
        if action is None:
            action = self._fallback.act(obs)
        self.telemetry.append(
            StepTelemetry(t, time.perf_counter() - start, res.iterations, res.trail, fallback=not res.ok)
        )
        return action

    def stand_in(self, obs, t, arrays, H_t, reply):
        """The action of a week whose solve was cut; None: there is none, and the week plays the naive rule.

        Last week's plan first (it costs nothing and loses nothing measurable), then a short window solved now,
        then a plan older than a week.
        """
        action = self.stale_action(obs, t, most=1)
        if action is None:
            action = self.short_action(obs, t, arrays, H_t, reply)
        if action is None:
            action = self.stale_action(obs, t)
        return action

    def stale_action(self, obs, t, most=None):
        """Week t's action from the last plan solved, if that is at most ``most`` weeks old; None: it has none.

        The plan's columns are week by week (``WindowShape``), so the plan of week ``w`` holds week t's flows
        ``t - w`` blocks in; they are read as a first week's are, against this week's prohibitions. The simulator
        clips what the stock or the capacity of this week does not allow, as it clips every action.
        """
        try:
            if self.last is None:
                return None
            model, x, week = self.last
            nc, behind = int(model.meta["nc"]), t - week
            if not 0 < behind < int(model.T) or len(x) != nc * int(model.T) or (most is not None and behind > most):
                return None
            moved = np.zeros_like(x)
            moved[: len(x) - behind * nc] = x[behind * nc :]
            return lp_common.week1_action(self._inst, model, moved, obs, lp_common.prohibited_now(self._memory, t))
        except Exception:  # noqa: BLE001 - any failure here: the week falls through to the next stand-in
            if PARAMS["strict"]:
                raise
            traceback.print_exc()
            return None

    def short_action(self, obs, t, arrays, H_t, reply):
        """Week t's action from a window of ``guard_window`` weeks, solved by ``reply`` (None: no limit); None: none.

        The same model as the week's own on the first weeks of the same forecast, with the end value at its end; a
        fifth to a third of the full window's time. Its session is its own: the week's basis is not touched.
        """
        try:
            H = min(int(PARAMS["guard_window"]), H_t)
            if any(a.shape[0] != H_t for a in arrays.values()):
                return None
            window = lp_common.read_only({name: a[:H].copy() for name, a in arrays.items()})
            rules = self.params.planning_rules
            model = reserve.rolled_lp(self._inst, obs, window, H, PARAMS, planning_rules=rules)
            session = GuardedSession()
            session.deadline = reply
            res = session.solve(lp_common.to_highs_lp(model), lp_common.WindowShape.of(model), warm=False)
            if not res.ok:
                return None
            return lp_common.week1_action(self._inst, model, res.x, obs, lp_common.prohibited_now(self._memory, t))
        except Exception:  # noqa: BLE001 - any failure here: the week falls through to the next stand-in
            if PARAMS["strict"]:
                raise
            traceback.print_exc()
            return None


def nullable(obs, key, cast=float):
    """A column as a list, None where it is not observed."""
    return [cast(v) if seen else None for v, seen in zip(obs[key].tolist(), obs[f"{key}.observed"].tolist())]


def rows(obs, block, fields):
    """The live entries of a padded list, as columns."""
    live = np.flatnonzero(obs[f"{block}.{fields[0]}.observed"])
    return {f: obs[f"{block}.{f}"][live].tolist() for f in fields}, live


class Agent:
    def __init__(self, config=None):
        # the guard: this process's CPU when the week began (here, then the end of the act before: what the kit
        # does between two acts is metered with the week), and the week's CPU budget
        self.cpu_mark = time.process_time()
        self.budget = float(PARAMS["guard_budget"] or (4.0 if int(config["T"]) > 52 else 2.0))
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
        try:
            return self.decide(observation)
        finally:
            self.cpu_mark = time.process_time()

    def decide(self, observation):
        if not self.broken:
            try:
                obs = self.wire(observation)
                if self.planner is None:
                    planner = Planner(MpcDetParams(), PolicyContext())
                    planner.reset(self.static, obs, self.seed)
                    self.planner = planner
                    self.pulse = self.new_pulse(planner._inst)
                if PARAMS["guard"]:
                    self.planner.deadline = self.cpu_mark + PARAMS["guard_share"] * self.budget
                    self.planner.reply = self.cpu_mark + PARAMS["guard_reply"] * self.budget
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
