# Secrethon 2026: our ShockBench-Flow solution

This repository is our solution to the **ShockBench-Flow** competition. Every week an agent decides how much of each
good (fuel, wafers, chips) to send along each route of a supply network, while disruptions hit it: straits close,
routes get sanctioned, tariffs jump. An episode costs money; lower is better. The score is **0** for the naive rule
(keep shipping the normal plan) and **1** for the clairvoyant plan (knew every disruption in advance).

The organisers' original README (installation, the task, the rules, all `sbf` commands) is kept as
**[READMEOLD.md](READMEOLD.md)**. Read it first if you are new; the details are in [docs/GUIDE.md](docs/GUIDE.md).

## Setup

```bash
uv sync          # installs Python and the exact packages (see READMEOLD.md, "Installation")
```

Our agents live in `agents/<name>/`, one folder each (`agent.py` and its files).

## Our agent

**The agent to submit is `agents/mpc_pulse_rec_v8_on` (0.8768 on the public board).** What it does, why, how it was
built and what did not work: **[SOLUTION.md](SOLUTION.md)**. The full log of every version and measurement is in
[experiments/pulse_agent/README.md](experiments/pulse_agent/README.md).

---

## The episode viewer (`sbf view`)

An interactive page that shows **what an agent actually did**, week by week: the network map, what it shipped on each
route, cargo on its way, closed straits, costs against the naive rule, and every disruption. Use it to find where an
agent loses money.

It opens in any browser (one HTML file, works offline) and is light enough for a weak laptop.

### 1. Make the page

```bash
uv run sbf view heuristic                    # an agent from agents/, on Tiny (the practice network)
uv run sbf view heuristic --task=small       # on Small, the public leaderboard's network
uv run sbf view heuristic --task=small --episode=7    # just one episode, no evaluation (faster)
uv run sbf view heuristic --quick            # a rough run in seconds, for a quick look
```

Replace `heuristic` with any agent name (a folder of `agents/`), or a path to a folder or zip.

What the command does:

1. **Evaluates** the agent on the 20 dev episodes (the same as `sbf evaluate`).
2. **Picks 3 episodes**: the agent's worst against naive, its best, and one where a strait closes.
3. **Replays and records** those episodes week by week, with the naive rule on the same scenarios for comparison.
4. **Writes the page** and prints its path.

> **The first run on a network is slow.** It computes the reference costs once: about a minute on Tiny, around
> 25 minutes on Small on a laptop (it uses all CPU cores and most of the RAM). They are cached in
> `~/.cache/shockbench-flow/`, so later runs, for any agent, are much faster.

### 2. Open it

```bash
xdg-open outputs/view/heuristic_small/viewer.html     # or double-click the file
```

### Where things are saved

Everything goes in `outputs/view/<agent>_<task>/` (not tracked by git):

| File               | What it is                                                     |
| ------------------ | -------------------------------------------------------------- |
| `viewer.html`      | the page: open this                                            |
| `eval.json`        | the evaluation: score and the cost of every episode            |
| `record_epN.npz`   | the full recording of episode N                                |
| `cache.json`       | a fingerprint of the agent's files                             |

Running the command again on an **unchanged** agent reuses these and only rewrites the page (a few seconds). If you
**edit** the agent, it is evaluated and recorded again automatically. `--fresh` forces a re-run.

### 3. Use the page

**Top bar:** the agent, the network, the **episode picker** (the 3 recorded episodes), and the key numbers: episode
cost, naive cost, saving vs naive, the episode's skill (0 = naive, 1 = clairvoyant) and the overall score.

**The map** reads left to right in the order goods travel: sources → ports → power grids and material suppliers →
chip fabs → packaging plants → markets. Straits sit in the gaps between those columns, next to the routes that pass
through them.

- **Nodes** are emojis on a glowing disc in their kind's colour:

  | 🛢️ source | ⚓ port terminal | ⚡ power grid | 🌊 strait | 🧪 material supplier | 🏭 chip fab | 📦 packaging plant | 🏙️ market |
  | --------- | ---------------- | ------------- | --------- | -------------------- | ----------- | ------------------ | --------- |

  A strait's disc turns green / yellow / orange / red as it closes; `queue 1.1M` above it is cargo stuck waiting.
  Markets show `% served` (demand met); grids show `shed` when they cut power.
- **Routes**: the line style is the transport (🚢 sea solid, ✈️ air dashed, 🔧 pipeline dash-dot, 🔌 grid link dotted);
  the colour is the state this week (grey open, orange reduced capacity, red closed or sanctioned).
- **Thick coloured lines** are what the agent shipped this week: the colour is the good, the width the share of the
  route's capacity. Goods: 🔥 LNG · 🛢️ crude · ☢️ nuclear fuel · 💿 wafers · 🔲 raw chips · 💾 chips.
- **🚢 / ✈️ on a route** is the cargo on its way there (one per route): bigger = more units, further along = arriving
  sooner. Hover it to see the shipments it carries and when each arrives.
- **Faint routes** are idle this week (nothing sent, no cargo), so the busy ones stand out. *Fade idle routes* on
  the left turns this off.
- **Filter by good**: click a good in the Legend (e.g. 🔥 LNG) to show only its routes, shipments, cargo and actions;
  click it again or *show all goods* to go back.
- **Hover** a route or node for details (capacity, what was sent, the normal plan, whether the order was clipped,
  stock, demand); the rest of the map dims so you see only its connections. **Click** a node to open it in the *Node* tab.
- **Drag a node** to move it (its routes follow); your arrangement is remembered in this browser, and *↺ Reset
  layout* puts everything back. Routes joining the same two nodes run side by side.
- Mouse **wheel** zooms, dragging the **background** pans, **double-click** or *Fit view* resets. The buttons on the left switch
  shipments, cargo, idle fading, capacity numbers and names on or off.

**The agent's actions (📋 Actions, or key `A`)** opens a drawer over the bottom of the map with every decision the
agent made this week, one row per route and good, the biggest departures from the normal plan first:

| Column          | Meaning                                                                                   |
| --------------- | ----------------------------------------------------------------------------------------- |
| route           | 🚢/✈️ from → to; routes through straits say `via 🌊 hormuz → malacca`                       |
| good            | what is shipped                                                                           |
| asked           | what the agent ordered                                                                    |
| sent            | what actually left (less than asked when stock or capacity ran out)                       |
| normal plan     | what the normal plan ships there every week (what the naive rule sticks to)               |
| vs plan         | asked minus plan: blue = more than the plan, pink = less, `=` the same                     |
| ⚠ clipped       | the order was cut down: the agent asked for more than could go                             |

*only changed from plan* and *only clipped* filter the rows. **Hover a row** to light its route up on the map. If the
agent takes over a strait's release (holds queued tankers or sets its own quantities), a *Strait releases* line shows
it above the table. The drawer does no work while it is closed.

**The week bar (bottom):** ⏮ ◀ ▶ ⏭ to move through weeks, **Play** to animate (speed on the right), the slider to jump.
Coloured marks above the slider are weeks with disruptions; **Next event** jumps to the next one.
Keys: `←` `→` weeks, `Space` play/pause, `Home` / `End`, `N` next event, `A` actions.

**The side tabs** (click any chart to jump to that week):

| Tab           | Shows                                                                                                 |
| ------------- | ----------------------------------------------------------------------------------------------------- |
| **Cost**      | this week's and cumulative cost against naive, and which weeks the agent was cheaper (green) or dearer (red) |
| **Breakdown** | cost by component (freight, tariff, holding, shortage, …) and a table of **where the agent spends more than naive**: the best place to look for improvements |
| **Node**      | the clicked node: stock over time, demand vs served (markets), open fraction and queue (straits), its routes this week |
| **Events**    | the disruptions of this week, all events of the episode (click one to jump), early-warning scores     |
| **Episodes**  | the evaluation: score, interval and every episode's result; click a recorded one (●) to open it       |
| **Scores**    | the evaluation by harm level: every episode's skill as a dot, each level's score and weight, the points of the score each level earned and missed, and the episodes that miss the most |

### A typical workflow

1. `uv run sbf view mine --task=small` and open the page.
2. Open the **worst** episode, **Breakdown** tab: which cost is much higher than naive's?
3. Press **Next event** / **Play** to find the weeks it goes wrong; hover the routes and straits involved.
   Open **📋 Actions** (`A`) with *only changed from plan* to see what the agent decided differently that week, and
   *only clipped* for orders that could not go.
4. Change `agents/mine/agent.py`, then confirm with `uv run sbf compare mine <old agent> --task=small` and view again.

### Where the code is

`src/sbf_starter/viewer/`: `__init__.py` (the command), `export.py` (turns a recording into the page's data),
`page.html` (the page). The test is `tests/test_viewer.py`.
