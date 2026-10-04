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

**The map:**

- **Nodes** have a shape and colour per kind: sources, terminals, power grids, straits, material suppliers, chip fabs,
  packaging plants, markets. The **Legend** card (top right, click to fold) explains every symbol.
- **Routes:** the line style is the transport (sea solid, air dashed, pipeline, grid dotted); the colour is the state
  this week (grey open, orange reduced capacity, red closed or sanctioned).
- **Thick coloured lines** are what the agent shipped this week: the colour is the good, the width is the share of the
  route's capacity.
- **Dots** on a route are cargo in transit: bigger = more units, closer to the end = arriving sooner.
- **Straits** turn green / yellow / orange / red as they close; `queue 1.1M` above one is cargo stuck waiting.
  Markets show `% served` (demand met); grids show `shed` when they cut power.
- **Hover** a route or node for details (capacity, what was sent, whether the order was clipped, stock, demand). The
  rest of the map dims so you see only its connections.
- **Click** a node to open it in the *Node* tab.
- Mouse **wheel** zooms, **drag** pans, **double-click** or *Fit view* resets. The buttons on the left switch
  shipments, transit dots, capacity numbers and names on or off.

**The week bar (bottom):** ⏮ ◀ ▶ ⏭ to move through weeks, **Play** to animate (speed on the right), the slider to jump.
Coloured marks above the slider are weeks with disruptions; **Next event** jumps to the next one.
Keys: `←` `→` weeks, `Space` play/pause, `Home` / `End`, `N` next event.

**The side tabs** (click any chart to jump to that week):

| Tab           | Shows                                                                                                 |
| ------------- | ----------------------------------------------------------------------------------------------------- |
| **Cost**      | this week's and cumulative cost against naive, and which weeks the agent was cheaper (green) or dearer (red) |
| **Breakdown** | cost by component (freight, tariff, holding, shortage, …) and a table of **where the agent spends more than naive**: the best place to look for improvements |
| **Node**      | the clicked node: stock over time, demand vs served (markets), open fraction and queue (straits), its routes this week |
| **Events**    | the disruptions of this week, all events of the episode (click one to jump), early-warning scores     |
| **Episodes**  | the evaluation: score, interval and every episode's result; click a recorded one (●) to open it       |

### A typical workflow

1. `uv run sbf view mine --task=small` and open the page.
2. Open the **worst** episode, **Breakdown** tab: which cost is much higher than naive's?
3. Press **Next event** / **Play** to find the weeks it goes wrong; hover the routes and straits involved.
4. Change `agents/mine/agent.py`, then confirm with `uv run sbf compare mine <old agent> --task=small` and view again.

### Where the code is

`src/sbf_starter/viewer/`: `__init__.py` (the command), `export.py` (turns a recording into the page's data),
`page.html` (the page). The test is `tests/test_viewer.py`.
