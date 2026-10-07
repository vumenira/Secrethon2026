"""Grids of a task: priority rule, base load, deliverable power, the draw of its fabs."""

import fire


def main(task="full"):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator(task)
    for go, g in enumerate(inst.grids):
        ga = inst.nodes[g].grid
        fabs = [inst.nodes[inst.fabs[fo]] for fo in inst.grid_fabs[go]]
        draw = sum(f.fab.e * f.fab.cap0 for f in fabs)
        print(
            f"{inst.nodes[g].id:10s} {ga.priority:16s} base {ga.base_load:9.0f} deliverable {ga.deliverable:9.0f}"
            f" fabs' full draw {draw:7.0f} ({draw / ga.deliverable:.1%} of deliverable; spare over base + fabs"
            f" {(ga.deliverable - ga.base_load - draw) / ga.deliverable:+.1%}) fabs {[f.id for f in fabs]}"
        )


if __name__ == "__main__":
    fire.Fire(main)
