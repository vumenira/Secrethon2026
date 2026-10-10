"""Put one of our finished submissions on the competition's leaderboard, or show which one is on it.

    uv run python scripts/leaderboard.py                  # the leaderboard's rule and our submissions on it
    uv run python scripts/leaderboard.py --submission=975379

The kit has no command for this. The competition's leaderboard holds one submission of the participant's choice at a
time (its rule is ``Add_And_Delete``), picked with the leaderboard button of "My Submissions" on the website; this
is the same call. It uploads nothing and spends no submission, and putting another submission on later takes this
one off again. Like an upload, it changes what others see: only when the participant asks. It reads the token that
``sbf token`` stored in ``.env``.
"""

import fire

from sbf_starter import codabench


def main(submission: int | None = None, phase: str | None = None) -> None:
    """Show the leaderboard's rule and what is on it; with ``submission``, put that finished submission on it."""
    client, pk, secret = codabench.from_env()
    comp = client.competition(pk, secret)
    for board in comp.get("leaderboards") or []:
        print(f"leaderboard {board.get('id')}: rule {board.get('submission_rule')}")
    if submission is None:
        ph = codabench.pick_phase(comp, phase)
        on = [s for s in client.submissions(ph["id"]) if s.get("on_leaderboard")]
        for s in on:
            print(f"on the leaderboard of phase {ph['name']!r}: {s['id']} {s.get('filename')}")
        if not on:
            print(f"nothing of ours is on the leaderboard of phase {ph['name']!r}")
        return
    sub = client.submission(int(submission))
    print(f"{sub['id']} {sub.get('filename')}: status {sub['status']}, on the leaderboard: {sub['on_leaderboard']}")
    if sub["status"] != "Finished":
        raise SystemExit("not a finished submission: nothing done")
    if not sub["on_leaderboard"]:
        client.request("POST", f"/api/submissions/{sub['id']}/submission_leaderboard_connection/", ok=(200, 201, 204))
        sub = client.submission(int(submission))
    print(f"{sub['id']} {sub.get('filename')}: on the leaderboard: {sub['on_leaderboard']}")


if __name__ == "__main__":
    fire.Fire(main)
