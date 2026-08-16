"""Download ALL 200 episodes per task (idempotent - skips anything already cached).
Also saves an 'episode_plan_first100.json' listing the first 100 by sorted episode ID
(not evenly spaced) for whichever training run wants exactly that subset.

Uses HfFileSystem.ls() scoped to each task's annotation directory instead of
list_repo_files() over the whole repo - the latter enumerates the entire dataset
(all tasks x all camera views) and took 25+ minutes with no output; a per-directory
listing takes well under a second.
"""
from huggingface_hub import HfFileSystem, hf_hub_download

REPO = "behavior-1k/2025-challenge-demos"
TASKS = ["task-0000", "task-0001", "task-0005", "task-0020", "task-0022",
         "task-0031", "task-0034", "task-0040", "task-0042", "task-0049"]


def main():
    fs = HfFileSystem()

    plan_all = {}
    plan_first100 = {}
    for tc in TASKS:
        files = fs.ls(f"datasets/{REPO}/annotations/{tc}", detail=False)
        eps = sorted(f.split("/")[-1].replace(".json", "") for f in files if f.endswith(".json"))
        plan_all[tc] = eps
        plan_first100[tc] = eps[:100]
        print(f"{tc}: {len(eps)} total episodes available")

    total = 0
    for tc, eps in plan_all.items():
        for i, ep in enumerate(eps):
            try:
                hf_hub_download(REPO, f"annotations/{tc}/{ep}.json", repo_type="dataset")
                hf_hub_download(REPO, f"videos/{tc}/observation.images.rgb.head/{ep}.mp4", repo_type="dataset")
                total += 1
            except Exception as e:
                print(f"  FAILED {tc}/{ep}: {e}")
            if (i + 1) % 25 == 0:
                print(f"  {tc}: {i+1}/{len(eps)} done")
        print(f"{tc}: ALL {len(eps)} episodes done")

    import json
    with open("/shared_work/markhsp/Behavior1k-MAPLE/behavior1k_statuschecker/results/episode_plan_all200.json", "w") as f:
        json.dump(plan_all, f, indent=2)
    with open("/shared_work/markhsp/Behavior1k-MAPLE/behavior1k_statuschecker/results/episode_plan_first100.json", "w") as f:
        json.dump(plan_first100, f, indent=2)
    print(f"Total episode-downloads attempted: {total}")
    print("Saved episode_plan_all200.json and episode_plan_first100.json")


if __name__ == "__main__":
    main()
