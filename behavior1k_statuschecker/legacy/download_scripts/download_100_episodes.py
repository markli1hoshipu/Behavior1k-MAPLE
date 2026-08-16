"""Download up to 100 episodes per task, evenly spread across the 200 available.
Idempotent - hf_hub_download skips anything already cached, so this safely covers
the 6 episodes we already have plus 94 new ones per task."""
from huggingface_hub import HfApi, hf_hub_download

REPO = "behavior-1k/2025-challenge-demos"
TASKS = ["task-0000", "task-0001", "task-0005", "task-0020", "task-0022",
         "task-0031", "task-0034", "task-0040", "task-0042", "task-0049"]

N_TARGET = 100


def main():
    api = HfApi()
    files = api.list_repo_files(REPO, repo_type="dataset")

    plan = {}
    for tc in TASKS:
        eps = sorted(set(f.split("/")[-1].replace(".json", "")
                          for f in files if f.startswith(f"annotations/{tc}/")))
        n = len(eps)
        k = min(N_TARGET, n)
        idxs = sorted(set(int(i * n / k) for i in range(k)))
        picked = [eps[i] for i in idxs]
        plan[tc] = picked
        print(f"{tc}: {len(picked)} episodes planned (of {n} available)")

    total_downloaded = 0
    for tc, eps in plan.items():
        for ep in eps:
            try:
                hf_hub_download(REPO, f"annotations/{tc}/{ep}.json", repo_type="dataset")
                hf_hub_download(REPO, f"videos/{tc}/observation.images.rgb.head/{ep}.mp4", repo_type="dataset")
                total_downloaded += 1
            except Exception as e:
                print(f"  FAILED {tc}/{ep}: {e}")
        print(f"{tc}: done ({len(eps)} episodes)")

    import json
    with open("/shared_work/markhsp/behavior1k-statuschecker/results/episode_plan_100.json", "w") as f:
        json.dump(plan, f, indent=2)
    print(f"Total episode-downloads attempted: {total_downloaded}")
    print("Saved episode_plan_100.json")


if __name__ == "__main__":
    main()
