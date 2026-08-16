"""Download 5 additional episodes per task (spread evenly across the 200 available),
on top of the 1 we already have, for a larger/more diverse LoRA training set."""
from huggingface_hub import HfApi, hf_hub_download

REPO = "behavior-1k/2025-challenge-demos"
TASKS = ["task-0000", "task-0001", "task-0005", "task-0020", "task-0022",
         "task-0031", "task-0034", "task-0040", "task-0042", "task-0049"]

ALREADY_HAVE = {
    "task-0000": "episode_00000010", "task-0001": "episode_00010010",
    "task-0005": "episode_00050020", "task-0020": "episode_00200010",
    "task-0022": "episode_00220020", "task-0031": "episode_00310010",
    "task-0034": "episode_00340020", "task-0040": "episode_00400010",
    "task-0042": "episode_00420010", "task-0049": "episode_00490010",
}

N_NEW = 5


def main():
    api = HfApi()
    files = api.list_repo_files(REPO, repo_type="dataset")

    plan = {}
    for tc in TASKS:
        eps = sorted(set(f.split("/")[-1].replace(".json", "")
                          for f in files if f.startswith(f"annotations/{tc}/")))
        eps = [e for e in eps if e != ALREADY_HAVE[tc]]
        n = len(eps)
        idxs = sorted(set(int(i * n / N_NEW) for i in range(N_NEW)))
        picked = [eps[i] for i in idxs]
        plan[tc] = picked
        print(f"{tc}: picked {picked}")

    for tc, eps in plan.items():
        for ep in eps:
            a = hf_hub_download(REPO, f"annotations/{tc}/{ep}.json", repo_type="dataset")
            v = hf_hub_download(REPO, f"videos/{tc}/observation.images.rgb.head/{ep}.mp4", repo_type="dataset")
            print(f"  {tc}/{ep}: OK")

    import json
    with open("/shared_work/markhsp/Behavior1k-MAPLE/behavior1k_statuschecker/results/episode_plan.json", "w") as f:
        json.dump({tc: [ALREADY_HAVE[tc]] + eps for tc, eps in plan.items()}, f, indent=2)
    print("Saved episode plan.")


if __name__ == "__main__":
    main()
