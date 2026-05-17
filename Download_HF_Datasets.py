import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

from config import DataConfig

DATASETS_DIR = Path("./datasets")


def download_dataset(
    repo_id: str,
    local_dir: Path,
    allow_patterns: list[str],
    token: str | None,
) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        token=token,
        allow_patterns=allow_patterns,
        max_workers=8,
    )
    return local_dir


def main() -> None:
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    data_cfg = DataConfig()

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    nsynth_dir = DATASETS_DIR / data_cfg.nsynth_subdir_name
    vctk_dir = DATASETS_DIR / data_cfg.vctk_subdir_name

    download_jobs = {
        "nsynth": (
            data_cfg.nsynth_hf_dataset_name,
            nsynth_dir,
            ["data/train/*.parquet"],
        ),
        "vctk": (
            data_cfg.vctk_hf_dataset_name,
            vctk_dir,
            ["data/train-*.parquet"],
        ),
    }

    print(f"Downloading {list(download_jobs)} in parallel into {DATASETS_DIR}/")
    with ThreadPoolExecutor(max_workers=len(download_jobs)) as pool:
        futures = {
            pool.submit(download_dataset, repo, path, patterns, hf_token): name
            for name, (repo, path, patterns) in download_jobs.items()
        }
        for fut in as_completed(futures):
            name = futures[fut]
            fut.result()
            print(f"  done: {name}")


if __name__ == "__main__":
    main()
