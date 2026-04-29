from __future__ import annotations

from pathlib import Path

from noshow_iq.model import train
from noshow_iq.preprocess import dataset_expected_path


def main() -> None:
    dataset_path = Path(dataset_expected_path())
    if not dataset_path.exists():
        dataset_path = Path("data/sample/KaggleV2-May-2016.sample.csv")

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = train(
        str(dataset_path),
        model_path=str(out_dir / "model.pkl"),
        log_training_to_mongo=True,
    )
    print(out)


if __name__ == "__main__":
    main()

