from __future__ import annotations

from pathlib import Path

from noshow_iq.model import train
from noshow_iq.preprocess import dataset_expected_path


def main() -> None:
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = train(
        dataset_expected_path(),
        model_path=str(out_dir / "model.pkl"),
        log_training_to_mongo=False,
    )
    print(out)


if __name__ == "__main__":
    main()

