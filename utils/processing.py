from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import shutil


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class DatasetSplitConfig:
    input_dir: Path
    output_dir: Path
    seed: int = 42
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    def validate(self) -> None:
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if round(total, 5) != 1.0:
            raise ValueError("Train, validation, and test ratios must sum to 1.0.")


def split_dataset(config: DatasetSplitConfig, classes: dict[str, str]) -> dict[str, dict[str, int]]:
    """
    Create train/validation/test folders from a class-mapped image dataset.

    This helper is intentionally side-effectful but no longer depends on hard-coded
    machine-specific paths or module-level execution.
    """

    config.validate()
    rng = random.Random(config.seed)
    results: dict[str, dict[str, int]] = {}

    for split_name in ("train", "val", "test"):
        for target_class in classes.values():
            (config.output_dir / split_name / target_class).mkdir(parents=True, exist_ok=True)

    for source_class, target_class in classes.items():
        source_dir = config.input_dir / source_class
        if not source_dir.exists():
            raise FileNotFoundError(f"Source class directory not found: {source_dir}")

        files = [path for path in source_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
        rng.shuffle(files)

        total = len(files)
        train_end = int(total * config.train_ratio)
        validation_end = int(total * (config.train_ratio + config.validation_ratio))

        grouped = {
            "train": files[:train_end],
            "val": files[train_end:validation_end],
            "test": files[validation_end:],
        }

        results[target_class] = {}
        for split_name, split_files in grouped.items():
            copied = 0
            destination_dir = config.output_dir / split_name / target_class
            for source_path in split_files:
                destination_path = destination_dir / source_path.name
                if destination_path.exists():
                    destination_path = destination_dir / f"{source_path.stem}_{rng.randint(1000, 9999)}{source_path.suffix.lower()}"
                shutil.copy2(source_path, destination_path)
                copied += 1
            results[target_class][split_name] = copied

    return results
