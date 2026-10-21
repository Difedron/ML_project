"""Create two small reproducible fixtures in the existing raw catalog, without downloads."""
import _bootstrap  # noqa: F401
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import load_iris

from meta_selector.datasets.registry import register_dataset
from meta_selector.paths import DEFAULT_PROJECT_ROOT


def create_examples(root=DEFAULT_PROJECT_ROOT):
    iris = load_iris(as_frame=True).frame.rename(columns={"target": "target"})
    rng = np.random.default_rng(42)
    labels = np.tile([0, 1, 2], 30)
    mixed = pd.DataFrame({"numeric": labels + rng.normal(0, .8, len(labels)),
                          "category": np.where(labels == 0, "a", "b"), "flag": labels == 2,
                          "target": labels})
    mixed.loc[[3, 10, 21], "numeric"] = np.nan
    for dataset_id, frame in [("_smoke_iris", iris), ("_smoke_mixed", mixed)]:
        folder = Path(root) / "data/raw" / dataset_id
        metadata = dict(dataset_id=dataset_id, name=f"Smoke fixture {dataset_id}", task_type="classification",
                        target="target", source=dict(type="other", url="", external_id=None),
                        file=dict(name="data.csv", format="csv"), problem=dict(type="multiclass"),
                        notes="Small reproducible integration fixture; excluded from scientific claims.")
        if (folder / "data.csv").exists():
            print(f"Preserved existing payload: {folder}")
        else:
            if folder.exists() and any(folder.iterdir()):
                path = folder / "metadata.yaml"
                if not path.is_file() or yaml.safe_load(path.read_text()) != metadata:
                    raise ValueError(f"Preserved unrelated existing folder; cannot create fixture: {folder}")
            folder.mkdir(parents=True, exist_ok=True)
            frame.to_csv(folder / "data.csv", index=False)
            (folder / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
        report = register_dataset(dataset_id, project_root=root)
        print(f"{dataset_id}: {report.status}")


if __name__ == "__main__":
    create_examples()
