"""Copy ten examples per data directory without modifying the original dataset."""
import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path


def export(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must be separate directories.")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Use an empty destination; existing data will not be overwritten.")
    plans = []
    rng = random.Random(42)
    for split in ("train", "val"):
        images = source / "images" / split
        selected = []
        # Keep normal corridors and all three obstacle scenarios in both splits.
        for prefix, count in (("zhenchang", 3), ("ren", 3), ("shitou", 2), ("gouhe", 2)):
            candidates = sorted(images.glob(prefix + "*.jpg"))
            selected.extend(rng.sample(candidates, count))
        for image in sorted(selected):
            plans.append((image, Path("images") / split / image.name))
            for kind in ("drivable_area_annotations", "lane_line_annotations"):
                relative = Path(kind) / split / (image.stem + ".png")
                plans.append((source / relative, relative))
    auxiliary = sorted((source / "qufeng").glob("*.jpg"))
    plans.extend((p, Path("qufeng") / p.name) for p in sorted(rng.sample(auxiliary, 10)))
    missing = [str(path) for path, _ in plans if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing paired files: " + ", ".join(missing))
    manifest = {
        "description": "Public examples only; not the full dataset or a benchmark split.",
        "selection_seed": 42,
        "original_counts": {"train": len(list((source / "images" / "train").glob("*.jpg"))),
                            "val": len(list((source / "images" / "val").glob("*.jpg"))),
                            "qufeng": len(auxiliary)},
        "examples_per_image_directory": 10,
        "obstacle_mask_values": {"0": "background", "128": "human", "255": "rock", "64": "ditch", "192": "annotation boundary; no separate output class"},
        "files": [],
    }
    for original, relative in plans:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(original), str(target))
        manifest["files"].append({
            "path": relative.as_posix(),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        })
    (destination / "sample_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Copied {} image/mask files to {}".format(len(plans), destination))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original dataset root")
    parser.add_argument("destination", type=Path, help="Empty output directory")
    options = parser.parse_args()
    export(options.source, options.destination)
