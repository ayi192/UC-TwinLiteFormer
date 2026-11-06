"""Validate sample counts, paired filenames, image dimensions and label values."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def check(root, expected_count=None):
    root = Path(root)
    groups = ("images", "drivable_area_annotations", "lane_line_annotations")
    # 192 is the boundary value present in the original annotation PNGs.
    expected_values = {"drivable_area_annotations": {0, 128, 192, 255}, "lane_line_annotations": {0, 64, 128, 192, 255}}
    for split in ("train", "val"):
        files = {}
        for group in groups:
            extension = ".jpg" if group == "images" else ".png"
            folder = root / group / split
            files[group] = {p.stem: p for p in folder.glob("*" + extension)}
            if not files[group]:
                raise ValueError("No images in {}".format(folder))
            if expected_count is not None and len(files[group]) != expected_count:
                raise ValueError("{}: expected {}, found {}".format(folder, expected_count, len(files[group])))
        names = set(files["images"])
        if any(set(files[g]) != names for g in groups):
            raise ValueError("Unpaired image and mask filenames in " + split)
        for name in sorted(names):
            with Image.open(files["images"][name]) as image:
                image.load()
                size = image.size
            for group in groups[1:]:
                with Image.open(files[group][name]) as mask:
                    if mask.size != size:
                        raise ValueError("Image/mask dimensions differ: " + str(files[group][name]))
                    values = set(np.unique(np.asarray(mask.convert("L"))).tolist())
                    if not values <= expected_values[group]:
                        raise ValueError("Unexpected mask values {}: {}".format(values, files[group][name]))
        print("{}: {} matched image/road/obstacle triples".format(split, len(names)))
    train_names = set(p.stem for p in (root / "images" / "train").glob("*.jpg"))
    val_names = set(p.stem for p in (root / "images" / "val").glob("*.jpg"))
    if train_names & val_names:
        raise ValueError("Train and validation filenames overlap.")
    auxiliary = list((root / "qufeng").glob("*.jpg"))
    if expected_count is not None and len(auxiliary) != expected_count:
        raise ValueError("qufeng must contain {} images".format(expected_count))
    for path in auxiliary:
        with Image.open(path) as image:
            image.verify()
    print("qufeng: {} auxiliary images (not used for training or validation)".format(len(auxiliary)))
    manifest_path = root / "sample_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded_paths = {entry["path"] for entry in manifest["files"]}
        actual_paths = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".png"}}
        if recorded_paths != actual_paths:
            raise ValueError("Image files do not match the sample manifest.")
        for entry in manifest["files"]:
            digest = hashlib.sha256((root / entry["path"]).read_bytes()).hexdigest()
            if digest != entry["sha256"]:
                raise ValueError("Checksum mismatch: " + entry["path"])
        print("Manifest verified: {} SHA-256 checksums".format(len(manifest["files"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "datasets")
    parser.add_argument("--expected-count", type=int, default=None)
    args = parser.parse_args()
    check(args.root, args.expected_count)
