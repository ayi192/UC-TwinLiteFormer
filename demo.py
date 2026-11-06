"""Run UC-TwinLiteFormer on images or videos and save overlays and label PNGs."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from model.model import TwinLiteNetPlus

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
OBSTACLE_VALUES = np.array([0, 128, 255, 64], dtype=np.uint8)
# BGR: background, human, rock, ditch.
OBSTACLE_COLORS = np.array([[0, 0, 0], [70, 70, 235], [30, 200, 255], [220, 80, 180]], dtype=np.uint8)


@torch.no_grad()
def segment(frame, model, half=False):
    height, width = frame.shape[:2]
    ratio = min(640 / width, 384 / height)
    resized_width, resized_height = round(width * ratio), round(height * ratio)
    left = (640 - resized_width) // 2
    top = (384 - resized_height) // 2
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, 384 - resized_height - top,
                               left, 640 - resized_width - left, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1))
    tensor = torch.from_numpy(rgb).unsqueeze(0).to(next(model.parameters()).device)
    tensor = (tensor.half() if half else tensor.float()) / 255.0
    outputs = model(tensor)
    labels = []
    for logits in outputs:
        cropped = logits[:, :, top:top + resized_height, left:left + resized_width]
        restored = torch.nn.functional.interpolate(cropped.float(), size=(height, width), mode="bilinear", align_corners=False)
        labels.append(restored.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8))
    road, obstacle = labels
    color = frame.copy()
    color[road == 1] = (65, 195, 70)
    for index in (1, 2, 3):
        color[obstacle == index] = OBSTACLE_COLORS[index]
    overlay = cv2.addWeighted(frame, 0.55, color, 0.45, 0)
    return overlay, road * 255, OBSTACLE_VALUES[obstacle]


def detect(args):
    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if args.half and device.type != "cuda":
        raise ValueError("--half requires a CUDA device")
    model = TwinLiteNetPlus(args).to(device)
    model.load_state_dict(torch.load(args.weight, map_location=device))
    model.eval()
    if args.half:
        model.half()
    source = Path(args.source)
    files = sorted(p for p in source.iterdir() if p.is_file()) if source.is_dir() else [source]
    files = [p for p in files if p.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS]
    if not files:
        raise FileNotFoundError("No supported images or videos found: " + str(source))
    for folder in ("overlays", "road_masks", "obstacle_masks", "videos"):
        (args.save_dir / folder).mkdir(parents=True, exist_ok=True)
    for path in files:
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            frame = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Cannot decode image: " + str(path))
            overlay, road, obstacle = segment(frame, model, args.half)
            for folder, data in (("overlays", overlay), ("road_masks", road), ("obstacle_masks", obstacle)):
                target = args.save_dir / folder / (path.stem + ".png")
                ok, encoded = cv2.imencode(".png", data)
                if not ok:
                    raise RuntimeError("Cannot encode output: " + str(target))
                encoded.tofile(str(target))
        else:
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise ValueError("Cannot open video: " + str(path))
            writer = None
            try:
                count = 0
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    overlay, _, _ = segment(frame, model, args.half)
                    if writer is None:
                        fps = capture.get(cv2.CAP_PROP_FPS)
                        fps = fps if np.isfinite(fps) and fps > 0 else 25.0
                        writer = cv2.VideoWriter(str(args.save_dir / "videos" / (path.stem + "_segmentation.mp4")),
                                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
                        if not writer.isOpened():
                            raise RuntimeError("Cannot create video writer")
                    writer.write(overlay)
                    count += 1
                    if args.max_frames > 0 and count >= args.max_frames:
                        break
                if count == 0:
                    raise ValueError("No readable video frames: " + str(path))
            finally:
                capture.release()
                if writer is not None:
                    writer.release()
        print("Saved: " + path.name)
    print("Results: " + str(args.save_dir.resolve()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", type=str, default="pretrained/best_model.pth")
    parser.add_argument("--source", type=str, default="datasets/images/val")
    parser.add_argument("--config", choices=["nano", "small", "medium", "large"], default="medium")
    parser.add_argument("--save-dir", type=Path, default=Path("inference/output"))
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda:0, etc.")
    parser.add_argument("--half", action="store_true", help="FP16 inference on CUDA")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum video frames; 0 means all")
    detect(parser.parse_args())
