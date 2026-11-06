# Dataset examples / 数据与标注说明

This directory contains a small public sample of the supplied maize-field dataset. Each image/mask directory contains **10 files**. The original full dataset is not included.

本目录为数据样例：每个图片/掩码文件夹 **10 张**。原图及标注按文件名配对，图像和掩码尺寸均为 **1280 × 720**。

| Directory | Training examples | Validation examples | Role |
|---|---:|---:|---|
| `images/` | 10 | 10 | RGB images / 原图 |
| `drivable_area_annotations/` | 10 | 10 | Road masks / 可通行区域掩码 |
| `lane_line_annotations/` | 10 | 10 | Obstacle masks / 障碍物掩码 |
| `qufeng/` | — | — | 10 auxiliary images / 10 张辅助图片 |

Each split contains 3 normal-corridor (`zhenchang`), 3 human (`ren`), 2 rock (`shitou`) and 2 ditch (`gouhe`) examples. Original split membership and file bytes are preserved. `qufeng/` is a separate unpaired example directory and is not used by `train.py` or `val.py`.

## Pairing / 文件对应关系

```text
images/val/ren112.jpg
drivable_area_annotations/val/ren112.png
lane_line_annotations/val/ren112.png
```

The two masks must have the same basename and dimensions as their RGB image. The `lane_line_annotations` directory name is retained for code compatibility; it contains obstacle labels for this project.

## Mask encoding / 灰度值编码

PNG masks are single-channel grayscale images. Raw values are not the same as network class indices.

| Task | Raw PNG value | Class index | Meaning |
|---|---:|---:|---|
| Road | 0 | 0 | Background / 背景 |
| Road | > 1 | 1 | Traversable area / 可通行区域 |
| Obstacle | 0 | 0 | Background / 背景 |
| Obstacle | 128 | 1 | Human / 行人 |
| Obstacle | 255 | 2 | Rock / 石块 |
| Obstacle | 64 | 3 | Ditch / 沟壑 |

The original road examples contain values **0, 128 and 192**. The supplied loader uses threshold `> 1`, so both 128 and 192 contribute to the road foreground.

The original obstacle PNGs contain **192-valued polygon boundary pixels**. Their bytes are retained. For compatibility, the supplied loader only constructs channels for 0, 128, 255 and 64; 192-valued pixels retain all-zero one-hot targets and become class 0 under `argmax`. This is the original target convention, not a fifth class or a formal ignore-index policy. New annotations should use the four explicit obstacle values unless this convention is intentionally reproduced.

原始可通行区域标注中的 128、192 都会被原加载器阈值化为前景。障碍物标注中的 192 为原始多边形轮廓边界值，未设独立输出类别，原实现保持该位置的 one-hot 目标全 0。样例按原始文件保留，没有转换或重新标注。

Resize or geometrically transform categorical masks with **nearest-neighbor interpolation**. Do not save masks as JPEG, normalize their gray values, or treat them as RGB visualizations. Model predictions exported by `demo.py` use road values 0/255 and obstacle values 0/128/255/64; they do not introduce 192-valued borders.

## Integrity / 完整性检查

From the repository root:

```shell
python tools/check_dataset.py --expected-count 10
```

`sample_manifest.json` records every selected image/mask path and its SHA-256 checksum. Selection uses seed 42 and preserves the original train/validation assignment. These examples are not a new benchmark split and should not be used to claim full-dataset results.

## Full dataset / 使用完整数据

Keep the same directory names and pairing convention, then pass its root to training or evaluation:

```shell
python train.py --config medium --ema --data-root /path/to/datasets --num_workers 0
python val.py --config medium --weight pretrained/best_model.pth --data-root /path/to/datasets --num_workers 0
```

To reproduce the sample export from a separate original archive into a new empty directory:

```shell
python tools/export_sample_dataset.py /path/to/original_datasets /path/to/empty_sample_directory
```
