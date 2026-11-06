# Pretrained segmentation weight

`best_model.pth` is the supplied `medium` UC-TwinLiteFormer state dictionary. It loads strictly into `model.model.TwinLiteNetPlus` with Swin and LAFAM enabled, predicts 2 road classes and 4 obstacle classes, and is supplied for inference.

```shell
python val.py --config medium --weight pretrained/best_model.pth --num_workers 0
python demo.py --config medium --weight pretrained/best_model.pth --source datasets/images/val
```

It does not contain optimizer/EMA training state. Use a `checkpoint.pth.tar` produced by `train.py` to resume training. Weights for `nano`, `small` and `large` are not included.
