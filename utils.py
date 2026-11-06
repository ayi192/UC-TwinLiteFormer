import torch
import numpy as np
from IOUEval import SegmentationMetric
import logging
import logging.config
from tqdm import tqdm
import os
import torch.nn as nn
from const import *
import yaml
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import os
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist


LOGGING_NAME="custom"
def set_logging(name=LOGGING_NAME, verbose=True):
    # sets up logging for the given name
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                'format': '%(message)s'}},
        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'formatter': name,
                'level': level,}},
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False,}}})
set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def poly_lr_scheduler(args, hyp, optimizer, epoch, power=1.5):
    lr = round(hyp['lr'] * (1 - epoch / args.max_epochs) ** power, 8)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(args, train_loader, model, criterion, optimizer, epoch,scaler,verbose=False,ema=None):
    model.train()
    print("epoch: ", epoch)
    total_batches = len(train_loader)
    pbar = enumerate(train_loader)
    if verbose:
        LOGGER.info(('\n' + '%13s' * 5) % ('Epoch','TverskyLoss','FocalLoss','DiceLoss','TotalLoss'))
        pbar = tqdm(pbar, total=total_batches, bar_format='{l_bar}{bar:10}{r_bar}')
    for i, (_,input, target) in pbar:
        optimizer.zero_grad()
        input = input.to(next(model.parameters()).device).float() / 255.0
        output = model(input)
        with torch.cuda.amp.autocast(enabled=input.is_cuda):
            focal_loss,tversky_loss,dice_loss,loss = criterion(output,target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)
        if verbose:
            pbar.set_description(('%13s' * 1 + '%13.4g' * 4) %
                                     (f'{epoch}/{args.max_epochs - 1}', tversky_loss, focal_loss, dice_loss, loss.item()))
    return ema if ema is not None else None


def extract_navigation_line_triangle(drivable_mask):
    """
    使用最小三角形拟合方法提取导航线
    
    参数:
        drivable_mask: numpy.ndarray, 可行驶区域的分割掩码 (0-1)
    
    返回:
        navigation_line: numpy.ndarray, 导航线的起点和终点坐标 [[x1,y1], [x2,y2]]
    """
    # 确保输入是二值图像
    if drivable_mask.max() > 1:
        drivable_mask = drivable_mask / 255.0
    # 获取可行驶区域的轮廓
    contours, _ = cv2.findContours(drivable_mask.astype(np.uint8), 
                                 cv2.RETR_EXTERNAL, 
                                 cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # 找到最大的轮廓
    max_contour = max(contours, key=cv2.contourArea)
    # 对轮廓进行多边形近似
    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    points = approx.reshape(-1, 2)
    # 如果近似后的点数大于3，取最高点及其相邻两个点
    if len(points) > 3:
        top_idx = np.argmin(points[:, 1])
        top_point = points[top_idx]
        adj1 = points[(top_idx - 1) % len(points)]
        adj2 = points[(top_idx + 1) % len(points)]
        triangle = np.array([top_point, adj1, adj2])
        points = triangle
    # 如果点数不是3，使用最小外接三角形
    if len(points) != 3:
        rect = cv2.minAreaRect(max_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        top_point = box[np.argmin(box[:, 1])]
        bottom_points = box[np.argsort(box[:, 1])[-2:]]
        points = np.array([top_point, bottom_points[0], bottom_points[1]])
    # 找到三角形的顶点（最上方的点）
    top_point = points[np.argmin(points[:, 1])]
    # 找到底边的两个点
    bottom_points = points[np.argsort(points[:, 1])[-2:]]
    # 计算底边中点
    base_mid = np.mean(bottom_points, axis=0)
    # 返回导航线（从底边中点到顶点）
    return np.array([base_mid, top_point])

def visualize_navigation_line_triangle(image, drivable_mask, navigation_line, save_path=None):
    """
    可视化基于三角形拟合的导航线
    
    参数:
        image: numpy.ndarray, 原始图像
        drivable_mask: numpy.ndarray, 可行驶区域的分割掩码
        navigation_line: numpy.ndarray, 导航线的起点和终点坐标 [[x1,y1], [x2,y2]]
        save_path: str, 保存路径（可选）
    """
    # 创建可视化图像
    vis_img = image.copy()
    
    # 绘制可行驶区域
    vis_img[drivable_mask > 0] = vis_img[drivable_mask > 0] * 0.7 + np.array([0, 255, 0]) * 0.3
    
    # 绘制导航线
    if navigation_line is not None:
        start_point = navigation_line[0].astype(np.int32)
        end_point = navigation_line[1].astype(np.int32)
        # 只绘制导航线，不绘制端点
        cv2.line(vis_img, tuple(start_point), tuple(end_point), (0, 0, 255), 2)
    
    if save_path:
        cv2.imwrite(save_path, vis_img)
    
    return vis_img

# === 以下为辅助函数和融合方法相关内容 ===
def extract_lane_endpoints_center(lane_mask, N=5):
    """
    提取车道线掩码的两个端点（最上和最下N行的x中值/均值）
    lane_mask: numpy.ndarray, 0/1 掩码
    返回: [[x1, y1], [x2, y2]]，分别为上端点和下端点
    """
    if lane_mask.max() > 1:
        lane_mask = lane_mask / 255.0
    lane_mask = (lane_mask > 0.5).astype(np.uint8)
    points = np.column_stack(np.where(lane_mask > 0))
    if len(points) == 0:
        return None
    # 按y排序
    points = points[np.argsort(points[:, 0])]
    # 取最上面N个点
    top_N = points[:N]
    # 取最下面N个点
    bottom_N = points[-N:]
    # 取x的中值和y的均值
    top_x = int(np.median(top_N[:, 1]))
    top_y = int(np.mean(top_N[:, 0]))
    bottom_x = int(np.median(bottom_N[:, 1]))
    bottom_y = int(np.mean(bottom_N[:, 0]))
    return np.array([[top_x, top_y], [bottom_x, bottom_y]])

def get_x_at_y(p1, p2, y):
    if p2[1] == p1[1]:
        return (p1[0] + p2[0]) / 2
    return p1[0] + (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])

def compare_lines(pred_line, gt_line, H):
    y1 = int(H * 0.5)
    y2 = int(H * 0.25)
    x1_pred = get_x_at_y(pred_line[0], pred_line[1], y1)
    x2_pred = get_x_at_y(pred_line[0], pred_line[1], y2)
    x1_gt = get_x_at_y(gt_line[0], gt_line[1], y1)
    x2_gt = get_x_at_y(gt_line[0], gt_line[1], y2)
    X1 = abs(x1_pred - x1_gt)
    X2 = abs(x2_pred - x2_gt)
    return X1, X2

def sort_line_by_y(line):
    if line[0,1] > line[1,1]:
        return np.array([line[1], line[0]])
    return line

def heading_angle(line):
    dx = line[1,0] - line[0,0]
    dy = line[1,1] - line[0,1]
    angle = np.arctan2(dy, dx) * 180 / np.pi
    return angle

def visualize_lane_endpoints(image, lane_mask, endpoints, gt_endpoints=None, save_path=None, y1_ratio=0.85, y2_ratio=0.7):
    vis_img = image.copy()
    H, W = vis_img.shape[:2]
    vis_img[lane_mask > 0] = vis_img[lane_mask > 0] * 0.7 + np.array([255, 0, 0]) * 0.3
    cv2.circle(vis_img, (0, H-1), 8, (0, 0, 0), -1)
    cv2.putText(vis_img, "Origin", (10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    y1 = int(H * y1_ratio)
    y2 = int(H * y2_ratio)
    cv2.line(vis_img, (0, y1), (W-1, y1), (255, 0, 0), 2)
    cv2.line(vis_img, (0, y2), (W-1, y2), (255, 0, 0), 2)
    cv2.putText(vis_img, "y1", (10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.putText(vis_img, "y2", (10, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    if gt_endpoints is not None:
        gt_pt1, gt_pt2 = gt_endpoints.astype(np.int32)
        cv2.line(vis_img, tuple(gt_pt1), tuple(gt_pt2), (0, 255, 0), 2)
        cv2.circle(vis_img, tuple(gt_pt1), 9, (0, 255, 0), -1)
        cv2.circle(vis_img, tuple(gt_pt2), 9, (0, 255, 0), -1)
        cv2.putText(vis_img, "GT", tuple(gt_pt1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    if endpoints is not None:
        pt1, pt2 = endpoints.astype(np.int32)
        cv2.line(vis_img, tuple(pt1), tuple(pt2), (0, 0, 255), 2)
        cv2.circle(vis_img, tuple(pt1), 7, (0, 0, 255), -1)
        cv2.circle(vis_img, tuple(pt2), 7, (0, 0, 255), -1)
        cv2.putText(vis_img, "Pred", tuple(pt1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    if endpoints is not None and gt_endpoints is not None:
        def get_x_at_y(p1, p2, y):
            if p2[1] == p1[1]:
                return int((p1[0] + p2[0]) / 2)
            return int(p1[0] + (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]))
        x1_pred = get_x_at_y(pt1, pt2, y1)
        x2_pred = get_x_at_y(pt1, pt2, y2)
        x1_gt = get_x_at_y(gt_pt1, gt_pt2, y1)
        x2_gt = get_x_at_y(gt_pt1, gt_pt2, y2)
        cv2.circle(vis_img, (x1_pred, y1), 9, (0,0,255), -1)
        cv2.circle(vis_img, (x1_gt, y1), 9, (0,255,0), -1)
        cv2.circle(vis_img, (x2_pred, y2), 9, (0,0,255), -1)
        cv2.circle(vis_img, (x2_gt, y2), 9, (0,255,0), -1)
        cv2.line(vis_img, (x1_pred, y1), (x1_gt, y1), (255,0,255), 2)
        cv2.line(vis_img, (x2_pred, y2), (x2_gt, y2), (255,0,255), 2)
        X1 = abs(x1_pred - x1_gt)
        X2 = abs(x2_pred - x2_gt)
        cv2.putText(vis_img, f'X1={X1}', (min(x1_pred, x1_gt)+10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
        cv2.putText(vis_img, f'X2={X2}', (min(x2_pred, x2_gt)+10, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
    if save_path:
        cv2.imwrite(save_path, vis_img)
    return vis_img

# === 替换val函数为融合方法版本 ===
@torch.no_grad()
def val(val_loader=None, model=None, half=False, args=None):
    if args is None:
        from argparse import Namespace
        args = Namespace(vis=False, verbose=False)
    elif not hasattr(args, 'vis'):
        args.vis = False
    print(f"args: {args}")
    print(f"Visualization Enabled: {args.vis}")
    if args.vis == True:
        print(f"可视化已经开启")
    model.eval()
    DA = SegmentationMetric(2)  # 可驾驶区域：2类
    LL = SegmentationMetric(4)  # 车道线：4类（背景、车道线、石头、沟壑）
    da_acc_seg = AverageMeter()
    da_IoU_seg = AverageMeter()
    da_mIoU_seg = AverageMeter()
    ll_acc_seg = AverageMeter()
    ll_IoU_seg = AverageMeter()
    ll_mIoU_seg = AverageMeter()
    if args.vis:
        os.makedirs(args.save_vis_dir, exist_ok=True)
        os.makedirs(args.save_mask_dir, exist_ok=True)
        total_vis_saved = 0
        total_mask_saved = 0
    total_batches = len(val_loader)
    pbar = enumerate(val_loader)
    if args.verbose:
        pbar = tqdm(pbar, total=total_batches)
    all_X1 = []
    all_X2 = []
    all_X1_da = []
    all_X2_da = []
    all_X1_ll = []
    all_X2_ll = []
    all_angle_err_fused = []
    all_angle_err_da = []
    all_angle_err_ll = []
    
    # 添加数据统计
    total_samples = 0
    samples_with_lane = 0
    total_lane_pixels = 0
    total_stone_pixels = 0  # 添加石头像素统计
    total_gully_pixels = 0  # 添加沟壑像素统计
    total_pixels = 0
    
    # 添加车道线IOU统计
    lane_iou_list = []
    lane_miou_list = []
    stone_iou_list = []
    stone_miou_list = []
    gully_iou_list = []
    gully_miou_list = []
    
    # 添加全局像素统计
    global_lane_metric = SegmentationMetric(4)  # 车道线全局统计
    global_stone_metric = SegmentationMetric(4)  # 石头全局统计
    global_gully_metric = SegmentationMetric(4)  # 沟壑全局统计
    global_all_metric = SegmentationMetric(4)    # 所有样本全局统计
    
    # 添加车道线存在性预测统计
    lane_existence_tp = 0  # 真正例：预测有车道线，实际有车道线
    lane_existence_fp = 0  # 假正例：预测有车道线，实际无车道线
    lane_existence_tn = 0  # 真负例：预测无车道线，实际无车道线
    lane_existence_fn = 0  # 假负例：预测无车道线，实际有车道线
    
    for i, (img_names, input, target) in pbar:
        input = input.to(next(model.parameters()).device)
        input = input.half() / 255.0 if half else input.float() / 255.0
        input_var = input
        target_var = target
        
        # 统计车道线数据分布
        target_ll = target[1]
        _, ll_gt = torch.max(target_ll, 1)
        
        # 统计三类别数据
        batch_samples_with_lane = ((ll_gt == 1).sum(dim=(1,2)) > 0).sum().item()  # 车道线
        batch_samples_with_stone = ((ll_gt == 2).sum(dim=(1,2)) > 0).sum().item()  # 石头
        batch_samples_with_gully = ((ll_gt == 3).sum(dim=(1,2)) > 0).sum().item()  # 沟壑
        batch_samples_with_any = batch_samples_with_lane + batch_samples_with_stone + batch_samples_with_gully
        
        samples_with_lane += ((ll_gt > 0).sum(dim=(1, 2)) > 0).sum().item()
        total_samples += input.size(0)
        total_lane_pixels += (ll_gt == 1).sum().item()  # 车道线像素
        total_stone_pixels += (ll_gt == 2).sum().item()  # 石头像素
        total_gully_pixels += (ll_gt == 3).sum().item()  # 沟壑像素
        total_pixels += ll_gt.numel()
        
        with torch.no_grad():
            output = model(input_var)
        out_da = output[0]
        target_da = target[0]
        _, da_predict = torch.max(out_da, 1)
        da_predict_cropped = da_predict[:, 12:-12]
        _, da_gt = torch.max(target_da, 1)
        DA.reset()
        DA.addBatch(da_predict_cropped.cpu(), da_gt.cpu())
        da_acc = DA.pixelAccuracy()
        da_IoU = DA.IntersectionOverUnion()
        da_mIoU = DA.meanIntersectionOverUnion()
        da_acc_seg.update(da_acc, input.size(0))
        da_IoU_seg.update(da_IoU, input.size(0))
        da_mIoU_seg.update(da_mIoU, input.size(0))
        out_ll = output[1]
        target_ll = target[1]
        _, ll_predict = torch.max(out_ll, 1)
        ll_predict_cropped = ll_predict[:, 12:-12]
        _, ll_gt = torch.max(target_ll, 1)
        
        # 统计车道线存在性预测
        for j in range(input.size(0)):
            # 获取真实标签
            gt_label = ll_gt[j]  # [H, W]
            
            # 检查是否有车道线或石头
            gt_has_lane = (gt_label == 1).sum() > 0  # 车道线类别
            gt_has_stone = (gt_label == 2).sum() > 0  # 石头类别
            gt_has_gully = (gt_label == 3).sum() > 0  # 沟壑类别
            gt_has_any = gt_has_lane or gt_has_stone or gt_has_gully  # 有任何目标
            
            # 获取预测标签
            pred_label = ll_predict_cropped[j]  # [H, W]
            pred_has_lane = (pred_label == 1).sum() > 0
            pred_has_stone = (pred_label == 2).sum() > 0
            pred_has_gully = (pred_label == 3).sum() > 0
            pred_has_any = pred_has_lane or pred_has_stone or pred_has_gully
            
            # 统计车道线预测
            if gt_has_lane and pred_has_lane:
                lane_existence_tp += 1
            elif gt_has_lane and not pred_has_lane:
                lane_existence_fn += 1
            elif not gt_has_lane and pred_has_lane:
                lane_existence_fp += 1
            elif not gt_has_lane and not pred_has_lane:
                lane_existence_tn += 1
        
        # 使用全局像素统计法计算车道线和石头的IOU
        for j in range(input.size(0)):
            gt_label = ll_gt[j]  # [H, W]
            pred_label = ll_predict_cropped[j]  # [H, W]
            
            # 为所有样本添加全局统计
            global_all_metric.addBatch(pred_label.cpu().unsqueeze(0), gt_label.cpu().unsqueeze(0))
            
            # 检查是否有车道线
            if (gt_label == 1).sum() > 0:  # 如果有车道线
                lane_iou_list.append(1)  # 记录有车道线的样本数量
                # 添加到全局统计
                global_lane_metric.addBatch(pred_label.cpu().unsqueeze(0), gt_label.cpu().unsqueeze(0))
            
            # 检查是否有石头
            if (gt_label == 2).sum() > 0:  # 如果有石头
                stone_iou_list.append(1)  # 记录有石头的样本数量
                # 添加到全局统计
                global_stone_metric.addBatch(pred_label.cpu().unsqueeze(0), gt_label.cpu().unsqueeze(0))
            
            # 检查是否有沟壑
            if (gt_label == 3).sum() > 0:  # 如果有沟壑
                gully_iou_list.append(1)  # 记录有沟壑的样本数量
                # 添加到全局统计
                global_gully_metric.addBatch(pred_label.cpu().unsqueeze(0), gt_label.cpu().unsqueeze(0))
        
        # 对所有样本计算整体指标（保持原有逻辑）
        LL.reset()
        LL.addBatch(ll_predict_cropped.cpu(), ll_gt.cpu())
        ll_acc = LL.lineAccuracy()
        ll_IoU = LL.IntersectionOverUnion()
        ll_mIoU = LL.meanIntersectionOverUnion()
        ll_acc_seg.update(ll_acc, input.size(0))
        ll_IoU_seg.update(ll_IoU[1], input.size(0))  # 使用车道线类别的IOU
        ll_mIoU_seg.update(ll_mIoU, input.size(0))
        if args is not None and getattr(args, 'vis', False):
            _, da_predict_full = torch.max(out_da, 1)
            _, ll_predict_full = torch.max(out_ll, 1)
            for j in range(input_var.size(0)):
                orig_img_tensor = input_var[j].detach().cpu()
                orig_img = (orig_img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                da_mask = da_predict_full[j].detach().cpu().numpy().astype(np.uint8)
                ll_mask = ll_predict_full[j].detach().cpu().numpy().astype(np.uint8)
                da_mask_cropped = da_mask[12:-12, :]
                ll_mask_cropped = ll_mask[12:-12, :]
                if getattr(args, 'save_mask', True):
                    if args.save_mask_max == -1 or total_mask_saved < args.save_mask_max:
                        base_name = os.path.basename(img_names[j])
                        mask_filename = f"{os.path.splitext(base_name)[0]}_mask.png"
                        mask_save_path = os.path.join(args.save_mask_dir, mask_filename)
                        print(f"保存掩码到: {mask_save_path}")
                        print(f"掩码形状: {ll_mask_cropped.shape}, 值范围: [{ll_mask_cropped.min()}, {ll_mask_cropped.max()}]")
                        os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
                        cv2.imwrite(mask_save_path, (ll_mask_cropped * 255).astype(np.uint8))
                        total_mask_saved += 1
                        print(f"掩码保存完成 ({total_mask_saved})")
                if args.vis_max == -1 or total_vis_saved < args.vis_max:
                    navigation_line = extract_navigation_line_triangle(da_mask_cropped)
                    orig_img_cropped = orig_img[12:-12, :, :]
                    vis_img = visualize_navigation_line_triangle(orig_img_cropped, da_mask_cropped, navigation_line)
                    img_name = os.path.basename(img_names[j])
                    cv2.putText(vis_img, img_name, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    save_name = f"{os.path.splitext(img_name)[0]}_nav.png"
                    save_path = os.path.join(args.save_vis_dir, save_name)
                    cv2.imwrite(save_path, vis_img)
                    ll_vis_img = visualize_lane_mask(orig_img_cropped, ll_mask_cropped)
                    lane_endpoints_geom = extract_lane_endpoints_center(ll_mask_cropped, N=5)
                    if lane_endpoints_geom is not None:
                        pt1, pt2 = lane_endpoints_geom.astype(np.int32)
                        cv2.line(ll_vis_img, tuple(pt1), tuple(pt2), (0, 0, 255), 2)
                        cv2.circle(ll_vis_img, tuple(pt1), 7, (0, 0, 255), -1)
                        cv2.circle(ll_vis_img, tuple(pt2), 7, (0, 0, 255), -1)
                    ll_save_name = f"{os.path.splitext(img_name)[0]}_lane.png"
                    ll_save_path = os.path.join(args.save_vis_dir, ll_save_name)
                    cv2.imwrite(ll_save_path, ll_vis_img)
                    gt_mask_path = img_names[j].replace("images", "lane_line_annotations").replace("jpg", "png")
                    gt_mask = cv2.imread(gt_mask_path, 0)
                    gt_mask = cv2.resize(gt_mask, (ll_mask_cropped.shape[1], ll_mask_cropped.shape[0]))
                    gt_line = extract_lane_endpoints_center(gt_mask, N=5)
                    da_line = extract_navigation_line_triangle(da_mask_cropped)
                    ll_line = extract_lane_endpoints_center(ll_mask_cropped, N=5)
                    conf1 = getattr(args, 'da_segment_results', [0,0,0.9])[2]
                    conf2 = getattr(args, 'll_segment_results', [0,0,0.5])[1]
                    conf1_norm = (conf1 - 0.8) / (1.0 - 0.8)
                    conf2_norm = (conf2 - 0.1) / (0.4 - 0.1)
                    conf1_norm = np.clip(conf1_norm, 0, 1)
                    conf2_norm = np.clip(conf2_norm, 0, 1)
                    if da_line is not None and ll_line is not None:
                        da_line = sort_line_by_y(da_line)
                        ll_line = sort_line_by_y(ll_line)
                        alpha = 0.5
                        beta = 0.5
                        fused_line = alpha * da_line + beta * ll_line
                    else:
                        fused_line = None
                    lane_endpoints_vis = visualize_lane_endpoints(
                        orig_img_cropped, ll_mask_cropped, fused_line, gt_endpoints=gt_line)
                    lane_endpoints_save_name = f"{os.path.splitext(img_name)[0]}_lane_endpoints.png"
                    lane_endpoints_save_path = os.path.join(args.save_vis_dir, lane_endpoints_save_name)
                    cv2.imwrite(lane_endpoints_save_path, lane_endpoints_vis)
                    total_vis_saved += 1
                da_line = extract_navigation_line_triangle(da_mask_cropped)
                ll_line = extract_lane_endpoints_center(ll_mask_cropped, N=5)
                gt_mask_path = img_names[j].replace("images", "lane_line_annotations").replace("jpg", "png")
                gt_mask = cv2.imread(gt_mask_path, 0)
                gt_mask = cv2.resize(gt_mask, (ll_mask_cropped.shape[1], ll_mask_cropped.shape[0]))
                gt_line = extract_lane_endpoints_center(gt_mask, N=5)
                conf1 = getattr(args, 'da_segment_results', [0,0,0.9])[2]
                conf2 = getattr(args, 'll_segment_results', [0,0,0.5])[1]
                conf1_norm = (conf1 - 0.8) / (1.0 - 0.8)
                conf2_norm = (conf2 - 0.1) / (0.4 - 0.1)
                conf1_norm = np.clip(conf1_norm, 0, 1)
                conf2_norm = np.clip(conf2_norm, 0, 1)
                if da_line is not None and ll_line is not None and gt_line is not None:
                    da_line = sort_line_by_y(da_line)
                    ll_line = sort_line_by_y(ll_line)
                    alpha = 0.5
                    beta = 0.5
                    fused_line = alpha * da_line + beta * ll_line
                    H = ll_mask_cropped.shape[0]
                    X1, X2 = compare_lines(fused_line, gt_line, H)
                    all_X1.append(X1)
                    all_X2.append(X2)
                    X1_da, X2_da = compare_lines(da_line, gt_line, H)
                    all_X1_da.append(X1_da)
                    all_X2_da.append(X2_da)
                    X1_ll, X2_ll = compare_lines(ll_line, gt_line, H)
                    all_X1_ll.append(X1_ll)
                    all_X2_ll.append(X2_ll)
                    angle_fused = heading_angle(fused_line)
                    angle_gt = heading_angle(gt_line)
                    angle_da = heading_angle(da_line)
                    angle_ll = heading_angle(ll_line)
                    err_fused = abs(angle_fused - angle_gt)
                    err_fused = min(err_fused, 360 - err_fused)
                    all_angle_err_fused.append(err_fused)
                    err_da = abs(angle_da - angle_gt)
                    err_da = min(err_da, 360 - err_da)
                    all_angle_err_da.append(err_da)
                    err_ll = abs(angle_ll - angle_gt)
                    err_ll = min(err_ll, 360 - err_ll)
                    all_angle_err_ll.append(err_ll)
    if len(all_X1) > 0 and len(all_X2) > 0:
        mean_X1 = sum(all_X1) / len(all_X1)
        mean_X2 = sum(all_X2) / len(all_X2)
        print(f"融合导航线 val集平均像素偏差：X1={mean_X1:.2f}, X2={mean_X2:.2f}")
    if len(all_X1_da) > 0 and len(all_X2_da) > 0:
        mean_X1_da = sum(all_X1_da) / len(all_X1_da)
        mean_X2_da = sum(all_X2_da) / len(all_X2_da)
        print(f"可驾驶区域导航线 val集平均像素偏差：X1={mean_X1_da:.2f}, X2={mean_X2_da:.2f}")
    if len(all_X1_ll) > 0 and len(all_X2_ll) > 0:
        mean_X1_ll = sum(all_X1_ll) / len(all_X1_ll)
        mean_X2_ll = sum(all_X2_ll) / len(all_X2_ll)
        print(f"直接预测导航线 val集平均像素偏差：X1={mean_X1_ll:.2f}, X2={mean_X2_ll:.2f}")
    def print_angle_metrics(name, err_list):
        if len(err_list) == 0:
            if args is not None and hasattr(args, 'vis') and args.vis:
                print(f'{name} 航向角无有效样本')
            return
        mae = np.mean(np.abs(err_list))
        sd = np.std(err_list)
        cr = np.mean(np.array(err_list) < 4) * 100
        print(f'{name} 航向角 MAE: {mae:.2f}° | SD: {sd:.2f}° | CR(<4°): {cr:.2f}%')
    print_angle_metrics('融合方法导航线', all_angle_err_fused)
    print_angle_metrics('可驾驶区域导航线', all_angle_err_da)
    print_angle_metrics('直接预测导航线', all_angle_err_ll)
    
    # 打印数据分布统计
    print(f"\n=== 数据分布统计 ===")
    print(f"总样本数: {total_samples}")
    print(f"包含车道线、石头或沟壑的样本数: {samples_with_lane}")
    print(f"车道线、石头或沟壑样本占比: {samples_with_lane/total_samples*100:.2f}%")
    print(f"车道线像素占比: {total_lane_pixels/total_pixels*100:.4f}%")
    print(f"石头像素占比: {total_stone_pixels/total_pixels*100:.4f}%")
    print(f"沟壑像素占比: {total_gully_pixels/total_pixels*100:.4f}%")
    print(f"目标像素总占比: {(total_lane_pixels+total_stone_pixels+total_gully_pixels)/total_pixels*100:.4f}%")
    print(f"==================\n")
    
    # 打印车道线IOU统计
    print(f"=== 车道线分割统计 (全局像素统计) ===")
    if len(lane_iou_list) > 0:
        # 使用全局统计计算IOU
        global_all_iou = global_all_metric.IntersectionOverUnion()
        global_all_miou = global_all_metric.meanIntersectionOverUnion()
        print(f"车道线样本数: {len(lane_iou_list)}")
        print(f"车道线全局IOU: {global_all_iou[1]:.3f}")  # 车道线类别的全局IOU
        print(f"车道线全局mIOU: {global_all_miou:.3f}")
    else:
        print(f"没有找到包含车道线的样本")
    print(f"========================\n")
    
    # 打印石头IOU统计
    print(f"=== 石头分割统计 (全局像素统计) ===")
    if len(stone_iou_list) > 0:
        # 使用全局统计计算IOU
        global_all_iou = global_all_metric.IntersectionOverUnion()
        global_all_miou = global_all_metric.meanIntersectionOverUnion()
        print(f"石头样本数: {len(stone_iou_list)}")
        print(f"石头全局IOU: {global_all_iou[2]:.3f}")  # 石头类别的全局IOU
        print(f"石头全局mIOU: {global_all_miou:.3f}")
    else:
        print(f"没有找到包含石头的样本")
    print(f"========================\n")
    
    # 打印沟壑IOU统计
    print(f"=== 沟壑分割统计 (全局像素统计) ===")
    if len(gully_iou_list) > 0:
        # 使用全局统计计算IOU
        global_all_iou = global_all_metric.IntersectionOverUnion()
        global_all_miou = global_all_metric.meanIntersectionOverUnion()
        print(f"沟壑样本数: {len(gully_iou_list)}")
        print(f"沟壑全局IOU: {global_all_iou[3]:.3f}")  # 沟壑类别的全局IOU
        print(f"沟壑全局mIOU: {global_all_miou:.3f}")
    else:
        print(f"没有找到包含沟壑的样本")
    print(f"========================\n")
    
    # 打印整体全局统计
    print(f"=== 整体分割统计 (全局像素统计) ===")
    global_all_iou = global_all_metric.IntersectionOverUnion()
    global_all_miou = global_all_metric.meanIntersectionOverUnion()
    print(f"总样本数: {total_samples}")
    print(f"背景全局IOU: {global_all_iou[0]:.3f}")
    print(f"行人全局IOU: {global_all_iou[1]:.3f}")
    print(f"石头全局IOU: {global_all_iou[2]:.3f}")
    print(f"沟壑全局IOU: {global_all_iou[3]:.3f}")
    print(f"整体全局mIOU: {global_all_miou:.3f}")
    print(f"========================\n")
    
    # 打印车道线存在性预测统计
    total_existence = lane_existence_tp + lane_existence_fp + lane_existence_tn + lane_existence_fn
    if total_existence > 0:
        accuracy = (lane_existence_tp + lane_existence_tn) / total_existence
        precision = lane_existence_tp / (lane_existence_tp + lane_existence_fp) if (lane_existence_tp + lane_existence_fp) > 0 else 0
        recall = lane_existence_tp / (lane_existence_tp + lane_existence_fn) if (lane_existence_tp + lane_existence_fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"=== 车道线存在性预测统计 ===")
        print(f"总样本数: {total_existence}")
        print(f"真正例(TP): {lane_existence_tp} - 预测有车道线，实际有车道线")
        print(f"假正例(FP): {lane_existence_fp} - 预测有车道线，实际无车道线")
        print(f"真负例(TN): {lane_existence_tn} - 预测无车道线，实际无车道线")
        print(f"假负例(FN): {lane_existence_fn} - 预测无车道线，实际有车道线")
        print(f"准确率(Accuracy): {accuracy:.3f}")
        print(f"精确率(Precision): {precision:.3f}")
        print(f"召回率(Recall): {recall:.3f}")
        print(f"F1分数: {f1_score:.3f}")
        print(f"==========================\n")
    else:
        print(f"=== 车道线存在性预测统计 ===")
        print(f"没有有效的预测数据")
        print(f"==========================\n")
    
    da_segment_result = (da_acc_seg.avg, da_IoU_seg.avg, da_mIoU_seg.avg)
    # 使用全局统计的mIOU
    global_all_miou = global_all_metric.meanIntersectionOverUnion()
    ll_segment_result = (ll_acc_seg.avg, ll_IoU_seg.avg, global_all_miou)
    return da_segment_result, ll_segment_result


def save_checkpoint(state, filenameCheckpoint='checkpoint.pth.tar'):
    torch.save(state, filenameCheckpoint)

def netParams(model):
    return np.sum([np.prod(parameter.size()) for parameter in model.parameters()])

def extract_navigation_line(drivable_mask, num_points=10):
    
    if drivable_mask.max() > 1:
        drivable_mask = drivable_mask / 255.0
    
    
    contours, _ = cv2.findContours(drivable_mask.astype(np.uint8), 
                                 cv2.RETR_EXTERNAL, 
                                 cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    
    max_contour = max(contours, key=cv2.contourArea)
    
    
    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    
    
    x, y, w, h = cv2.boundingRect(max_contour)
    
    
    points = []
    for i in range(num_points):
        
        y_coord = y + (i + 1) * h / (num_points + 1)
        
        row = drivable_mask[int(y_coord)]
        left = np.where(row > 0)[0][0] if len(np.where(row > 0)[0]) > 0 else 0
        right = np.where(row > 0)[0][-1] if len(np.where(row > 0)[0]) > 0 else len(row)
        
        x_coord = (left + right) / 2
        points.append([x_coord, y_coord])
    
    return np.array(points)

def visualize_navigation_line(image, drivable_mask, navigation_line, save_path=None):
    """
    可视化导航线
    
    参数:
        image: numpy.ndarray, 原始图像
        drivable_mask: numpy.ndarray, 可行驶区域的分割掩码
        navigation_line: numpy.ndarray, 导航线上的点坐标
        save_path: str, 保存路径（可选）
    """
    # 创建可视化图像
    vis_img = image.copy()
    
    # 绘制可行驶区域
    vis_img[drivable_mask > 0] = vis_img[drivable_mask > 0] * 0.7 + np.array([0, 255, 0]) * 0.3
    
    # 绘制导航线
    if navigation_line is not None:
        points = navigation_line.astype(np.int32)
        for i in range(len(points) - 1):
            cv2.line(vis_img, tuple(points[i]), tuple(points[i+1]), (0, 0, 255), 2)
        # 绘制导航点
        for point in points:
            cv2.circle(vis_img, tuple(point), 3, (255, 0, 0), -1)
    
    if save_path:
        cv2.imwrite(save_path, vis_img)
    
    return vis_img

def visualize_lane_mask(image, lane_mask, save_path=None):
    """
    可视化车道线掩码与原图叠加
    """
    vis_img = image.copy()
    # 叠加蓝色通道
    vis_img[lane_mask > 0] = vis_img[lane_mask > 0] * 0.7 + np.array([255, 0, 0]) * 0.3
    if save_path:
        cv2.imwrite(save_path, vis_img)
    return vis_img
