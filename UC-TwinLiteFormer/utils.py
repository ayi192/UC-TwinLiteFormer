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
        if args.onGPU == True:
            input = input.cuda().float() / 255.0        
        output = model(input)
        with torch.cuda.amp.autocast():
            focal_loss,tversky_loss,dice_loss,loss = criterion(output,target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)
        if verbose:
            pbar.set_description(('%13s' * 1 + '%13.4g' * 4) %
                                     (f'{epoch}/{300 - 1}', tversky_loss, focal_loss, dice_loss, loss.item()))
    return ema if ema is not None else None


def extract_navigation_line_triangle(road_mask):
    """
    使用最小三角形拟合方法提取导航线
    
    参数:
        road_mask: numpy.ndarray, 道路分割掩码 (0-1)
    
    返回:
        navigation_line: numpy.ndarray, 导航线的起点和终点坐标 [[x1,y1], [x2,y2]]
    """
    # 确保输入是二值图像
    if road_mask.max() > 1:
        road_mask = road_mask / 255.0
    # 获取道路区域的轮廓
    contours, _ = cv2.findContours(road_mask.astype(np.uint8), 
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

def visualize_navigation_line_triangle(image, road_mask, navigation_line, save_path=None):
    """
    可视化基于三角形拟合的导航线
    
    参数:
        image: numpy.ndarray, 原始图像
        road_mask: numpy.ndarray, 道路分割掩码
        navigation_line: numpy.ndarray, 导航线的起点和终点坐标 [[x1,y1], [x2,y2]]
        save_path: str, 保存路径（可选）
    """
    # 创建可视化图像
    vis_img = image.copy()
    
    # 绘制道路区域
    vis_img[road_mask > 0] = vis_img[road_mask > 0] * 0.7 + np.array([0, 255, 0]) * 0.3
    
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
def extract_ob_endpoints_center(ob_mask, N=5):
    """
    提取障碍物掩码的两个端点（最上和最下N行的x中值/均值）
    ob_mask: numpy.ndarray, 0/1 掩码
    返回: [[x1, y1], [x2, y2]]，分别为上端点和下端点
    """
    if ob_mask.max() > 1:
        ob_mask = ob_mask / 255.0
    ob_mask = (ob_mask > 0.5).astype(np.uint8)
    points = np.column_stack(np.where(ob_mask > 0))
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

def visualize_ob_endpoints(image, ob_mask, endpoints, gt_endpoints=None, save_path=None, y1_ratio=0.85, y2_ratio=0.7):
    vis_img = image.copy()
    H, W = vis_img.shape[:2]
    vis_img[ob_mask > 0] = vis_img[ob_mask > 0] * 0.7 + np.array([255, 0, 0]) * 0.3
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
    if args is None or not hasattr(args, 'vis'):
        args.vis = False
    print(f"args: {args}")
    print(f"Visualization Enabled: {args.vis}")
    if args.vis == True:
        print(f"可视化已经开启")
    model.eval()
    ROAD = SegmentationMetric(2)  # 道路：2类
    OB = SegmentationMetric(4)  # 障碍物：4类（背景、障碍物1、石头、沟壑）
    road_acc_seg = AverageMeter()
    road_IoU_seg = AverageMeter()
    road_mIoU_seg = AverageMeter()
    ob_acc_seg = AverageMeter()
    ob_IoU_seg = AverageMeter()
    ob_mIoU_seg = AverageMeter()
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
    all_X1_road = []
    all_X2_road = []
    all_X1_ob = []
    all_X2_ob = []
    all_angle_err_fused = []
    all_angle_err_road = []
    all_angle_err_ob = []
    
    # 添加数据统计
    total_samples = 0
    samples_with_ob = 0
    total_ob_pixels = 0
    total_stone_pixels = 0  # 添加石头像素统计
    total_gully_pixels = 0  # 添加沟壑像素统计
    total_pixels = 0
    
    # 添加障碍物IOU统计
    ob_iou_list = []
    ob_miou_list = []
    stone_iou_list = []
    stone_miou_list = []
    gully_iou_list = []
    gully_miou_list = []
    
    # 添加全局像素统计
    global_ob_metric = SegmentationMetric(4)  # 障碍物全局统计
    global_stone_metric = SegmentationMetric(4)  # 石头全局统计
    global_gully_metric = SegmentationMetric(4)  # 沟壑全局统计
    global_all_metric = SegmentationMetric(4)    # 所有样本全局统计
    
    # 添加障碍物存在性预测统计
    ob_existence_tp = 0  # 真正例：预测有障碍物，实际有障碍物
    ob_existence_fp = 0  # 假正例：预测有障碍物，实际无障碍物
    ob_existence_tn = 0  # 真负例：预测无障碍物，实际无障碍物
    ob_existence_fn = 0  # 假负例：预测无障碍物，实际有障碍物
    
    for i, (img_names, input, target) in pbar:
        input = input.cuda().half() / 255.0 if half else input.cuda().float() / 255.0
        input_var = input
        target_var = target
        
        # 统计障碍物数据分布
        target_ob = target[1]
        _, ob_gt = torch.max(target_ob, 1)
        
        # 统计三类别数据
        batch_samples_with_ob = ((ob_gt == 1).sum(dim=(1,2)) > 0).sum().item()  # 障碍物1
        batch_samples_with_stone = ((ob_gt == 2).sum(dim=(1,2)) > 0).sum().item()  # 石头
        batch_samples_with_gully = ((ob_gt == 3).sum(dim=(1,2)) > 0).sum().item()  # 沟壑
        batch_samples_with_any = batch_samples_with_ob + batch_samples_with_stone + batch_samples_with_gully
        
        samples_with_ob += batch_samples_with_any  # 包含障碍物、石头或沟壑的样本
        total_samples += input.size(0)
        total_ob_pixels += (ob_gt == 1).sum().item()  # 障碍物像素
        total_stone_pixels += (ob_gt == 2).sum().item()  # 石头像素
        total_gully_pixels += (ob_gt == 3).sum().item()  # 沟壑像素
        total_pixels += ob_gt.numel()
        
        with torch.no_grad():
            output = model(input_var)
        out_road = output[0]
        target_road = target[0]
        _, road_predict = torch.max(out_road, 1)
        road_predict_cropped = road_predict[:, 12:-12]
        _, road_gt = torch.max(target_road, 1)
        ROAD.reset()
        ROAD.addBatch(road_predict_cropped.cpu(), road_gt.cpu())
        road_acc = ROAD.pixelAccuracy()
        road_IoU = ROAD.IntersectionOverUnion()
        road_mIoU = ROAD.meanIntersectionOverUnion()
        road_acc_seg.update(road_acc, input.size(0))
        road_IoU_seg.update(road_IoU, input.size(0))
        road_mIoU_seg.update(road_mIoU, input.size(0))
        out_ob = output[1]
        target_ob = target[1]
        _, ob_predict = torch.max(out_ob, 1)
        ob_predict_cropped = ob_predict[:, 12:-12]
        _, ob_gt = torch.max(target_ob, 1)
        
        # 统计障碍物存在性预测
        for j in range(input.size(0)):
            # 获取真实标签
            gt_label = ob_gt[j]  # [H, W]
            
            # 检查是否有障碍物或石头
            gt_has_ob = (gt_label == 1).sum() > 0  # 障碍物类别
            gt_has_stone = (gt_label == 2).sum() > 0  # 石头类别
            gt_has_gully = (gt_label == 3).sum() > 0  # 沟壑类别
            gt_has_any = gt_has_ob or gt_has_stone or gt_has_gully  # 有任何目标
            
            # 获取预测标签
            pred_label = ob_predict_cropped[j]  # [H, W]
            pred_has_ob = (pred_label == 1).sum() > 0
            pred_has_stone = (pred_label == 2).sum() > 0
            pred_has_gully = (pred_label == 3).sum() > 0
            pred_has_any = pred_has_ob or pred_has_stone or pred_has_gully
            
            # 统计障碍物预测
            if gt_has_ob and pred_has_ob:
                ob_existence_tp += 1
            elif gt_has_ob and not pred_has_ob:
                ob_existence_fn += 1
            elif not gt_has_ob and pred_has_ob:
                ob_existence_fp += 1
            elif not gt_has_ob and not pred_has_ob:
                ob_existence_tn += 1
        
        # 使用全局像素统计法计算障碍物和石头的IOU
        for j in range(input.size(0)):
            gt_label = ob_gt[j]  # [H, W]
            pred_label = ob_predict_cropped[j]  # [H, W]
            
            # 为所有样本添加全局统计
            global_all_metric.addBatch(pred_label.cpu().unsqueeze(0), gt_label.cpu().unsqueeze(0))
            
            # 检查是否有障碍物
            if (gt_label == 1).sum() > 0:  # 如果有障碍物
                ob_iou_list.append(1)  # 记录有障碍物的样本数量
                # 添加到全局统计
                global_ob_metric.addBatch(pred_label.cpu().unsqueeze(0), gt_label.cpu().unsqueeze(0))
            
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
        OB.reset()
        OB.addBatch(ob_predict_cropped.cpu(), ob_gt.cpu())
        ob_acc = OB.lineAccuracy()
        ob_IoU = OB.IntersectionOverUnion()
        ob_mIoU = OB.meanIntersectionOverUnion()
        ob_acc_seg.update(ob_acc, input.size(0))
        ob_IoU_seg.update(ob_IoU[1], input.size(0))  # 使用障碍物类别的IOU
        ob_mIoU_seg.update(ob_mIoU, input.size(0))
        if args.vis:
            _, road_predict_full = torch.max(out_road, 1)
            _, ob_predict_full = torch.max(out_ob, 1)
            for j in range(input_var.size(0)):
                orig_img_tensor = input_var[j].detach().cpu()
                orig_img = (orig_img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                road_mask = road_predict_full[j].detach().cpu().numpy().astype(np.uint8)
                ob_mask = ob_predict_full[j].detach().cpu().numpy().astype(np.uint8)
                road_mask_cropped = road_mask[12:-12, :]
                ob_mask_cropped = ob_mask[12:-12, :]
                if getattr(args, 'save_mask', True):
                    if args.save_mask_max == -1 or total_mask_saved < args.save_mask_max:
                        base_name = os.path.basename(img_names[j])
                        mask_filename = f"{os.path.splitext(base_name)[0]}_mask.png"
                        mask_save_path = os.path.join(args.save_mask_dir, mask_filename)
                        print(f"保存掩码到: {mask_save_path}")
                        print(f"掩码形状: {ob_mask_cropped.shape}, 值范围: [{ob_mask_cropped.min()}, {ob_mask_cropped.max()}]")
                        os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
                        cv2.imwrite(mask_save_path, (ob_mask_cropped * 255).astype(np.uint8))
                        total_mask_saved += 1
                        print(f"掩码保存完成 ({total_mask_saved})")
                if args.vis_max == -1 or total_vis_saved < args.vis_max:
                    navigation_line = extract_navigation_line_triangle(road_mask_cropped)
                    orig_img_cropped = orig_img[12:-12, :, :]
                    vis_img = visualize_navigation_line_triangle(orig_img_cropped, road_mask_cropped, navigation_line)
                    img_name = os.path.basename(img_names[j])
                    cv2.putText(vis_img, img_name, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    save_name = f"{os.path.splitext(img_name)[0]}_nav.png"
                    save_path = os.path.join(args.save_vis_dir, save_name)
                    cv2.imwrite(save_path, vis_img)
                    ob_vis_img = visualize_ob_mask(orig_img_cropped, ob_mask_cropped)
                    ob_endpoints_geom = extract_ob_endpoints_center(ob_mask_cropped, N=5)
                    if ob_endpoints_geom is not None:
                        pt1, pt2 = ob_endpoints_geom.astype(np.int32)
                        cv2.line(ob_vis_img, tuple(pt1), tuple(pt2), (0, 0, 255), 2)
                        cv2.circle(ob_vis_img, tuple(pt1), 7, (0, 0, 255), -1)
                        cv2.circle(ob_vis_img, tuple(pt2), 7, (0, 0, 255), -1)
                    ob_save_name = f"{os.path.splitext(img_name)[0]}_ob.png"
                    ob_save_path = os.path.join(args.save_vis_dir, ob_save_name)
                    cv2.imwrite(ob_save_path, ob_vis_img)
                    gt_mask_path = img_names[j].replace("images", "lane_line_annotations").replace("jpg", "png")
                    gt_mask = cv2.imread(gt_mask_path, 0)
                    gt_mask = cv2.resize(gt_mask, (ob_mask_cropped.shape[1], ob_mask_cropped.shape[0]))
                    gt_line = extract_ob_endpoints_center(gt_mask, N=5)
                    road_line = extract_navigation_line_triangle(road_mask_cropped)
                    ob_line = extract_ob_endpoints_center(ob_mask_cropped, N=5)
                    conf1 = getattr(args, 'road_segment_results', [0,0,0.9])[2]
                    conf2 = getattr(args, 'ob_segment_results', [0,0,0.5])[1]
                    conf1_norm = (conf1 - 0.8) / (1.0 - 0.8)
                    conf2_norm = (conf2 - 0.1) / (0.4 - 0.1)
                    conf1_norm = np.clip(conf1_norm, 0, 1)
                    conf2_norm = np.clip(conf2_norm, 0, 1)
                    if road_line is not None and ob_line is not None:
                        road_line = sort_line_by_y(road_line)
                        ob_line = sort_line_by_y(ob_line)
                        alpha = 0.5
                        beta = 0.5
                        fused_line = alpha * road_line + beta * ob_line
                    else:
                        fused_line = None
                    ob_endpoints_vis = visualize_ob_endpoints(
                        orig_img_cropped, ob_mask_cropped, fused_line, gt_endpoints=gt_line)
                    ob_endpoints_save_name = f"{os.path.splitext(img_name)[0]}_ob_endpoints.png"
                    ob_endpoints_save_path = os.path.join(args.save_vis_dir, ob_endpoints_save_name)
                    cv2.imwrite(ob_endpoints_save_path, ob_endpoints_vis)
                    total_vis_saved += 1
                road_line = extract_navigation_line_triangle(road_mask_cropped)
                ob_line = extract_ob_endpoints_center(ob_mask_cropped, N=5)
                gt_mask_path = img_names[j].replace("images", "lane_line_annotations").replace("jpg", "png")
                gt_mask = cv2.imread(gt_mask_path, 0)
                gt_mask = cv2.resize(gt_mask, (ob_mask_cropped.shape[1], ob_mask_cropped.shape[0]))
                gt_line = extract_ob_endpoints_center(gt_mask, N=5)
                conf1 = getattr(args, 'road_segment_results', [0,0,0.9])[2]
                conf2 = getattr(args, 'ob_segment_results', [0,0,0.5])[1]
                conf1_norm = (conf1 - 0.8) / (1.0 - 0.8)
                conf2_norm = (conf2 - 0.1) / (0.4 - 0.1)
                conf1_norm = np.clip(conf1_norm, 0, 1)
                conf2_norm = np.clip(conf2_norm, 0, 1)
                if road_line is not None and ob_line is not None and gt_line is not None:
                    road_line = sort_line_by_y(road_line)
                    ob_line = sort_line_by_y(ob_line)
                    alpha = 0.5
                    beta = 0.5
                    fused_line = alpha * road_line + beta * ob_line
                    H = ob_mask_cropped.shape[0]
                    X1, X2 = compare_lines(fused_line, gt_line, H)
                    all_X1.append(X1)
                    all_X2.append(X2)
                    X1_road, X2_road = compare_lines(road_line, gt_line, H)
                    all_X1_road.append(X1_road)
                    all_X2_road.append(X2_road)
                    X1_ob, X2_ob = compare_lines(ob_line, gt_line, H)
                    all_X1_ob.append(X1_ob)
                    all_X2_ob.append(X2_ob)
                    angle_fused = heading_angle(fused_line)
                    angle_gt = heading_angle(gt_line)
                    angle_road = heading_angle(road_line)
                    angle_ob = heading_angle(ob_line)
                    err_fused = abs(angle_fused - angle_gt)
                    err_fused = min(err_fused, 360 - err_fused)
                    all_angle_err_fused.append(err_fused)
                    err_road = abs(angle_road - angle_gt)
                    err_road = min(err_road, 360 - err_road)
                    all_angle_err_road.append(err_road)
                    err_ob = abs(angle_ob - angle_gt)
                    err_ob = min(err_ob, 360 - err_ob)
                    all_angle_err_ob.append(err_ob)
    if len(all_X1) > 0 and len(all_X2) > 0:
        mean_X1 = sum(all_X1) / len(all_X1)
        mean_X2 = sum(all_X2) / len(all_X2)
        print(f"融合导航线 val集平均像素偏差：X1={mean_X1:.2f}, X2={mean_X2:.2f}")
    if len(all_X1_road) > 0 and len(all_X2_road) > 0:
        mean_X1_road = sum(all_X1_road) / len(all_X1_road)
        mean_X2_road = sum(all_X2_road) / len(all_X2_road)
        print(f"道路导航线 val集平均像素偏差：X1={mean_X1_road:.2f}, X2={mean_X2_road:.2f}")
    if len(all_X1_ob) > 0 and len(all_X2_ob) > 0:
        mean_X1_ob = sum(all_X1_ob) / len(all_X1_ob)
        mean_X2_ob = sum(all_X2_ob) / len(all_X2_ob)
        print(f"直接预测导航线 val集平均像素偏差：X1={mean_X1_ob:.2f}, X2={mean_X2_ob:.2f}")
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
    print_angle_metrics('道路导航线', all_angle_err_road)
    print_angle_metrics('直接预测导航线', all_angle_err_ob)
    
    # 打印数据分布统计
    print(f"\n=== 数据分布统计 ===")
    print(f"总样本数: {total_samples}")
    print(f"包含障碍物、石头或沟壑的样本数: {samples_with_ob}")
    print(f"障碍物、石头或沟壑样本占比: {samples_with_ob/total_samples*100:.2f}%")
    print(f"障碍物像素占比: {total_ob_pixels/total_pixels*100:.4f}%")
    print(f"石头像素占比: {total_stone_pixels/total_pixels*100:.4f}%")
    print(f"沟壑像素占比: {total_gully_pixels/total_pixels*100:.4f}%")
    print(f"目标像素总占比: {(total_ob_pixels+total_stone_pixels+total_gully_pixels)/total_pixels*100:.4f}%")
    print(f"==================\n")
    
    # 打印障碍物IOU统计
    print(f"=== 障碍物分割统计 (全局像素统计) ===")
    if len(ob_iou_list) > 0:
        # 使用全局统计计算IOU
        global_all_iou = global_all_metric.IntersectionOverUnion()
        global_all_miou = global_all_metric.meanIntersectionOverUnion()
        print(f"障碍物样本数: {len(ob_iou_list)}")
        print(f"障碍物全局IOU: {global_all_iou[1]:.3f}")  # 障碍物类别的全局IOU
        print(f"障碍物全局mIOU: {global_all_miou:.3f}")
    else:
        print(f"没有找到包含障碍物的样本")
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
    print(f"障碍物全局IOU: {global_all_iou[1]:.3f}")
    print(f"石头全局IOU: {global_all_iou[2]:.3f}")
    print(f"沟壑全局IOU: {global_all_iou[3]:.3f}")
    print(f"整体全局mIOU: {global_all_miou:.3f}")
    print(f"========================\n")
    
    # 打印障碍物存在性预测统计
    total_existence = ob_existence_tp + ob_existence_fp + ob_existence_tn + ob_existence_fn
    if total_existence > 0:
        accuracy = (ob_existence_tp + ob_existence_tn) / total_existence
        precision = ob_existence_tp / (ob_existence_tp + ob_existence_fp) if (ob_existence_tp + ob_existence_fp) > 0 else 0
        recall = ob_existence_tp / (ob_existence_tp + ob_existence_fn) if (ob_existence_tp + ob_existence_fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"=== 障碍物存在性预测统计 ===")
        print(f"总样本数: {total_existence}")
        print(f"真正例(TP): {ob_existence_tp} - 预测有障碍物，实际有障碍物")
        print(f"假正例(FP): {ob_existence_fp} - 预测有障碍物，实际无障碍物")
        print(f"真负例(TN): {ob_existence_tn} - 预测无障碍物，实际无障碍物")
        print(f"假负例(FN): {ob_existence_fn} - 预测无障碍物，实际有障碍物")
        print(f"准确率(Accuracy): {accuracy:.3f}")
        print(f"精确率(Precision): {precision:.3f}")
        print(f"召回率(Recall): {recall:.3f}")
        print(f"F1分数: {f1_score:.3f}")
        print(f"==========================\n")
    else:
        print(f"=== 障碍物存在性预测统计 ===")
        print(f"没有有效的预测数据")
        print(f"==========================\n")
    
    road_segment_result = (road_acc_seg.avg, road_IoU_seg.avg, road_mIoU_seg.avg)
    # 使用全局统计的mIOU
    global_all_miou = global_all_metric.meanIntersectionOverUnion()
    ob_segment_result = (ob_acc_seg.avg, ob_IoU_seg.avg, global_all_miou)
    return road_segment_result, ob_segment_result


def save_checkpoint(state, filenameCheckpoint='checkpoint.pth.tar'):
    torch.save(state, filenameCheckpoint)

def netParams(model):
    return np.sum([np.prod(parameter.size()) for parameter in model.parameters()])

def extract_navigation_line(road_mask, num_points=10):
    
    if road_mask.max() > 1:
        road_mask = road_mask / 255.0
    
    
    contours, _ = cv2.findContours(road_mask.astype(np.uint8), 
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
        
        row = road_mask[int(y_coord)]
        left = np.where(row > 0)[0][0] if len(np.where(row > 0)[0]) > 0 else 0
        right = np.where(row > 0)[0][-1] if len(np.where(row > 0)[0]) > 0 else len(row)
        
        x_coord = (left + right) / 2
        points.append([x_coord, y_coord])
    
    return np.array(points)

def visualize_navigation_line(image, road_mask, navigation_line, save_path=None):
    """
    可视化导航线
    
    参数:
        image: numpy.ndarray, 原始图像
        road_mask: numpy.ndarray, 道路分割掩码
        navigation_line: numpy.ndarray, 导航线上的点坐标
        save_path: str, 保存路径（可选）
    """
    # 创建可视化图像
    vis_img = image.copy()
    
    # 绘制道路区域
    vis_img[road_mask > 0] = vis_img[road_mask > 0] * 0.7 + np.array([0, 255, 0]) * 0.3
    
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

def visualize_ob_mask(image, ob_mask, save_path=None):
    """
    可视化障碍物掩码与原图叠加
    """
    vis_img = image.copy()
    # 叠加蓝色通道
    vis_img[ob_mask > 0] = vis_img[ob_mask > 0] * 0.7 + np.array([255, 0, 0]) * 0.3
    if save_path:
        cv2.imwrite(save_path, vis_img)
    return vis_img
