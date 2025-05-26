from damo.detectors.detector import build_local_model
import torch
from damo.base_models.core.ops import RepConv
from damo.config.base import parse_config
from time import time
import cv2
import numpy as np
from damo.utils import vis
import copy
import onnxruntime

import argparse

# オプションの確認
parser = argparse.ArgumentParser(description='DAMO-YOLOのwebカメラ推論')
parser.add_argument('--mode', type=str, default='cuda')
args = parser.parse_args()


# モデルの読み込み
# ONNXでGPUorCPU
if args.mode == 'cuda':
    prov = ['CUDAExecutionProvider']
elif args.mode == 'cpu':
    prov = ['CPUExecutionProvider']

model = onnxruntime.InferenceSession("./weights/damo_yolo_T.onnx", providers=prov)
cap = cv2.VideoCapture(0)
conf = 0.5
config = parse_config("./configs/damoyolo_tinynasL20_T.py")

# NMS部分
# 矩形aと複数の矩形bのIoUを計算
def iou_np(a, b, a_area, b_area):
    # aは1つの矩形を表すshape=(4,)のnumpy配列
    # array([xmin, ymin, xmax, ymax])
    # bは任意のN個の矩形を表すshape=(N, 4)のnumpy配列
    # 2次元目の4は、array([xmin, ymin, xmax, ymax])
    
    # a_areaは矩形aの面積
    # b_areaはbに含まれる矩形のそれぞれの面積
    # shape=(N,)のnumpy配列。Nは矩形の数
    
    # aとbの矩形の共通部分(intersection)の面積を計算するために、
    # N個のbについて、aとの共通部分のxmin, ymin, xmax, ymaxを一気に計算
    abx_mn = np.maximum(a[0], b[:,0]) # xmin
    aby_mn = np.maximum(a[1], b[:,1]) # ymin
    abx_mx = np.minimum(a[2], b[:,2]) # xmax
    aby_mx = np.minimum(a[3], b[:,3]) # ymax
    # 共通部分の幅を計算。共通部分が無ければ0
    w = np.maximum(0, abx_mx - abx_mn + 1)
    # 共通部分の高さを計算。共通部分が無ければ0
    h = np.maximum(0, aby_mx - aby_mn + 1)
    # 共通部分の面積を計算。共通部分が無ければ0
    intersect = w*h
    
    # N個のbについて、aとのIoUを一気に計算
    iou_np = intersect / (a_area + b_area - intersect)
    return iou_np


# NMSの計算
def nms_fast(bboxes, scores, classes, iou_threshold=0.5):
    # bboxesは任意のN個の矩形を格納したshape=(N, 4)のnumpy配列
    # 2次元目の4要素は、array([xmin, ymin, xmax, ymax])
    # scoresは任意のN個の信頼度を格納したshape=(N,)のnumpy配列
    # classesは任意のN個のクラスを格納したshape=(N,)のnumpy配列
    
    # bboxesの矩形の面積を一気に計算
    areas = (bboxes[:,2] - bboxes[:,0] + 1) \
             * (bboxes[:,3] - bboxes[:,1] + 1)
    
    # scoreの昇順(小さい順)の矩形インデックスのリストを取得
    sort_index = np.argsort(scores)
    
    i = -1 # 未処理の矩形のindex
    while(len(sort_index) >= 2 - i):
        # score最大のindexを取得
        max_scr_ind = sort_index[i]
        # score最大以外のindexを取得
        ind_list = sort_index[:i]
        # score最大の矩形それ以外の矩形のIoUを計算
        iou = iou_np(bboxes[max_scr_ind], bboxes[ind_list], \
                     areas[max_scr_ind], areas[ind_list])
        
        # IoUが閾値iou_threshold以上の矩形を計算
        del_index = np.where(iou >= iou_threshold)
        # IoUが閾値iou_threshold以上の矩形を削除
        sort_index = np.delete(sort_index, del_index)
        #print(len(sort_index), i, flush=True)
        i -= 1 # 未処理の矩形のindexを1減らす
    
    # bboxes, scores, classesから削除されなかった矩形のindexのみを抽出
    bboxes = bboxes[sort_index]
    scores = scores[sort_index]
    classes = classes[sort_index]
    
    return bboxes, scores, classes

person_cls_id = 0

# メイン部分
while True:   
    ret, frame = cap.read()
    image = cv2.resize(frame, (640, 640))
    # print(image.shape)
    original_img = copy.deepcopy(image)
    frame_in = image.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    # print(frame_in.shape)
    start = time()
    preds = model.run(None, {"input": frame_in})
    bboxes, scores = preds[1][0], preds[0][0]
    cls_inds, scores = np.argmax(scores, axis=1), np.max(scores, axis=1)

    mask = (cls_inds == person_cls_id) & (scores > conf)
    
    # 人間だけ
    bboxes = bboxes[mask]
    scores = scores[mask]
    cls_inds = cls_inds[mask]


    bboxes, scores, cls_inds = nms_fast(bboxes, scores, cls_inds)
    vis_img = vis(original_img, bboxes, scores, cls_inds, conf, config.dataset.class_names)
    end = time()
    cv2.putText(vis_img, f"time:{(end-start)*1000:4.3f}ms", (0, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_4)
    cv2.imshow("show", vis_img)
    cv2.waitKey(1)