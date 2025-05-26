from damo.detectors.detector import build_local_model
import torch
from damo.base_models.core.ops import RepConv
from damo.config.base import parse_config
from time import time
import cv2
import numpy as np
from damo.utils import vis
import copy

import argparse

# オプションの確認
parser = argparse.ArgumentParser(description='DAMO-YOLOのwebカメラ推論')
parser.add_argument('--mode', type=str, default='cuda')
args = parser.parse_args()

device = args.mode

# モデルの準備
config = parse_config("configs/damoyolo_tinynasL20_T.py")
model = build_local_model(config, device)
ckpt = torch.load("weights/damoyolo_tinynasL20_T.pth", map_location=device)
model.load_state_dict(ckpt['model'], strict=True)
for layer in model.modules():
    if isinstance(layer, RepConv):
        layer.switch_to_deploy()
model.eval()

# 後処理
def  postprocess(preds):
    output = preds
    output = output[0]
    bboxes = output.bbox
    scores = output.get_field('scores')
    cls_inds = output.get_field('labels')

    return bboxes, scores, cls_inds

cap = cv2.VideoCapture(0)
conf = 0.5

# メイン動作部分
while True:   
    with torch.no_grad():
        ret, frame = cap.read()
        image = cv2.resize(frame, (640, 640))
        # print(image.shape)
        original_img = copy.deepcopy(image)
        frame_in = image.transpose(2, 0, 1)[np.newaxis] # [640, 640, 3] -> [1, 3, 640, 640]
        # print(frame_in.shape)
        frame_in = torch.from_numpy(frame_in).to(torch.float32).to(device)
        start = time()
        preds = model(frame_in)
        bboxes, scores, cls_inds = postprocess(preds)
        # print("bbox:", bboxes)
        # print("scores:", scores)
        # print("indx:", cls_inds)
        vis_img = vis(original_img, bboxes, scores, cls_inds, conf, config.dataset.class_names)
        end = time()
        cv2.putText(vis_img, f"time:{(end-start)*1000:4.3f}ms", (0, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_4)
        cv2.imshow("show", vis_img)
        cv2.waitKey(1)
