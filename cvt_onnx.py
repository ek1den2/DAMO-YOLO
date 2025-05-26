import torch
from damo.config.base import parse_config
from damo.detectors.detector import build_local_model
from damo.base_models.core.ops import RepConv, SiLU
from damo.utils.model_utils import get_model_info, replace_module
from time import time
import torch.nn as nn

import argparse

# オプションの確認
parser = argparse.ArgumentParser(description='DAMO-YOLOのwebカメラ推論')
parser.add_argument('--mode', type=str, default='cuda')
args = parser.parse_args()

device = args.mode

# モデルの準備
config = parse_config("configs/damoyolo_tinynasL20_T.py")
model = build_local_model(config, device)
ckpt = torch.load("checkpoints/damoyolo_tinynasL20_T.pth", map_location=device)
model.load_state_dict(ckpt['model'], strict=True)
model = replace_module(model, nn.SiLU, SiLU)
model.head.nms = False
for layer in model.modules():
    if isinstance(layer, RepConv):
        layer.switch_to_deploy()
model.eval()


# モデルの変換
dummy_input = torch.randn(1, 3, 640, 640).to(device)
torch.onnx.export(
    model,
    dummy_input,
    "./checkpoints/damo_yolo_T.onnx",
    input_names=["input"],
    output_names=["output_1", "output_2"],
    opset_version=17
)