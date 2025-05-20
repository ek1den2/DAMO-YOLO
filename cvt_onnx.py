import torch
from damo.config.base import parse_config
from damo.detectors.detector import build_local_model
from damo.base_models.core.ops import RepConv, SiLU
from damo.utils.model_utils import get_model_info, replace_module
from time import time
import torch.nn as nn

# モデルの準備
config = parse_config("configs/damoyolo_tinynasL20_T.py")
model = build_local_model(config, "cuda")
ckpt = torch.load("weights/damoyolo_tinynasL20_T.pth", map_location="cuda")
model.load_state_dict(ckpt['model'], strict=True)
model = replace_module(model, nn.SiLU, SiLU)
model.head.nms = False
for layer in model.modules():
    if isinstance(layer, RepConv):
        layer.switch_to_deploy()
model.eval()


# モデルの変換
dummy_input = torch.randn(1, 3, 640, 640).to('cuda')
torch.onnx.export(
    model,
    dummy_input,
    "damo_yolo_T.onnx",
    input_names=["input"],
    output_names=["output_1", "output_2"],
    opset_version=17
)