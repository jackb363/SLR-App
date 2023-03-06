import numpy as np
import decord
import torch

from gluoncv.torch.utils.model_utils import download
from gluoncv.torch.data.transforms.videotransforms import video_transforms, volume_transforms
from gluoncv.torch.engine.config import get_cfg_defaults
from gluoncv.torch.model_zoo import get_model
from gluoncv.data import VideoClsCustom

config_file = 'i3d_resnet50_v1_custom.yaml'
cfg = get_cfg_defaults()
cfg.merge_from_file(config_file)
model = get_model(cfg)
model.eval()
print('%s model is successfully loaded.' % cfg.CONFIG.MODEL.NAME)