from ViTFusionRGB_TNet import TwinVitFusion
from timm.models.vision_transformer import default_cfgs
from VIT import VisionTransformer
import json
from timm.models.helpers import load_custom_pretrained
from decoder import pixel_decoder

def creat_vit():
    with open("configs/TwinViTSeg.json", 'r') as fp:
        cfg_model = json.load(fp)
    backbone = cfg_model.pop("backbone")
    default_cfg = default_cfgs[backbone]
    default_cfg["input_size"] = (
        3, 480, 640
    )
    cfg_model["image_size"] = list(cfg_model["image_size"].split(" "))
    cfg_model["image_size"][0] = int(cfg_model["image_size"][0])
    cfg_model["image_size"][1] = int(cfg_model["image_size"][1])

    model = VisionTransformer(**cfg_model)
    print(111)
    load_custom_pretrained(model, default_cfg)
    print(222)
    return model

def create_decoder():
    with open("configs/TwinViTSeg.json", 'r') as fp:
        cfg_model = json.load(fp)
    backbone = cfg_model.pop("backbone")
    decoder = pixel_decoder(**cfg_model)
    return decoder

def get_model():
    vit1_encoder = creat_vit()
    vit2_encoder = creat_vit()
    decoder = create_decoder()
    fusion_net = TwinVitFusion(vit1_encoder, vit2_encoder, decoder)
    return fusion_net