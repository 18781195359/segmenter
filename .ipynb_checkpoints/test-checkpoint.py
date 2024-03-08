import timm
import torch
import torch.nn as nn
import VIT
from timm.models.helpers import load_custom_pretrained
from timm.models.vision_transformer import default_cfgs
import json
from VIT import VisionTransformer
from torch.utils.data import DataLoader
from optim.Ranger import Ranger
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from tool.metrics import averageMeter,runningScore
from tool.utils import save_ckpt
from apex import amp
import logging
import os
import sys
import time
import shutil
from factory import get_model
from datasets_.irseg import IRSeg

def get_logger(logdir):

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logname = f'run-{time.strftime("%Y-%m-%d-%H-%M")}.log'
    log_file = os.path.join(logdir, logname)

    # create log
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)

    # Formatter 设置日志输出格式
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # StreamHandler 日志输出1 -> 到控制台
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # FileHandler 日志输出2 -> 保存到文件log_file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

class eeemodelLoss(nn.Module):
    def __init__(self):
        super(eeemodelLoss, self).__init__()
        self.class_weight_semantic = torch.from_numpy(np.array(
            [1.5105, 16.6591, 29.4238, 34.6315, 40.0845, 41.4357, 47.9794, 45.3725, 44.9000])).float()
        self.class_weight_binary = torch.from_numpy(np.array([1.5121, 10.2388])).float()

        self.binary_loss = nn.CrossEntropyLoss(weight=self.class_weight_binary)
        self.semantic_loss = nn.CrossEntropyLoss(weight=self.class_weight_semantic)

    def forward(self, inputs, targets):
        semantic_gt, binary_gt = targets
        semantic_out = inputs

        return self.semantic_loss(semantic_out, semantic_gt)*2

def get_dataset(cfg):
    assert cfg['dataset'] in ['nyuv2', 'nyuv2_new', 'sunrgbd', 'cityscapes', 'camvid', 'irseg', 'pst900', 'irseg_msv']

    if cfg['dataset'] == 'irseg':
        # return IRSeg(cfg, mode='trainval'), IRSeg(cfg, mode='test')
        return IRSeg(cfg, mode='train'), IRSeg(cfg, mode='val'), IRSeg(cfg, mode='test')
    elif cfg['dataset'] == 'pst900':
        from datasets_.pst900 import PSTSeg
        # return IRSeg(cfg, mode='trainval'), IRSeg(cfg, mode='test')
        return PSTSeg(cfg, mode='train'), PSTSeg(cfg, mode='val'), PSTSeg(cfg, mode='test')

def test():
    model_cfg = {
        "image_size": 384,
        "patch_size": 16,
        "num_classes": 1000,
        "dim": 768,
        "depth": 12,
        "heads": 12,
        "mlp_dim": 2048,
        "dropout": 0.1,
        "emb_dropout": 0.1,
    }
    v = ViT(
        # image_size = 256,
        # patch_size = 16,
        # num_classes = 1000,
        # dim = 1024,
        # depth = 24,
        # heads = 16,
        # mlp_dim = 2048,
        # dropout = 0.1,
        # emb_dropout = 0.1
        **model_cfg
    )

    print(timm.list_models())
    default_cfg = default_cfgs["vit_base_patch16_384"]
    default_cfg["input_size"] = (
        3, 384, 384
    )
    print(default_cfg)
    load_custom_pretrained(v, default_cfg)
    img = torch.randn(1, 3, 384, 384)

    preds = v(img)
    assert preds.shape == (1, 1000), 'correct logits outputted'

def test1():
    with open("configs/TwinViTSeg.json", 'r') as fp:
        cfg = json.load(fp)
    backbone = cfg.pop("backbone")
    default_cfg = default_cfgs[backbone]
    default_cfg["input_size"] = (
        3, 480, 640
    )
    print(default_cfg)

    cfg["image_size"] = list(cfg["image_size"].split(" "))
    cfg["image_size"][0] = int(cfg["image_size"][0])
    cfg["image_size"][1] = int(cfg["image_size"][1])

    model = VisionTransformer(**cfg)
    load_custom_pretrained(model, default_cfg)
    print(cfg)

def test2():
    device = torch.device('cuda:0')
    with open("configs/Data.json", 'r') as fp:
        cfg = json.load(fp)

    logdir = f'run/{time.strftime("%Y-%m-%d-%H-%M")}-{cfg["dataset"]}-{cfg["model_name"]}-'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    shutil.copy("configs/TwinViTSeg.json", logdir)

    logger = get_logger(logdir)
    logger.info(f'Conf | use logdir {logdir}')

    trainset, _, testset = get_dataset(cfg)
    train_loader = DataLoader(trainset, batch_size=cfg['ims_per_gpu'], shuffle=True, num_workers=cfg['num_workers'],
                              pin_memory=True)
    test_loader = DataLoader(testset, batch_size=cfg['ims_per_gpu'], shuffle=False, num_workers=cfg['num_workers'],
                             pin_memory=True)

    # with open("configs/TwinViTSeg.json", 'r') as fp:
    #     cfg_model = json.load(fp)
    # backbone = cfg_model.pop("backbone")
    # default_cfg = default_cfgs[backbone]
    # default_cfg["input_size"] = (
    #     3, 480, 640
    # )
    #
    # cfg_model["image_size"] = list(cfg_model["image_size"].split(" "))
    # cfg_model["image_size"][0] = int(cfg_model["image_size"][0])
    # cfg_model["image_size"][1] = int(cfg_model["image_size"][1])

    model = get_model()
    model.to(device)

    params_list = model.parameters()
    optimizer = Ranger(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay'])
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / cfg['epochs']) ** 0.9)

    train_criterion = eeemodelLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    train_loss_meter = averageMeter()
    test_loss_meter = averageMeter()
    running_metrics_test = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])
    best_test = 0

    amp.register_float_function(torch, 'sigmoid')
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')


    for ep in range(cfg['epochs']):

        # training
        model.train()
        train_loss_meter.reset()
        for i, sample in enumerate(train_loader):
            optimizer.zero_grad()
            image = sample['image'].to(device)
            #print("image shape: " + str(image.shape))
            depth = sample['depth'].to(device)
            #print("TIR shape: " + str(depth.shape))
            label = sample['label'].to(device)
            #print("label shape: " + str(label.shape))
            bound = sample['bound'].to(device)
            #print("bound shape: " + str(bound.shape))
            binary_label = sample['binary_label'].to(device)
            #print("binary_label shape: " + str(binary_label.shape))
            targets = [label, binary_label]
            predict = model(image, depth)

            loss = train_criterion(predict, targets)
            ####################################################

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            train_loss_meter.update(loss.item())

        scheduler.step(ep)

        # test
        with torch.no_grad():
            model.eval()
            running_metrics_test.reset()
            test_loss_meter.reset()
            for i, sample in enumerate(test_loader):

                image = sample['image'].to(device)
                # Here, depth is TIR.
                depth = sample['depth'].to(device)
                label = sample['label'].to(device)
                predict = model(image, depth)

                loss = criterion(predict, label)
                test_loss_meter.update(loss.item())

                predict = predict.max(1)[1].cpu().numpy()  # [1, h, w]
                label = label.cpu().numpy()
                running_metrics_test.update(label, predict)

        train_loss = train_loss_meter.avg
        test_loss = test_loss_meter.avg

        test_macc = running_metrics_test.get_scores()[0]["class_acc: "]
        test_miou = running_metrics_test.get_scores()[0]["mIou: "]
        test_avg = (test_macc + test_miou) / 2
        logger.info(
            f'Iter | [{ep + 1:3d}/{cfg["epochs"]}] loss={train_loss:.3f}/{test_loss:.3f}, mPA={test_macc:.3f}, miou={test_miou:.3f}, avg={test_avg:.3f}')
        if test_avg > best_test:
            best_test = test_avg
            save_ckpt(logdir, model,ep+1)
            logger.info(
            	f'Save Iter = [{ep + 1:3d}],  mPA={test_macc:.3f}, miou={test_miou:.3f}, avg={test_avg:.3f}')

test2()