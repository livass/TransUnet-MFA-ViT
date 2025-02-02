from easydict import EasyDict

cfg = EasyDict()
cfg.batch_size = 32
cfg.epoch = 1
cfg.learning_rate = 1e-3
cfg.momentum = 0.9
cfg.weight_decay = 1e-4
cfg.patience = 25
cfg.inference_threshold = 0.75

cfg.transunet = EasyDict()
cfg.transunet.img_dim = 128
cfg.transunet.in_channels = 3
cfg.transunet.out_channels = 128
cfg.transunet.head_num = 4
cfg.transunet.mlp_dim = 128
cfg.transunet.block_num = 8
cfg.transunet.patch_dim = 16
cfg.transunet.class_num = 1
