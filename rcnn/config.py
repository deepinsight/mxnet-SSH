import numpy as np
from easydict import EasyDict as edict

config = edict()

# network related params
config.PIXEL_MEANS = np.array([103.939, 116.779, 123.68])
config.IMAGE_STRIDE = 0
config.RCNN_FEAT_STRIDE = 16

# dataset related params
config.NUM_CLASSES = 2
config.PRE_SCALES = [(1200, 1600)]  # first is scale (the shorter side); second is max size
config.SCALES = [(1200, 1600)]  # first is scale (the shorter side); second is max size
#config.SCALES = [(1200, 1600), (600, 1000)]  # first is scale (the shorter side); second is max size
config.SCALES = [(640, 640)]  # first is scale (the shorter side); second is max size
config.ORIGIN_SCALE = False
#config.ANCHOR_SCALES = (4, 8, 16, 32)
#config.ANCHOR_RATIOS = (1,)
#config.NUM_ANCHORS = len(config.ANCHOR_SCALES) * len(config.ANCHOR_RATIOS)
config.RPN_FEAT_STRIDE = [32,16,8]
config.ANCHOR_SCALES = [(32,16),(8,4),(2,1)]
config.ANCHOR_RATIOS = [(1,),(1,),(1,)]
config.NUM_ANCHORS = 2
#config.RPN_ANCHOR_CFG = {0: 1}

config.RPN_ANCHOR_CFG = {
    '32': {'SCALES': (32, 16), 'BASE_SIZE': 16, 'RATIOS': (1,), 'NUM_ANCHORS': 2, 'ALLOWED_BORDER': 512},
    '16': {'SCALES': (8, 4), 'BASE_SIZE': 16, 'RATIOS': (1,), 'NUM_ANCHORS': 2, 'ALLOWED_BORDER': 0},
    '8': {'SCALES': (2, 1), 'BASE_SIZE': 16, 'RATIOS': (1,), 'NUM_ANCHORS': 2, 'ALLOWED_BORDER': 0},
}

config.BBOX_MASK_THRESH = 0
#config.COLOR_JITTERING = 0.125
config.COLOR_JITTERING = 0

#config.RPN_FEAT_STRIDE = [32,]
#config.ANCHOR_SCALES = [(32,16),]
#config.ANCHOR_RATIOS = [(1,),]
#config.NUM_ANCHORS = 2

config.TRAIN = edict()

config.TRAIN.IMAGE_ALIGN = 0
config.TRAIN.MIN_BOX_SIZE = 6
# R-CNN and RPN
# size of images for each device, 2 for rcnn, 1 for rpn and e2e
config.TRAIN.BATCH_IMAGES = 8
# e2e changes behavior of anchor loader and metric
config.TRAIN.END2END = True
# group images with similar aspect ratio
config.TRAIN.ASPECT_GROUPING = True
config.USE_MAXOUT = False

# R-CNN
# rcnn rois batch size
config.TRAIN.BATCH_ROIS = 128
# rcnn rois sampling params
config.TRAIN.FG_FRACTION = 0.25
config.TRAIN.FG_THRESH = 0.5
config.TRAIN.BG_THRESH_HI = 0.3
config.TRAIN.BG_THRESH_LO = 0.0
# rcnn bounding box regression params
config.TRAIN.BBOX_REGRESSION_THRESH = 0.5
config.TRAIN.BBOX_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0])

# RPN anchor loader
# rpn anchors batch size
config.TRAIN.RPN_ENABLE_OHEM = 2
config.TRAIN.RPN_BATCH_SIZE = 256
# rpn anchors sampling params
config.TRAIN.RPN_FG_FRACTION = 0.25
config.TRAIN.RPN_POSITIVE_OVERLAP = 0.5
config.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
config.TRAIN.RPN_CLOBBER_POSITIVES = False
config.TRAIN.RPN_FORCE_POSITIVE = False
# rpn bounding box regression params
config.TRAIN.RPN_BBOX_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
config.TRAIN.RPN_POSITIVE_WEIGHT = -1.0

# used for end2end training
# RPN proposal
config.TRAIN.CXX_PROPOSAL = True
config.TRAIN.RPN_NMS_THRESH = 0.7
config.TRAIN.RPN_PRE_NMS_TOP_N = 12000
config.TRAIN.RPN_POST_NMS_TOP_N = 2000
config.TRAIN.RPN_MIN_SIZE = config.RPN_FEAT_STRIDE
# approximate bounding box regression
config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED = True
config.TRAIN.BBOX_MEANS = (0.0, 0.0, 0.0, 0.0)
config.TRAIN.BBOX_STDS = (0.1, 0.1, 0.2, 0.2)

config.TEST = edict()

# R-CNN testing
# use rpn to generate proposal
config.TEST.HAS_RPN = False
# size of images for each device
config.TEST.BATCH_IMAGES = 1

# RPN proposal
config.TEST.CXX_PROPOSAL = True
config.TEST.RPN_NMS_THRESH = 0.3
config.TEST.RPN_PRE_NMS_TOP_N = 1000
config.TEST.RPN_POST_NMS_TOP_N = 3000
config.TEST.RPN_MIN_SIZE = config.RPN_FEAT_STRIDE
config.TEST.RPN_MIN_SIZE = [0,0,0]

# RPN generate proposal
config.TEST.PROPOSAL_NMS_THRESH = 0.3
config.TEST.PROPOSAL_PRE_NMS_TOP_N = 1000 
config.TEST.PROPOSAL_POST_NMS_TOP_N = 3000
config.TEST.PROPOSAL_MIN_SIZE = config.RPN_FEAT_STRIDE
config.TEST.PROPOSAL_MIN_SIZE = [0,0,0]

# RCNN nms
config.TEST.NMS = 0.3

config.TEST.SCORE_THRESH = 0.05
config.TEST.IOU_THRESH = 0.5


# network settings
network = edict()

network.vgg = edict()
network.ssh = edict()

# dataset settings
dataset = edict()


dataset.widerface = edict()
dataset.widerface.dataset = 'widerface'
dataset.widerface.image_set = 'train'
dataset.widerface.test_image_set = 'val'
dataset.widerface.root_path = 'data'
dataset.widerface.dataset_path = 'data/widerface'
dataset.widerface.NUM_CLASSES = 2


# default settings
default = edict()

config.FIXED_PARAMS = ['^conv1', '^conv2', '^conv3', '^.*upsampling']
#config.FIXED_PARAMS = ['^conv0', '^stage1', 'gamma', 'beta']  #for resnet

# default network
default.network = 'ssh'
default.pretrained = 'model/vgg16'
#default.network = 'resnetssh'

default.pretrained_epoch = 0
default.base_lr = 0.004
# default dataset
default.dataset = 'widerface'
default.image_set = 'train'
default.test_image_set = 'val'
default.root_path = 'data/widerface'
default.dataset_path = 'data/widerface/data'
# default training
default.frequent = 20
default.kvstore = 'device'
# default e2e
default.e2e_prefix = 'model/e2e'
default.e2e_epoch = 10000
default.e2e_lr = default.base_lr
default.e2e_lr_step = '55,68,80'
# default rpn
default.rpn_prefix = 'model/rpn'
default.rpn_epoch = 4
default.rpn_lr = default.base_lr
default.rpn_lr_step = '2'
# default rcnn
default.rcnn_prefix = 'model/rcnn'
default.rcnn_epoch = 4
default.rcnn_lr = default.base_lr
default.rcnn_lr_step = '2'

def generate_config(_network, _dataset):
    for k, v in network[_network].items():
        if k in config:
            config[k] = v
        elif k in default:
            default[k] = v
    for k, v in dataset[_dataset].items():
        if k in config:
            config[k] = v
        elif k in default:
            default[k] = v

