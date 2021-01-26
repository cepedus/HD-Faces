import torch

configurations = {
    1: dict(
        SEED = 1337, # random seed for reproduce results
        
        DATA_ROOT = '../data', # the parent root where your train/val/test data are stored
        DATA_RECORD = '../data/highres_ids.txt', # the parent root where your train/val/test data are stored
        VAL_DATA_ROOT = '../data/hres_eval_pairs', # the parent root where your val/test data are stored
        # VAL_SET = 'lfw, cfp_fp, agedb_30, vgg2_fp', # validation set name
        VAL_SET = 'pairs_hres', # validation set name
        MODEL_ROOT = '../debug/model', # the root to buffer your checkpoints
        LOG_ROOT = '../debug/logs', # the root to log your train/val status
        IS_RESUME = False,
        BACKBONE_RESUME_ROOT = "",
        HEAD_RESUME_ROOT = "",

        VAL_IN_LOWRES = True, # Whether to train in 32x32 ou 144x144
        EVAL_FREQ = 2000, # when VAL_IN_LOWRES and highres training, put less than 1000


        # IMPORTANT: with VAL_IN_LOWRES, INPUT_SIZE is defined internally in train.py
        # INPUT_SIZE = [32, 32], # support: [32, 32] [112, 112], [144, 144] and [224, 224]
        
        BACKBONE_NAME = 'IR_SE_18', # support: ['MobileFaceNet', 'ResNet_50', 'ResNet_101', 'ResNet_152', 
                                #'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_100', 'IR_SE_101', 'IR_SE_152',
                                #'AttentionNet_IR_56', 'AttentionNet_IRSE_56','AttentionNet_IR_92', 'AttentionNet_IRSE_92',
                                #'ResNeSt_50', 'ResNeSt_101', 'ResNeSt_100']
        HEAD_NAME = "ArcFace", # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax', 'ArcNegFace', 'CurricularFace', 'SVX']
        LOSS_NAME = 'Softmax', # support: [''Softmax', Focal', 'HardMining', 'LabelSmooth', 'Softplus']
        
        
        RGB_MEAN = [0.5, 0.5, 0.5], # for normalize inputs to [-1, 1]
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 128, # feature dimension ## HARDCODED IN RESNET
        BATCH_SIZE = 10,
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        
        OPTIMIZER = 'sgd', # sgd, adam, lookahead, radam, ranger, adamp, sgdp
        LR = 0.1, # initial LR, use smaller lr for adam seris
        LR_SCHEDULER = 'cosine', # step/multi_step/cosine
        WARMUP_EPOCH = 0,
        WARMUP_LR = 0.0,
        START_EPOCH = 0, # start epoch
        NUM_EPOCH = 25, # total epoch number
        LR_STEP_SIZE = 10, # 'step' scheduler, period of learning rate decay. 
        LR_DECAY_EPOCH = [10, 18, 22], # ms1m epoch stages to decay learning rate
        LR_DECAT_GAMMA = 0.1, # multiplicative factor of learning rate decay
        LR_END = 1e-5, # minimum learning rate
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        
        WORLD_SIZE = 1,
        RANK = 0,
        GPU = [0], # specify your GPU id
        DIST_BACKEND = 'nccl', # 'nccl', 'gloo'
        DIST_URL = 'tcp://localhost:23456',
        NUM_WORKERS = 1,
        TEST_GPU_ID = [0],

        USE_APEX = False,
        SYNC_BN = False,

        # Data Augmentation
        RANDAUGMENT = False,
        RANDAUGMENT_N = 2, # random pick numer of aug typr form aug_list 
        RANDAUGMENT_M = 9,
        RANDOM_ERASING = False,
        MIXUP = False,
        MIXUP_ALPHA = 1.0,
        MIXUP_PROB = 0.5,
        CUTOUT = False, 
        CUTMIX = False, 
        CUTMIX_ALPHA = 1.0,
        CUTMIX_PROB = 0.5,
        COLORJITTER = False
    ),
}
