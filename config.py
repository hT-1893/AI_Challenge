import argparse

def get_args(auto=True):
    args = argparse.Namespace()

    args.IMG_TRAIN_DIR = 'data/train/'
    args.IMG_VAL_DIR = 'data/val/'
    args.IMG_TEST_DIR = 'data/test/'
    args.n_classes = 117

    args.train_batch_size = 96
    args.val_batch_size = 48
    args.image_size = 224
    args.seed = 1

    args.min_scale = 0.8
    args.max_scale = 1.0
    args.random_horiz_flip = 0.5
    args.jitter = 0.
    args.tile_random_grayscale = 0.1

    args.limit_source = None
    args.limit_target = None
    args.learning_rate = 0.001
    args.epochs = 30
    args.network = 'efficientnet_b0'
    args.tf_logger = True
    args.val_size = 0.1
    args.folder_name = 'test'
    args.bias_whole_image = 0.9
    args.TTA = False
    args.classify_only_sane = False
    args.train_all = True
    args.suffix = ""
    args.nesterov = False
    args.visualization = False
    args.epochs_min = 1
    args.eval = False
    args.ckpt = "logs/model"
    args.save_path = "model/"

    args.alpha1 = 1.0
    args.alpha2 = 1.0
    args.beta = 0.1
    args.lr_sc = 10.0
    return args