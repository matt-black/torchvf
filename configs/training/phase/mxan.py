from ml_collections import config_dict


def get_config():
    cfg = config_dict.ConfigDict()

    cfg.DEVICE     = "cuda:0"
    cfg.MODEL_TYPE = "h1"

    cfg.CONFIG_DIR = "training/phase/mxan.py"
    cfg.WEIGHT_DIR = "./weights/phase/mxan/h1/"

    # basic options
    cfg.EPOCHS      = 1000
    cfg.BATCH_SIZE  = 2
    cfg.SAVE_EVERY  = 50
    cfg.IMAGE_EVERY = 50

    # learning rate/training options
    cfg.LR                   = 0.0001
    cfg.EARLY_STOP           = False
    cfg.EARLY_STOP_PATIENCE  = 100
    cfg.EARLY_STOP_MINDELTA  = 0.1

    # pretrained?
    cfg.PRETRAINED     = False
    cfg.PRETRAINED_DIR = None

    cfg.DATA            = config_dict.ConfigDict()
    cfg.DATA.TYPE       = "phase"
    cfg.DATA.DIR        = "./data/phase/mxan"
    cfg.DATA.C          = 1
    cfg.DATA.VF         = True
    cfg.DATA.VF_DELIM   = "vf_k11_a10_e2"
    cfg.DATA.SET_TYPE   = "mat"
    cfg.DATA.TRANSFORMS = "train"
    cfg.DATA.REMOVE     = None
    cfg.DATA.COPY       = None
    cfg.DATA.EXPAND     = 2
    cfg.DATA.RNG_SEED   = 42

    # loss function configuration
    cfg.LOSS                     = config_dict.ConfigDict()
    # IVP loss
    cfg.LOSS.IVP                 = config_dict.ConfigDict()
    cfg.LOSS.IVP.APPLY           = True
    cfg.LOSS.IVP.DX              = 0.25
    cfg.LOSS.IVP.STEPS           = 8
    cfg.LOSS.IVP.SOLVER          = "euler"
    # MSE loss
    cfg.LOSS.MSE                 = config_dict.ConfigDict()
    cfg.LOSS.MSE.APPLY           = True
    # Tversky loss
    cfg.LOSS.TVERSKY             = config_dict.ConfigDict()
    cfg.LOSS.TVERSKY.APPLY       = True
    cfg.LOSS.TVERSKY.ALPHA       = 0.50
    cfg.LOSS.TVERSKY.BETA        = 0.50
    cfg.LOSS.TVERSKY.FROM_LOGITS = True
    # BCE loss
    cfg.LOSS.BCE                 = config_dict.ConfigDict()
    cfg.LOSS.BCE.APPLY           = False

    return cfg
