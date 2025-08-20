import os
import numpy as np
import random
import time
import torch
import argparse
import logging
import datetime
from moco_framework import Moco_v3
from dataset import Moco_data_loader
from train_epoch import moco_process


parser = argparse.ArgumentParser(description="moco training log")

# Definition dataset folder and save path
parser.add_argument('-data', metavar='DIR', default='./data/rdidata_32frames/train/', help="path to dataset")
parser.add_argument('-save-dir', default="/home/sensor_alg/WeiLiu/test/mocov3_test/checkpoints/", help="path to save folder")

# Initial training settings
parser.add_argument('-b', '--batch-size', default= 1024, type=int, metavar='N', help="mini-batch size (default: 128")
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help="initial learning rate", dest='lr')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help="number of training epochs")
  
# Encoder definition
parser.add_argument('--encoder-embed', '--encoder-embed', default=256, type=int, help="initial encoder dimension")
parser.add_argument('--encoder-depth', '--encoder-depth', default= 6, type=int, help="initial encoder depth")
parser.add_argument('--num_heads', '--num_heads', default=8, type=int, help="initial model depth")

# SimSiam definition
parser.add_argument('--mlp-dim', '--mlp-dim', default=512, type=int, help="initial projection head dimension")
parser.add_argument('--mask_ratio', '--mask_ratio', default=0.2, type=float, help="initial masking ratio in encoder")

def main():
    args = parser.parse_args()

    # Set logger formate
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = "logs"
    run_log_dir = os.path.join(base_log_dir, f"run_{timestamp}_ti_p1-4")
    os.makedirs(run_log_dir, exist_ok=True)
    log_file = os.path.join(run_log_dir, "run_log.log")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )

    # Set text logger
    logging_logger = logging.getLogger('moco_v3')
    # Log model architecture and hyperparameters
    logging_logger.info("Training Hyperparameters:")
    for arg in vars(args):
        logging_logger.info(f"{arg}: {getattr(args, arg)}")



    # Load data from npy file
    logging_logger.info("Start load dataset")
    train_loader = Moco_data_loader(args.data, batch_size=args.batch_size)

    # Model definition
    Main_framework = Moco_v3(embed_dim= args.encoder_embed, mlp_dim = args.mlp_dim, encoder_depth= args.encoder_depth, mask_ratio= args.mask_ratio)

    # Device definitioon : default 'cuda'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging_logger.info(f"Using device: {device}")

    # Save data folder creating if not have
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_name = f"Moco-{timestamp}.pth"
    save_path = os.path.join(run_log_dir, save_name)
    logging_logger.info(f"The saving name is {save_name}")

    # Log model architecture and hyperparameters
    logging_logger.info("Training Hyperparameters:")
    for arg in vars(args):
        logging_logger.info(f"{arg}: {getattr(args, arg)}")

    # without split
    moco_process(Main_framework, train_loader, args.lr, args.batch_size, args.epochs, device, save_path, logger = logging_logger)

    logging_logger.info(f"{' ' * 15} End session")
    logging_logger.info("*" * 40)


if __name__ == "__main__":
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(42)
    main()                