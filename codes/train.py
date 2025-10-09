# codes/train.py
# Revised training script for CVPR-2026:
# - Uses correct create_dataloader signature (dataset, dataset_opt)
# - Steps LR schedulers AFTER optimizer.step() (silences PyTorch warning)
# - Saves checkpoints reliably (model.save + save_training_state)
# - Simple validation with PSNR logging
# - Works with options/options.py, data/, models/

import os
import sys
import argparse
import math
import logging
from pathlib import Path

import torch
import numpy as np

# repo imports
import options.options as option
from data import create_dataset, create_dataloader
from models import create_model


def setup_logger(log_dir: str, name: str = 'base'):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{name}.log')
    logger = logging.getLogger('base')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def dict_to_str(dic, indent_l=1):
    msg = '\n'
    for k, v in dic.items():
        if isinstance(v, dict):
            msg += '  ' * indent_l + f'{k}:[\n'
            msg += dict_to_str(v, indent_l + 1)
            msg += '  ' * indent_l + ']\n'
        else:
            msg += '  ' * indent_l + f'{k}: {v}\n'
    return msg


def ensure_dirs(opt):
    paths = opt.get('path', {})
    for k in ['experiments_root', 'models', 'training_state', 'log', 'val_images']:
        p = paths.get(k, None)
        if p:
            Path(p).mkdir(parents=True, exist_ok=True)


def tensor_to_img_uint8(t):
    t = t.detach().float().cpu().clamp(0, 1).numpy()
    t = np.transpose(t, (1, 2, 0))
    return (t * 255.0 + 0.5).astype(np.uint8)


def calc_psnr_uint8(a, b, border=0):
    if border > 0:
        a = a[border:-border, border:-border, ...]
        b = b[border:-border, border:-border, ...]
    a = a.astype(np.float64); b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    args = parser.parse_args()

    # parse options
    opt = option.parse(args.opt, is_train=True)
    ensure_dirs(opt)

    # set random seed
    if opt['train'].get('manual_seed', None) is not None:
        seed = int(opt['train']['manual_seed'])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    # logger
    logger = setup_logger(opt['path']['log'])
    logger.info('export CUDA_VISIBLE_DEVICES={}'.format(os.environ.get('CUDA_VISIBLE_DEVICES', '')))
    logger.info(dict_to_str(opt))

    # datasets & loaders  (NOTE: create_dataloader(dataset, dataset_opt))
    train_set = create_dataset(opt['datasets']['train'])
    train_loader = create_dataloader(train_set, opt['datasets']['train'])
    logger.info(f"Dataset [{train_set.__class__.__name__} - {opt['datasets']['train']['name']}] is created.")
    logger.info(f"Number of train images: {len(train_set)}, iters: {len(train_loader)}")

    val_set = create_dataset(opt['datasets']['val'])
    val_loader = create_dataloader(val_set, opt['datasets']['val'])
    logger.info(f"Dataset [{val_set.__class__.__name__} - {opt['datasets']['val']['name']}] is created.")
    logger.info(f"Number of val images in [{opt['datasets']['val']['name']}]: {len(val_set)}")

    # model
    model = create_model(opt)
    logger.info(f"Model [{model.__class__.__name__}] is created.")
    logger.info("Start training from epoch: 0, iter: 0")

    # training params
    niter = int(opt['train']['niter'])
    print_freq = int(float(opt['logger']['print_freq']))
    val_freq = int(float(opt['train']['val_freq']))
    save_freq = int(float(opt['logger']['save_checkpoint_freq']))
    scale = int(opt.get('scale', 4))
    current_step = 0
    epoch = 0

    try:
        while current_step < niter:
            epoch += 1
            for _, train_data in enumerate(train_loader):
                if current_step >= niter:
                    break
                current_step += 1

                # one step
                model.feed_data(train_data)
                model.optimize_parameters(current_step)  # optimizer.step() is inside

                # STEP SCHEDULERS AFTER OPTIMIZER to silence warning
                for s in model.schedulers:
                    s.step()

                # logging
                if current_step % print_freq == 0:
                    try:
                        lr_now = model.optimizers[0].param_groups[0]['lr']
                    except Exception:
                        lr_now = 0.0
                    logs = model.get_current_log()
                    items = ' '.join([f"{k}: {v:.4e}" for k, v in logs.items()])
                    logger.info(f"<epoch: {epoch:3d}, iter: {current_step:7d}, lr:{lr_now:.3e}> {items} ")

                # validation
                if current_step % val_freq == 0:
                    try:
                        val_batch = next(iter(val_loader))
                        model.feed_data(val_batch)
                        model.test()
                        vis = model.get_current_visuals(need_HR=True)
                        sr = tensor_to_img_uint8(vis['SR'])
                        hr = tensor_to_img_uint8(vis['HR'])
                        psnr = calc_psnr_uint8(sr, hr, border=scale)
                        logger.info(f"# Validation # PSNR: {psnr:.4e}")
                        logger.info(f"<epoch: {epoch:3d}, iter: {current_step:7d}> psnr: {psnr:.4e}")
                    except Exception as e:
                        logger.info(f"# Validation # (skipped PSNR calc due to: {e})")

                # save
                if current_step % save_freq == 0:
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)
                    logger.info("Saving models and training states.")

        # end of training
        model.save('final')
        logger.info("Saving the final model.")
        logger.info("End of training.")

    except KeyboardInterrupt:
        logger.info("Interrupted by user! Saving current state...")
        model.save(current_step)
        model.save_training_state(epoch, current_step)
        logger.info("State saved. Exiting.")


if __name__ == '__main__':
    main()
