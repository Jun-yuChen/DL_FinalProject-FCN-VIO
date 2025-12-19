import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from path import Path
from utils import custom_transform
from dataset.KITTI_dataset import KITTI

from SmallCMIF_model import SmallCMIF_VIO
from CMIF_model import CMIF_VIO

from collections import defaultdict
from utils.kitti_eval import KITTI_tester
import numpy as np
import math

from torch.optim.lr_scheduler import StepLR

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', type=str, default='/nfs/turbo/coe-hunseok/mingyuy/KITTI_odometry', help='path to the dataset')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--save_dir', type=str, default='./results', help='path to save the result')

parser.add_argument('--train_seq', type=list, default=['00', '01', '02', '04', '06', '08'], help='sequences for training')
parser.add_argument('--val_seq', type=list, default=['09'], help='sequences for validation')
parser.add_argument('--seed', type=int, default=42, help='random seed')

# Student model parameters
parser.add_argument('--img_w', type=int, default=512, help='image width')
parser.add_argument('--img_h', type=int, default=256, help='image height')
parser.add_argument('--v_f_len', type=int, default=512, help='visual feature length (student, smaller than teacher)')
parser.add_argument('--i_f_len', type=int, default=256, help='imu feature length (student, smaller than teacher)')
parser.add_argument('--imu_dropout', type=float, default=0, help='dropout for the IMU encoder')

parser.add_argument('--rnn_hidden_size', type=int, default=1024, help='size of the LSTM latent (student, smaller than teacher)')
parser.add_argument('--rnn_dropout_out', type=float, default=0.2, help='dropout for the LSTM output layer')
parser.add_argument('--rnn_dropout_between', type=float, default=0.2, help='dropout within LSTM')

# Teacher model parameters
parser.add_argument('--teacher_v_f_len', type=int, default=512, help='teacher visual feature length')
parser.add_argument('--teacher_i_f_len', type=int, default=256, help='teacher imu feature length')
parser.add_argument('--teacher_rnn_hidden_size', type=int, default=1024, help='teacher LSTM hidden size')
parser.add_argument('--teacher_model', type=str, required=True, help='path to the pretrained teacher model')

# Distillation parameters
parser.add_argument('--distill_temperature', type=float, default=4.0, help='temperature for distillation')
parser.add_argument('--distill_alpha', type=float, default=1.0, help='weight for distillation loss (vs hard target loss)')
parser.add_argument('--distill_beta', type=float, default=0.3, help='weight for feature distillation loss')
parser.add_argument('--distill_gamma', type=float, default=0.3, help='weight for visual encoder distillation loss')

parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay for the optimizer')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--seq_len', type=int, default=11, help='sequence length for LSTM')
parser.add_argument('--workers', type=int, default=4, help='number of workers')

parser.add_argument('--epochs_visual', type=int, default=20, help='number of epochs for strong visual encoder distillation')
parser.add_argument('--epochs_joint', type=int, default=60, help='number of epochs for joint training')
parser.add_argument('--epochs_fine', type=int, default=20, help='number of epochs for finetuning')

parser.add_argument('--lr_init', type=float, default=1e-4, help='Initial learning rate')
parser.add_argument('--lr_stepSize', type=int, default=20, help='Step size for learning rate scheduler (StepLR)')
parser.add_argument('--eta', type=float, default=0.05, help='exponential decay factor for temperature')
parser.add_argument('--temp_init', type=float, default=5, help='initial temperature for gumbel-softmax')

parser.add_argument('--experiment_name', type=str, default='distillation_experiment', help='experiment name')
parser.add_argument('--optimizer', type=str, default='Adam', help='type of optimizer [Adam, SGD]')

# parser.add_argument('--pretrain_flownet',type=str, default='./pretrain_models/flownets_bn_EPE2.459.pth.tar', help='whether to use the pre-trained flownet')
parser.add_argument('--pretrain_student', type=str, default=None, help='path to the pretrained student model')
parser.add_argument('--hflip', default=False, action='store_true', help='whether to use horizonal flipping as augmentation')
parser.add_argument('--color', default=False, action='store_true', help='whether to use color augmentations')

parser.add_argument('--print_frequency', type=int, default=10, help='print frequency for loss values')
parser.add_argument('--weighted', default=False, action='store_true', help='whether to use weighted sum')

args = parser.parse_args()

# Set the random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)


def prediction_distill_loss(student_pred, teacher_pred):
    """
    Compute distillation loss using MSE
    Args:
        student_logits: predictions from student model
        teacher_logits: predictions from teacher model
    """
    L_distill = F.smooth_l1_loss(student_pred, teacher_pred)
    return L_distill


def visualFeature_distill_loss(student_v_features, teacher_v_features):
    """
    MSE loss between student and teacher intermediate features
    """
    return F.mse_loss(student_v_features, teacher_v_features)


def update_status(ep, args):
    if ep < args.epochs_visual:  
        gamma = args.distill_gamma * (0.8 ** (ep // 20))

    elif ep >= args.epochs_visual and ep < args.epochs_visual + args.epochs_joint: # Joint training stage
        gamma = args.distill_gamma * (0.8 ** (args.epochs_visual // 20)) * (0.5 ** ((ep-args.epochs_visual) // 20))

    elif ep >= args.epochs_visual + args.epochs_joint: # Finetuning stage
        gamma = 0

    return gamma


def train_with_distillation(student_model, teacher_model, optimizer, train_loader, 
                           logger, ep, 
                           args=None, weighted=False):
    
    total_losses = []
    hard_losses = []
    pred_distil_losses = []
    visual_distil_losses = []
    
    data_len = len(train_loader)

    gamma = update_status(ep, args)
    print("current gamma: ", gamma)

    for i, (imgs, imus, gts, rot, weight) in enumerate(train_loader):

        imgs = imgs.cuda().float()
        imus = imus.cuda().float()
        gts = gts.cuda().float() 
        weight = weight.cuda().float()

        optimizer.zero_grad()
        
        # Get teacher predictions (no gradient)
        with torch.no_grad():
            teacher_poses, teacher_visual_feat, teacher_hc = teacher_model(imgs, imus, hc=None)
        
        # Get student predictions
        student_poses, student_visual_feat, student_hc = student_model(imgs, imus, hc=None)
        
        # === 1. Hard target loss (MSE with ground truth) ===
        if not weighted:
            angle_loss = F.mse_loss(student_poses[:,:,:3], gts[:, :, :3])
            translation_loss = F.mse_loss(student_poses[:,:,3:], gts[:, :, 3:])
        else:
            weight = weight/weight.sum()
            angle_loss = (weight.unsqueeze(-1).unsqueeze(-1) * (student_poses[:,:,:3] - gts[:, :, :3]) ** 2).mean()
            translation_loss = (weight.unsqueeze(-1).unsqueeze(-1) * (student_poses[:,:,3:] - gts[:, :, 3:]) ** 2).mean()
        
        hard_loss = 100 * angle_loss + translation_loss
        
        # === 2. Soft target loss (distillation from teacher) ===
        # Reshape poses for distillation (treat each pose dimension as a logit)
        batch_size, seq_len, pose_dim = student_poses.shape
        student_flat = student_poses.reshape(-1, pose_dim)
        teacher_flat = teacher_poses.reshape(-1, pose_dim)
        
        # Use MSE as soft target loss for regression task
        pred_distil_loss = F.mse_loss(student_flat, teacher_flat)

        # === 3. Visual encoder distillation loss ===
        visual_distil_loss = visualFeature_distill_loss(student_visual_feat, teacher_visual_feat)
        
        # === Combined loss ===
        # Alpha controls balance between hard and soft targets
        # Gamma controls hidden state matching
        loss = ((1 - args.distill_alpha) * hard_loss + 
                args.distill_alpha * pred_distil_loss + 
                gamma * visual_distil_loss)
        
        loss.backward()
        optimizer.step()
        
        if i % args.print_frequency == 0: 
            message = (f'Epoch: {ep}, iters: {i}/{data_len}, '
                      f'total_loss: {loss.item():.6f}, '
                      f'hard_loss: {hard_loss.item():.6f}, '
                      f'pred_distil_loss: {pred_distil_loss.item():.6f}, ' 
                      f'visual_distil_loss: {visual_distil_loss.item():.6f}')
            print(message)
            logger.info(message)

        total_losses.append(loss.item())
        hard_losses.append(hard_loss.item())
        pred_distil_losses.append(pred_distil_loss.item())
        visual_distil_losses.append(visual_distil_loss.item())

    metrics = {
        'total_loss': np.mean(total_losses),
        'hard_loss': np.mean(hard_losses),
        'pred_distil_loss': np.mean(pred_distil_losses),
        'visual_distil_loss': np.mean(visual_distil_losses),
    }
    
    
    return metrics


def main():

    TEACHER_MODEL = CMIF_VIO
    STUDENT_MODEL = SmallCMIF_VIO
    

    # Create Dir
    experiment_dir = Path('./results')
    experiment_dir.mkdir_p()
    file_dir = experiment_dir.joinpath('{}/'.format(args.experiment_name))
    file_dir.mkdir_p()
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir_p()
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir_p()
    
    # Create logs
    logger = logging.getLogger(args.experiment_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/train_distill_%s.txt'%args.experiment_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('----------------------------------------DISTILLATION TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)
    
    # Load the dataset
    transform_train = [custom_transform.ToTensor(),
                       custom_transform.Resize((args.img_h, args.img_w))]
    if args.hflip:
        transform_train += [custom_transform.RandomHorizontalFlip()]
    if args.color:
        transform_train += [custom_transform.RandomColorAug()]
    transform_train = custom_transform.Compose(transform_train)

    train_dataset = KITTI(args.data_dir,
                        sequence_length=args.seq_len,
                        train_seqs=args.train_seq,
                        transform=transform_train
                        )
    logger.info('train_dataset: ' + str(train_dataset))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    
    # GPU selections
    str_ids = args.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])

    print("CUDA available:", torch.cuda.is_available())
    
    # Initialize the tester
    tester = KITTI_tester(args)

    # === Initialize TEACHER model ===
    print('Loading teacher model...')
    logger.info('Loading teacher model...')
    
    # Create teacher args with teacher-specific dimensions
    teacher_args = argparse.Namespace(**vars(args))
    teacher_args.v_f_len = args.teacher_v_f_len
    teacher_args.i_f_len = args.teacher_i_f_len
    teacher_args.rnn_hidden_size = args.teacher_rnn_hidden_size
    
    teacher_model = TEACHER_MODEL(teacher_args)
    
    teacher_model.load_state_dict(torch.load(args.teacher_model))
    teacher_model.cuda(gpu_ids[0])
    teacher_model = torch.nn.DataParallel(teacher_model, device_ids=gpu_ids)
    teacher_model.eval()  # Set teacher to eval mode
    
    # Freeze teacher parameters
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    print(f'Teacher model loaded from {args.teacher_model}')
    logger.info(f'Teacher model loaded from {args.teacher_model}')

    # === Initialize STUDENT model ===
    print('Initializing student model...')
    logger.info('Initializing student model...')

    student_model = STUDENT_MODEL(args)

    # Load pretrained student if available
    if args.pretrain_student is not None:
        student_model.load_state_dict(torch.load(args.pretrain_student))
        print(f'Student model loaded from {args.pretrain_student}')
        logger.info(f'Student model loaded from {args.pretrain_student}')
    else:
        print('Training student from scratch')
        logger.info('Training student from scratch')

    # Feed student model to GPU
    student_model.cuda(gpu_ids[0])
    student_model = torch.nn.DataParallel(student_model, device_ids=gpu_ids)

    # === Initialize feature/hidden adapters if needed ===
    
    pretrain = args.pretrain_student
    init_epoch = int(pretrain[-7:-4])+1 if args.pretrain_student is not None else 0    
    
    # Initialize the optimizer (including adapter parameters if present)
    init_lr = args.lr_init

    params_to_optimize = list(student_model.parameters())
    
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params_to_optimize, lr=init_lr, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params_to_optimize, lr=init_lr, betas=(0.9, 0.999), 
                                     eps=1e-08, weight_decay=args.weight_decay)
    
    # Using LR scheduler
    scheduler = StepLR(optimizer, step_size=args.lr_stepSize, gamma=0.1)

    print("Initial LR: ", init_lr)
    print("StepLR step size: ", args.lr_stepSize)

    best = 10000

    for ep in range(init_epoch, args.epochs_visual + args.epochs_joint + args.epochs_fine):
        current_lr = optimizer.param_groups[0]['lr']
        message = f'Epoch: {ep}, lr: {current_lr}'
        print(message)
        logger.info(message)

        student_model.train()
        
        # Train with distillation
        metrics = train_with_distillation(student_model, teacher_model, optimizer, 
                                         train_loader, logger, ep,
                                         args, weighted=args.weighted)
        
        # Update the learning rate for the NEXT epoch
        scheduler.step()
        
        # Save the model after training
        torch.save(student_model.module.state_dict(), f'{checkpoints_dir}/{ep:003}.pth')
        
        message = (f'Epoch {ep} training finished, '
                  f'total_loss: {metrics["total_loss"]:.6f}, '
                  f'hard_loss: {metrics["hard_loss"]:.6f}, '
                  f'pred_distil_loss: {metrics["pred_distil_loss"]:.6f}, '
                  f'visual_distil_loss: {metrics["visual_distil_loss"]:.6f}, '
                  f'model saved')
        print(message)
        logger.info(message)
        
        # Evaluate the model
        print('Evaluating the student model')
        logger.info('Evaluating the student model')
        with torch.no_grad(): 
            student_model.eval()
            errors = tester.eval(student_model, num_gpu=len(gpu_ids))
    
        t_rel = np.mean([errors[i]['t_rel'] for i in range(len(errors))])
        r_rel = np.mean([errors[i]['r_rel'] for i in range(len(errors))])
        t_rmse = np.mean([errors[i]['t_rmse'] for i in range(len(errors))])
        r_rmse = np.mean([errors[i]['r_rmse'] for i in range(len(errors))])

        if t_rel < best:
            best = t_rel 
            torch.save(student_model.module.state_dict(), f'{checkpoints_dir}/best_{best:.2f}.pth')
    
        message = f'Epoch {ep} evaluation finished , t_rel: {t_rel:.4f}, r_rel: {r_rel:.4f}, t_rmse: {t_rmse:.4f}, r_rmse: {r_rmse:.4f}, best t_rel: {best:.4f}'
        logger.info(message)
        print(message)
    
    message = f'Distillation training finished, best t_rel: {best:.4f}'
    logger.info(message)
    print(message)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted — cleaning up GPU memory…")
    finally:
        torch.cuda.empty_cache()
        for p in torch.multiprocessing.active_children():
            p.terminate()
            p.join()
        print("Cleanup complete. Exiting safely.")