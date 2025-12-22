import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import logging
from path import Path
from utils import custom_transform
from dataset.KITTI_dataset import KITTI
from torch.cuda.amp import autocast, GradScaler

from torch.optim.lr_scheduler import CosineAnnealingLR

from models.FlowNet_encoder import FlowNet_Encoder
from models.Tiny_Visual_encoder import Visual_encoder_for_Distill


def visual_encoder_distillation(student_encoder, teacher_encoder, optimizer, train_loader, 
                           logger, ep, scaler, args=None):
    
    losses = []
    
    data_len = len(train_loader)

    for i, (imgs, imus, gts, rot, weight) in enumerate(train_loader):

        imgs = imgs.cuda(non_blocking=True).float()
        img_pairs = torch.cat((imgs[:, :-1], imgs[:, 1:]), dim=2)

        batch_size = img_pairs.size(0)
        seq_len = img_pairs.size(1)
        H, W = args.img_h, args.img_w

        img_pairs = img_pairs.view(batch_size * seq_len, 6, H, W)

        optimizer.zero_grad(set_to_none=True)
        
        # Use mixed precision training
        with autocast():
            # Get teacher predictions (no gradient)
            with torch.no_grad():
                teacher_feat = teacher_encoder(img_pairs)

            # Get student predictions
            student_feat = student_encoder(img_pairs)
            
            # Align teacher output with student by interpolating to student's spatial size
            teacher_feat = F.interpolate(teacher_feat, size=student_feat.shape[-2:], 
                                        mode='bilinear', align_corners=False)

            # Loss
            loss = F.l1_loss(
                F.normalize(teacher_feat, dim=1),
                F.normalize(student_feat, dim=1)
            )
        
        # Backward with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if i % args.print_frequency == 0: 
            message = (f'Epoch: {ep}, iters: {i}/{data_len}, '
                      f'distill_loss: {loss.item():.6f}')
            print(message)
            logger.info(message)

        losses.append(loss.item())
    
    return np.mean(losses)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='../Visual-Selective-VIO/data', help='path to the dataset')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--save_dir', type=str, default='./results', help='path to save the result')
    
    parser.add_argument('--train_seq', type=list, default=['00', '01', '02', '04', '06', '08'], help='sequences for training')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    
    parser.add_argument('--img_w', type=int, default=512, help='image width')
    parser.add_argument('--img_h', type=int, default=256, help='image height')
    
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--seq_len', type=int, default=11, help='sequence length for LSTM')
    parser.add_argument('--workers', type=int, default=8, help='number of workers')
    
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs for distillation')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay for the optimizer')
    
    parser.add_argument('--teacher_ckpt', type=str, default='./results/training/checkpoints/best_3.61.pth', help='path to teacher checkpoint')
    parser.add_argument('--experiment_name', type=str, default='visual_encoder_distillation', help='experiment name')
    
    parser.add_argument('--hflip', default=False, action='store_true', help='whether to use horizonal flipping as augmentation')
    parser.add_argument('--color', default=False, action='store_true', help='whether to use color augmentations')
    parser.add_argument('--print_frequency', type=int, default=10, help='print frequency for loss values')
    
    args = parser.parse_args()
    
    # Set the random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create directories
    experiment_dir = Path(args.save_dir)
    experiment_dir.mkdir_p()
    file_dir = experiment_dir.joinpath('{}/'.format(args.experiment_name))
    file_dir.mkdir_p()
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir_p()
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir_p()
    
    # Create logger
    logger = logging.getLogger(args.experiment_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/train_%s.txt' % args.experiment_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('----------------------------------------DISTILLATION TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)
    
    # GPU setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("CUDA available:", torch.cuda.is_available())
    logger.info("CUDA available: {}".format(torch.cuda.is_available()))

    # ============================================================
    #    Load teacher encoder (keep on GPU but no DataParallel)
    # ============================================================
    print("Loading teacher encoder...")
    logger.info("Loading teacher encoder...")
    
    teacher = FlowNet_Encoder().to(device)

    teacher_ckpt = torch.load(args.teacher_ckpt, map_location="cpu")

    state_dict = {}
    for k, v in teacher_ckpt.items():
        if k.startswith("Encoders_net.conv"):
            new_key = k.replace("Encoders_net.", "")
            state_dict[new_key] = v

    missing, unexpected = teacher.load_state_dict(state_dict, strict=False)

    print("Teacher - Missing keys:", missing)
    print("Teacher - Unexpected keys:", unexpected)
    logger.info("Teacher - Missing keys: {}".format(missing))
    logger.info("Teacher - Unexpected keys: {}".format(unexpected))

    for p in teacher.parameters():
        p.requires_grad = False

    teacher.eval()
    # Don't wrap teacher in DataParallel for single GPU

    # ============================================================
    #    Load student encoder (no DataParallel for single GPU)
    # ============================================================
    print("Initializing student encoder...")
    logger.info("Initializing student encoder...")
    
    student = Visual_encoder_for_Distill().to(device)
    # Don't wrap student in DataParallel for single GPU

    student.load_from_rgb_checkpoint("./results/1219_visual_encoder_distillation/checkpoints/best_0.0047.pth")

    # ============================================================
    #    Load dataset
    # ============================================================
    print("Loading dataset...")
    logger.info("Loading dataset...")
    
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
                        transform=transform_train)
    
    logger.info('train_dataset: ' + str(train_dataset))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True if args.workers > 0 else False,
        prefetch_factor=2 if args.workers > 0 else None)

    # ============================================================
    #    Initialize optimizer and gradient scaler
    # ============================================================
    optimizer = torch.optim.Adam(student.parameters(), 
                                 lr=args.lr, 
                                 betas=(0.9, 0.999), 
                                 eps=1e-08, 
                                 weight_decay=args.weight_decay)
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler()

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ============================================================
    #    Knowledge distillation training
    # ============================================================
    print("Starting distillation training...")
    logger.info("Starting distillation training...")
    logger.info(f"Batch size: {args.batch_size}, Workers: {args.workers}")
    logger.info(f"Using mixed precision training (FP16)")
    
    best_loss = float('inf')
    
    for ep in range(args.epochs):
        student.train()
        
        avg_loss = visual_encoder_distillation(
            student_encoder=student,
            teacher_encoder=teacher,
            optimizer=optimizer,
            train_loader=train_loader,
            logger=logger,
            ep=ep,
            scaler=scaler,
            args=args,
        )

        scheduler.step()
        
        # Save checkpoint
        torch.save(student.state_dict(), f'{checkpoints_dir}/epoch_{ep:03d}.pth')
        
        message = f'Epoch {ep} finished, avg_distill_loss: {avg_loss:.6f}, model saved'
        print(message)
        logger.info(message)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(student.state_dict(), f'{checkpoints_dir}/best_{best_loss:.4f}.pth')
            message = f'Best model saved with loss: {best_loss:.6f}'
            print(message)
            logger.info(message)
    
    message = f'Training finished, best loss: {best_loss:.6f}'
    print(message)
    logger.info(message)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted – cleaning up GPU memory…")
    finally:
        torch.cuda.empty_cache()
        for p in torch.multiprocessing.active_children():
            p.terminate()
            p.join()
        print("Cleanup complete. Exiting safely.")