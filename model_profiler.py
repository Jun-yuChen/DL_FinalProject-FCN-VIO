import argparse
import torch
import time
from thop import profile
from utils.kitti_eval import data_partition

from models.CMIF_model import CMIF_VIO
from models.SmallCMIF_model import SmallCMIF_VIO
from models.TinyCMIF_model import TinyCMIF_VIO

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', type=str, default='../Visual-Selective-VIO/data', help='path to the dataset')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--save_dir', type=str, default='./results', help='path to save the result')
parser.add_argument('--seq_len', type=int, default=11, help='sequence length for LSTM')
parser.add_argument('--use_grey_img', action='store_true', help='use grayscale images')

parser.add_argument('--train_seq', type=list, default=['00', '01', '02', '04', '06', '08', '09'], help='sequences for training')
parser.add_argument('--val_seq', type=list, default=['05', '07', '10'], help='sequences for validation')
parser.add_argument('--seed', type=int, default=0, help='random seed')

parser.add_argument('--img_w', type=int, default=512, help='image width')
parser.add_argument('--img_h', type=int, default=256, help='image height')
parser.add_argument('--v_f_len', type=int, default=512, help='visual feature length')
parser.add_argument('--i_f_len', type=int, default=256, help='imu feature length')
parser.add_argument('--fuse_method', type=str, default='cat', help='fusion method [cat, soft, hard]')
parser.add_argument('--imu_dropout', type=float, default=0, help='dropout for the IMU encoder')

parser.add_argument('--rnn_hidden_size', type=int, default=1024, help='size of the LSTM latent')
parser.add_argument('--rnn_dropout_out', type=float, default=0.2, help='dropout for the LSTM output layer')
parser.add_argument('--rnn_dropout_between', type=float, default=0.2, help='dropout within LSTM')

parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--experiment_name', type=str, default='test', help='experiment name')
parser.add_argument('--model', type=str, default='./pretrain_models/vf_512_if_256_3e-05.model', help='path to the pretrained model')

args = parser.parse_args()


# load model
model = CMIF_VIO(args)
model_name = "CMIF_VIO"

# model = TinyCMIF_VIO(args)
# model_name = "TinyCMIF_VIO"

print("Model:", model_name)

model.load_state_dict(torch.load(args.model))
print('load model %s'%args.model)
    
# Feed model to GPU
model.eval()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(DEVICE)

# load data
seq = '07'
dataset = data_partition(args, seq)
print("Number of samples:", len(dataset))

def rgb_pair_to_gray(x):
    """
    x: (B, 6, H, W)  -> two RGB images concatenated on channel dim
    return: (B, 2, H, W) -> two grayscale images concatenated on channel dim
    """
    r1, g1, b1 = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    gray1 = 0.299 * r1 + 0.587 * g1 + 0.114 * b1

    r2, g2, b2 = x[:, 3:4], x[:, 4:5], x[:, 5:6]
    gray2 = 0.299 * r2 + 0.587 * g2 + 0.114 * b2

    return torch.cat([gray1, gray2], dim=1)

# Access one sample
image_seq, imu_seq, _ = dataset[0]

image_seq = image_seq[0:2]

imus = imu_seq[0:11]

print(image_seq.shape, imus.shape)

image_seq = image_seq.unsqueeze(0).to(DEVICE)  # [1, 2, 3, H, W]
imus = imus.unsqueeze(0).to(DEVICE)         # [1, 11, 6]
print(image_seq.shape, imus.shape)

# ==============================================
# Test FLOPs
# ==============================================

# calculate FLOPs
input_test = (image_seq, imus)
flops, params = profile(model, inputs=input_test)

# output
print("flops ={:.0f}".format(flops))


# ==============================================
# Test parameter size
# ==============================================
# calculate parameter size
param_size = 0
for param in model.parameters(): 
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()
total_size_KB = (param_size + buffer_size) / 1024.0

# output
print("Model parameter size ={:.3f} KB".format(total_size_KB))


# ==============================================
# Measure Inference Time
# ==============================================

model.eval()

# Warm-up runs (important for GPU timing)
print("\nWarming up...")
with torch.no_grad():
    for _ in range(10):
        _ = model(image_seq, imus)

# Synchronize GPU before timing
if DEVICE.type == 'cuda':
    torch.cuda.synchronize()

# Method 1: Using time.perf_counter (CPU time)
num_iterations = 100
print(f"\nRunning {num_iterations} iterations...")

start_time = time.perf_counter()
with torch.no_grad():
    for _ in range(num_iterations):
        _ = model(image_seq, imus)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()  # Wait for GPU operations to complete

end_time = time.perf_counter()
avg_time_cpu = (end_time - start_time) / num_iterations

# Method 2: Using CUDA Events (more accurate for GPU)
if DEVICE.type == 'cuda':
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    
    timings = []
    with torch.no_grad():
        for _ in range(num_iterations):
            starter.record()
            _ = model(image_seq, imus)
            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender))  # milliseconds
    
    avg_time_gpu = sum(timings) / len(timings)
    
    print(f"\nInference Time per image pair (CPU timing): {avg_time_cpu*1000:.2f} ms")
    print(f"Inference Time per image pair (CUDA Events): {avg_time_gpu:.2f} ms")

else:
    print(f"\nInference Time: {avg_time_cpu*1000:.2f} ms")
