import torch
import argparse
import os

from models.TinyCMIF_model import TinyCMIF_VIO
from models.Tiny_Visual_encoder import Tiny_Visual_encoder

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--experiment_name', type=str, default='experiment', help='experiment name')
parser.add_argument('--pretrain_model',type=str, default=None, help='path to the whole pre-trained VIO model')
parser.add_argument('--pretrain_visual_encoder_conv',type=str, default=None, help='path to the pre-trained conv layer in visual encoder')

parser.add_argument('--img_w', type=int, default=512, help='image width')
parser.add_argument('--img_h', type=int, default=256, help='image height')
parser.add_argument('--v_f_len', type=int, default=512, help='visual feature length')
parser.add_argument('--i_f_len', type=int, default=256, help='imu feature length')
parser.add_argument('--fuse_method', type=str, default='cat', help='fusion method [cat, soft, hard]')
parser.add_argument('--imu_dropout', type=float, default=0, help='dropout for the IMU encoder')

parser.add_argument('--rnn_hidden_size', type=int, default=1024, help='size of the LSTM latent')
parser.add_argument('--rnn_dropout_out', type=float, default=0.2, help='dropout for the LSTM output layer')
parser.add_argument('--rnn_dropout_between', type=float, default=0.2, help='dropout within LSTM')
args = parser.parse_args()

model = TinyCMIF_VIO(args)

cmif_weight_pth = args.pretrain_model
visual_encoder_weight_pth = args.pretrain_visual_encoder_conv

save_dir = os.path.join("results", args.experiment_name)
os.makedirs(save_dir, exist_ok=True)
ckpt_path = os.path.join(
    save_dir,
    f"{args.experiment_name}_fused.pth"
)

cmif = torch.load(cmif_weight_pth, map_location="cpu")
cmif_state = cmif["state_dict"] if "state_dict" in cmif else cmif

model_state = model.state_dict()

cmif_filtered = {
    k: v for k, v in cmif_state.items()
    if not k.startswith("Encoders_net.visual_encoder")
    and k in model_state
}

model_state.update(cmif_filtered)
model.load_state_dict(model_state)

model.Encoders_net.visual_encoder.load_conv_layers(visual_encoder_weight_pth)

torch.save(model.state_dict(), ckpt_path)
print("Saved: ", ckpt_path)

missing, unexpected = model.load_state_dict(model.state_dict(), strict=False)
print("Missing:", missing)
print("Unexpected:", unexpected)
