import os
import re
import json
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import argparse
import torchvision.transforms as T
from torch.nn.functional import mse_loss, l1_loss
from nuscenes.nuscenes import NuScenes
from movqgan import get_movqgan_model

def show_images(batch, file_path):
    """ Display a batch of images inline. """
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(torch.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    image = Image.fromarray(reshaped.numpy())
    image.save(file_path)


model = get_movqgan_model('270M', pretrained=True, device='cuda')

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--input_json', type=str, required=True, help='Path to input JSON file')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output images')
args = parser.parse_args()

idxs_path = args.input_json
save_dir = args.output_dir
os.makedirs(save_dir, exist_ok=True)

idxs = json.load(open(idxs_path, "r"))
for key, value in idxs.items():
    try:
        token = key
        idx = value
        numbers = re.findall(r'<\|(\d+)\|>', idx)
        idx = [int(num) for num in numbers]
        idx = torch.tensor(idx).to(model.device)
        idx = torch.clamp(idx, min=0, max=16383)
        
        current_length = idx.size(0)
        required_length = 384

        if current_length < required_length:
           pad_length = required_length - current_length
           padding = torch.randint(0, 16384, (pad_length,), device=idx.device, dtype=idx.dtype)
           idx = torch.cat([idx, padding], dim=0)

        with torch.no_grad():
            out = model.decode_code(idx[:required_length].unsqueeze(0))
        save_path = os.path.join(save_dir, f"{token}.png")
        show_images(out, save_path)
    except Exception as e:
        continue