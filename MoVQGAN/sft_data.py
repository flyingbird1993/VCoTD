import os
import json
import torch
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from tqdm import tqdm
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


def prepare_image(img):
    """ Transform and normalize PIL Image to tensor. """
    transform = T.Compose([
            T.Resize((128, 192), interpolation=T.InterpolationMode.BICUBIC),
        ])
    pil_image = transform(img)
    arr = np.array(pil_image.convert("RGB"))
    arr = arr.astype(np.float32) / 127.5 - 1
    
    return torch.from_numpy(np.transpose(arr, [2, 0, 1]))

dataroot = './LLaMA-Factory/data/nuscenes'
nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
model = get_movqgan_model('270M', pretrained=True, device='cuda')

samples = nusc.sample
cams = [
            'CAM_FRONT',
            # 'CAM_FRONT_LEFT',
            # 'CAM_FRONT_RIGHT',
            # 'CAM_BACK',
            # 'CAM_BACK_LEFT',
            # 'CAM_BACK_RIGHT',
        ]
gt_indices = {}
success, skipped = 0, 0

for i in tqdm(range(len(samples))):
   rec = samples[i]
   sample = {}

   try:
      for cam in cams:
         samp = nusc.get('sample_data', rec['data'][cam])
         imgname = os.path.join(nusc.dataroot, samp['filename'])
         img_path = imgname
         if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
         img = prepare_image(Image.open(img_path))

         with torch.no_grad():
            out = model(img.to('cuda').unsqueeze(0))

         sample[cam] = str(out.cpu().tolist())

      gt_indices[rec['token']] = sample
      success += 1

   except FileNotFoundError:
      skipped += 1
      continue
   except Exception as e:
      print(f"\n[WARN] sample {i} failed: {e}")
      skipped += 1
      continue

print(f"\nDone: success={success}, skipped={skipped}")
with open("./MoVQGAN/gt_indices_sft.json", "w") as f:
    json.dump(gt_indices, f, indent=4)
print(f"Saved {success} entries to ./MoVQGAN/gt_indices_sft.json")


