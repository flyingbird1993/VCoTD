import os
import json
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import torchvision.transforms as T
from movqgan import get_movqgan_model
import torch.distributed as dist
import torch.multiprocessing as mp
from functools import partial

def show_images(batch, file_path):
    scaled = ((batch + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    image = Image.fromarray(reshaped.numpy())
    image.save(file_path)

def prepare_image(img):
    transform = T.Compose([
        T.Resize((128, 192), interpolation=T.InterpolationMode.BICUBIC),
    ])
    pil_image = transform(img)
    arr = np.array(pil_image.convert("RGB"))
    arr = arr.astype(np.float32) / 127.5 - 1

    return torch.from_numpy(np.transpose(arr, [2, 0, 1]))

def convert_to_png(img):
    png_buffer = BytesIO()
    img.save(png_buffer, format='PNG')
    png_buffer.seek(0)

    return Image.open(png_buffer)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12358'#'12258'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
    # dist.destroy_process_group()

def save_partial_results(partial_results, rank,output_json_path):
    with open(output_json_path, "w") as f:
        json.dump(partial_results, f, indent=4)

    print(f"Rank {rank} has completed processing and some results have been saved to {output_json_path}.")

def run_inference(rank, world_size, image_files, image_dir,output_dir,output_name):
    try:
        setup(rank, world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')

        model = get_movqgan_model('270M', pretrained=True, device=device, cache_dir='/home/flyingbird/models')
        model.eval()
        total_images = len(image_files)
        images_per_rank = (total_images + world_size - 1) // world_size
        start_idx = rank * images_per_rank
        end_idx = min(start_idx + images_per_rank, total_images)
        local_image_files = image_files[start_idx:end_idx]

        gt_indices = {}

        for img_file in tqdm(local_image_files, desc=f"Rank {rank} process the image", position=rank):
            img_path = os.path.join(image_dir, img_file)

            try:
                original_img = Image.open(img_path).convert("RGB")

            except Exception as e:
                print(f"Failed to open image {img_path}: {e}")
                continue

            png_img = convert_to_png(original_img)
            img_tensor = prepare_image(png_img)
            img_tensor = img_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)

            gt_indices[img_file] = str(output.cpu().tolist())

        name=output_name.split(".")[0]
        output_json_path = os.path.join(output_dir, f"{name}_{rank}.json")
        save_partial_results(gt_indices, rank,output_json_path)

    finally:
        cleanup()

def aggregate_results(world_size, output_dir, final_output_path,output_name):
    aggregated = {}
    name=output_name.split(".")[0]

    for rank in range(world_size):
        partial_json_path = os.path.join(output_dir, f"{name}_{rank}.json")

        if os.path.exists(partial_json_path):
            with open(partial_json_path, "r") as f:
                partial_data = json.load(f)

            aggregated.update(partial_data)
            os.remove(partial_json_path)
        else:
            print(f"Warning: {partial_json_path} not found")

    with open(final_output_path, "w") as f:
        json.dump(aggregated, f, indent=4)

    print(f"All the partial results have been merged and saved to {final_output_path}")

def main(image_dir, output_dir, output_name):
    supported_extensions = ('.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG')
    image_files = []

    for f in os.listdir(image_dir):
        if os.path.isfile(os.path.join(image_dir, f)):
            image_files.append(f)


    print(f"Found {len(image_files)} images for processing.")
    world_size = torch.cuda.device_count()

    if world_size < 1:
        raise ValueError("There is no available GPU on this machine.")

    os.makedirs(output_dir, exist_ok=True)
    final_output_path = os.path.join(output_dir, output_name)

    mp.spawn(
        run_inference,
        args=(world_size, image_files, image_dir, output_dir, output_name),
        nprocs=world_size,
        join=True
    )

    aggregate_results(world_size, output_dir, final_output_path,output_name)

    print(f"Processing completed. All results have been saved to {final_output_path}")

if __name__ == "__main__":
    image_dir="./LLaMA-Factory/data/nuscenes/v1.0-mini/sweeps/CAM_FRONT"
    output_dir="./MoVQGAN"
    output_name="gt_indices_pretrain.json"
    main(image_dir, output_dir, output_name)
