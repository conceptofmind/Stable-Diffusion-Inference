from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import importlib.util
import argparse
import torch
from tqdm import tqdm

assert importlib.util.find_spec("PIL") is not None, "pip install Pillow"
assert importlib.util.find_spec("diffusers") is not None, "pip install diffusers==0.2.4"
assert importlib.util.find_spec("transformers") is not None, "pip install transformers"
assert importlib.util.find_spec("scipy") is not None, "pip install scipy"
assert importlib.util.find_spec("ftfy") is not None, "pip install ftfy"

class CFG:
    prompt = "A photo of a cat"
    height = 512 #768
    width = 512
    num_images_per_batch = 1
    num_steps = 20
    num_inference_steps = 50
    use_gpu = True
    use_low_mem = False
    output_dir = "../diffusion_images/"
    my_token = None

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=CFG.prompt)
    parser.add_argument("--height", type=int, default=CFG.height)
    parser.add_argument("--width", type=int, default=CFG.width)
    parser.add_argument("--num_images_per_batch", type=int, default=CFG.num_images_per_batch)
    parser.add_argument("--num_steps", type=int, default=CFG.num_steps)
    parser.add_argument("--num_inference_steps", type=int, default=CFG.num_inference_steps)
    parser.add_argument("--use_gpu", type=bool, default=CFG.use_gpu)
    parser.add_argument("--use_low_mem", type=bool, default=CFG.use_low_mem)
    parser.add_argument("--output_dir", type=str, default=CFG.output_dir)
    parser.add_argument("--my_token", type=str, default=CFG.my_token)
    return parser.parse_args()

def main():
    args = get_args()

    assert args.my_token is not None, "Please provide your Huggingface token! Learn how to get your token here: https://huggingface.co/docs/hub/security-tokens"
    if not args.my_token.strip():
        raise ValueError("Please provide your Huggingface token! Learn how to get your token here: https://huggingface.co/docs/hub/security-tokens")
    assert args.output_dir is not None
    if not args.output_dir.strip(): raise ValueError("Please specify an output directory.")
    os.path.isdir(args.output_dir) or os.makedirs(args.output_dir)
    assert os.path.exists(args.output_dir), "Output directory does not exist! Please provide a valid path."
    assert args.prompt is not None, "Please provide a prompt!"
    if not args.prompt.strip(): raise ValueError("Please provide a prompt!")
    assert args.num_images_per_batch > 0, "Please provide a positive number of images per batch to generate!"
    assert args.num_steps > 0, "Please provide a positive number of steps to take!"

    # get your token at https://huggingface.co/settings/tokens
    if args.use_low_mem == True:
        print("Using low memory model.")
        #pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=args.my_token)
    else:
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=args.my_token)

    if args.use_gpu:
        pipe.to("cuda")
    else:
        pipe.to("cpu")

    def image_grid(imgs, rows, cols):
        assert len(imgs) == rows*cols

        w, h = imgs[0].size
        grid = Image.new('RGB', size=(cols*w, rows*h))
        grid_w, grid_h = grid.size
        
        for i, img in enumerate(imgs):
            grid.paste(img, box=(i%cols*w, i//cols*h))
        return grid

    prompt = [args.prompt] * args.num_images_per_batch

    print("Start generating images...")
    for i in tqdm(range(args.num_steps)):

        images = pipe(prompt, height=args.height, width=args.width, num_inference_steps=args.num_inference_steps)["sample"]

        grid = image_grid(images, rows=1, cols=args.num_images_per_batch)

        # you can save the image with
        grid.save(f"../diffusion_images/image_{i+1}.png")

        print(f"Total images generated so far: {args.num_images_per_batch * (i + 1)}.")
    
    print("Finished generating images!")
    print("Total images generated: ", args.num_images_per_batch * args.num_steps)
    print("Images saved to folder: ", args.output_dir)

if __name__ == "__main__":
    main()