from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import importlib.util
import argparse

assert importlib.util.find_spec("diffusers") is not None, "pip install diffusers==0.2.4"
assert importlib.util.find_spec("transformers") is not None, "pip install transformers"
assert importlib.util.find_spec("scipy") is not None, "pip install scipy"
assert importlib.util.find_spec("ftfy") is not None, "pip install ftfy"

class CFG:
    prompt = "A photo of a cat"
    height = 768
    width = 512
    num_images = 1
    num_steps = 1
    use_gpu = True
    output_dir = "../diffusion_images/"
    my_token = None

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=CFG.prompt)
    parser.add_argument("--height", type=int, default=CFG.height)
    parser.add_argument("--width", type=int, default=CFG.width)
    parser.add_argument("--num_images", type=int, default=CFG.num_images)
    parser.add_argument("--num_steps", type=int, default=CFG.num_steps)
    parser.add_argument("--use_gpu", type=bool, default=CFG.use_gpu)
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
    assert args.num_images > 0, "Please provide a positive number of images to generate!"
    assert args.num_steps > 0, "Please provide a positive number of steps to take!"

    # get your token at https://huggingface.co/settings/tokens
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

    prompt = [args.prompt] * args.num_images

    for i in range(args.num_steps):
        print(f"Generating images for step {i+1}.")

        images = pipe(prompt, height=args.height, width=args.width)["sample"]

        grid = image_grid(images, rows=1, cols=args.num_images)

        # you can save the image with
        grid.save(f"../diffusion_images/images_{i+1}.png")

        print(f"Generated {args.num_images * (i + 1)} images.")
        print(f"Finished generating images.")
        print(f"Total images generated: {args.num_images * (i + 1)}.")

if __name__ == "__main__":
    main()