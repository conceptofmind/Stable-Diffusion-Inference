my_token = ""

from diffusers import StableDiffusionPipeline
from PIL import Image
import os

class CFG:
    height = 768
    width = 512
    num_images = 2
    num_steps = 3
    use_gpu = True

prompt = "dwarf knight portrait, highly detailed, d & d, fantasy, highly detailed, digital painting, trending on artstation, concept art, sharp focus, illustration, global illumination, ray tracing, realistic shaded, art by artgerm and greg rutkowski and fuji choko and viktoria gavrilenko and hoang lap"

# get your token at https://huggingface.co/settings/tokens
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=my_token)

if CFG.use_gpu:
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

prompt = [prompt] * CFG.num_images

for i in range(CFG.num_steps):
    print(f"Generating images for step {i+1}")

    images = pipe(prompt, height=CFG.height, width=CFG.width)["sample"]

    grid = image_grid(images, rows=1, cols=CFG.num_images)

    # you can save the image with
    grid.save(f"./diffusion_images/test_images_{i+1}.png")

    print(f"Generated {CFG.num_images * (i + 1)} images.")
    print(f"Finished generating images.")
    print(f"Total images generated: {CFG.num_images * (i + 1)}.")
