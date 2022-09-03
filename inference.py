my_token = ""

from diffusers import StableDiffusionPipeline
from PIL import Image

class CFG:
    height = 768
    width = 512
    num_images = 2

prompt = "dwarf knight portrait, highly detailed, d & d, fantasy, highly detailed, digital painting, trending on artstation, concept art, sharp focus, illustration, global illumination, ray tracing, realistic shaded, art by artgerm and greg rutkowski and fuji choko and viktoria gavrilenko and hoang lap"

# get your token at https://huggingface.co/settings/tokens
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=my_token)
pipe.to("cuda")

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

prompt = [prompt] * CFG.num_images

images = pipe(prompt, height=CFG.height, width=CFG.width)["sample"]

grid = image_grid(images, rows=1, cols=CFG.num_images)

# you can save the image with
grid.save(f"astronaut_rides_horse.png")
