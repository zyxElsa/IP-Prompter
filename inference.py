import torch
from diffusers.pipelines.flux.pipeline_flux_fill import FluxFillPipeline
from diffusers.utils import load_image
from PIL import Image
import os
import random
from itertools import permutations

from search_images import search_images, evaluate_images
from prompts_list import prompt_content_list, elements_list

def generate_permutations(seq_len):
    """
    Generate all permutations of indices for a given length.
    """
    return [list(p) for p in permutations(range(seq_len))]

def center_crop(image, width, height):
    """
    Resize and crop the image to the desired width and height, centered.
    """
    W, H = image.size
    scale = max(width / min(W, H), height / min(W, H))
    image = image.resize((int(W * scale), int(H * scale)))
    W, H = image.size
    return image.crop((W/2 - width/2, H/2 - height/2, W/2 + width/2, H/2 + height/2))

def paste_high2low(canvas, image, index, width, height):
    """
    Paste an image onto the canvas in a 3x3 grid using high-to-low layout.
    """
    positions = [
        (width, 0), (0, 0), (width*2, 0),
        (0, height), (width*2, height), (width, height),
        (width, height*2), (0, height*2), (width*2, height*2)
    ]
    canvas.paste(image, box=positions[index])
    return canvas

def paste_specific(canvas, image, index, width, height):
    """
    Paste user-specific images at fixed positions.
    """
    positions = [(width, 0), (width, height*2), (0, height), (width*2, height)]
    canvas.paste(image, box=positions[index])
    return canvas

# Image size configuration
width, height = 512, 512

# Load reference style images
style_images = []
data_dir = "./images/tintin"
for image_name in os.listdir(data_dir)[:8]:
    image = load_image(os.path.join(data_dir, image_name))
    style_images.append(center_crop(image, width, height))

# Load pipeline
pipe = FluxFillPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Fill-dev",
    torch_dtype=torch.bfloat16
).to("cuda")

# Setup
seed = 0
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

user_specific_image_names = []
is_user_specific = len(user_specific_image_names) > 0
user_specific_images = []
if is_user_specific:
    for name in user_specific_image_names:
        image_path = os.path.join(data_dir, name)
        user_specific_images.append(Image.open(image_path).convert("RGB"))

standard_prompt = 'A diptych with images sharing the same art style. high-quality. '
all_permutations = generate_permutations(3)

# Iterate over prompts and elements
for frame_index, (prompt_content, elements) in enumerate(zip(prompt_content_list, elements_list)):
    images_dict, image_dirs = search_images(elements, data_dir)
    canvas_size = (width * 3, height * 3)
    candidates = []

    for permutation in all_permutations:
        concat_image = Image.new('RGB', canvas_size)
        reordered_elements = [elements[i] for i in permutation]
        permutation_name = "{" + "+".join(" ".join(e.split()[:3]) for e in reordered_elements) + "}"
        reference_images = []

        position_index = 0
        if isinstance(images_dict, list):
            reference_images = []
            for image in images_dict:
                image = center_crop(image, width, height)
                concat_image = paste_high2low(concat_image, image, position_index, width, height)
                position_index += 1
                reference_images.append(image)
        else:
            reference_images = []
            for element in reordered_elements:
                while len(images_dict[element]) < 3:
                    images_dict[element] = images_dict[element] + images_dict[element]
                for image in images_dict[element][:3]:
                    image = center_crop(image, width, height)
                    concat_image = paste_high2low(concat_image, image, position_index, width, height)
                    position_index += 1
                    reference_images.append(image)

        if is_user_specific:
            for idx, user_img in enumerate(user_specific_images):
                concat_image = paste_specific(concat_image, user_img, idx, width, height)

        concat_image.save('condition.jpg')

        # Build mask
        mask = Image.new('RGB', canvas_size, (0, 0, 0))
        mask_tile = Image.new('RGB', (width, height), (255, 255, 255))
        mask.paste(mask_tile, box=(width, height))
        mask.save('mask.jpg')

        prompt = standard_prompt + prompt_content
        seed = random.randint(0, 1000)
        result = pipe(
            prompt=prompt,
            image=concat_image,
            mask_image=mask,
            height=canvas_size[1],
            width=canvas_size[0],
            guidance_scale=30,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]

        safe_name = prompt_content[:150].replace('/', '_').replace(' ', '_')
        output_path = os.path.join(output_dir, f"{frame_index:02d}-{permutation_name}-{safe_name}-{seed}.png")
        result.save(output_path)
        candidates.append(output_path)

    results, output_grid, top_matches = evaluate_images(prompt_content, reference_images, candidates, width, height)
    print(f"Top match for frame {frame_index}: {top_matches}")

    # Save result and grid
    base_name = f"{frame_index:02d}-{safe_name}-{seed}"
    results.save(os.path.join(output_dir, f"{base_name}-best.png"))
    output_grid.save(os.path.join(output_dir, f"{base_name}-grid.png"))