import os
from PIL import Image
import torch
import clip
import random

def paste(canvas, image, index, width, height):
    """
    Paste a given image onto a canvas at a specific index in a 3x3 grid.
    """
    x = (index % 3) * width
    y = (index // 3) * height
    canvas.paste(image, box=(x, y))
    return canvas

def load_images(data_dir):
    """
    Load all images from a directory.
    """
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images = {}
    for file in image_files:
        try:
            image_path = os.path.join(data_dir, file)
            images[image_path] = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {file}: {e}")
    return images

def load_list_images(file_paths, crop=False, width=None, height=None):
    """
    Load a list of image file paths, optionally cropping to the center.
    Returns a dictionary of images and a combined output image.
    """
    images = {}
    output = Image.new('RGB', (3 * width, 2 * height), (0, 0, 0))
    for index, file in enumerate(file_paths):
        try:
            image = Image.open(file).convert("RGB")
            if crop:
                W, H = image.size
                image = image.crop((W/2 - width/2, H/2 - height/2, W/2 + width/2, H/2 + height/2))
            images[os.path.basename(file).split('.')[0]] = image
            if crop:
                output = paste(output, image, index, width, height)
        except Exception as e:
            print(f"Error loading image {file}: {e}")
    return images, output

def search_images(elements, data_dir, random_return=False):
    """
    Use CLIP to search for images matching given textual elements.
    Returns a dict of top-9 image matches per element and their paths.
    """
    if random_return:
        images = load_images(data_dir)
        return [images[k] for k in random.sample(list(images.keys()), 9)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", download_root="sentence-transformers/clip-ViT-B-32", device=device)

    images = load_images(data_dir)
    preprocessed_images = {file: preprocess(img).unsqueeze(0) for file, img in images.items()}

    text_tokens = clip.tokenize(elements).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    image_features = {}
    with torch.no_grad():
        for file, tensor in preprocessed_images.items():
            feat = model.encode_image(tensor.to(device))
            feat /= feat.norm(dim=-1, keepdim=True)
            image_features[file] = feat

    results = {}
    image_dirs = {}
    used_files = set()
    for i, element in enumerate(elements):
        sims = {file: (feat @ text_features[i].T).item() for file, feat in image_features.items() if file not in used_files}
        top_files = sorted(sims, key=sims.get, reverse=True)[:9]
        results[element] = [images[f] for f in top_files]
        image_dirs[element] = top_files
        used_files.update(top_files[:3])

    return results, image_dirs

def evaluate_images(prompt, reference_images, data_dir, width, height):
    """
    Evaluate candidate images against the prompt and reference images using CLIP.
    Returns the best image, grid of candidates, and a similarity score dict.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", download_root="sentence-transformers/clip-ViT-B-32", device=device)

    images, grid = load_list_images(data_dir, crop=True, width=width, height=height)
    image_tensors = {f: preprocess(img).unsqueeze(0) for f, img in images.items()}
    ref_tensors = {i: preprocess(img).unsqueeze(0) for i, img in enumerate(reference_images)}

    text_tokens = clip.tokenize(prompt).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(text_tokens)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)

    scores = {}
    with torch.no_grad():
        ref_feats = {i: model.encode_image(t.to(device)) / model.encode_image(t.to(device)).norm(dim=-1, keepdim=True) for i, t in ref_tensors.items()}
        for file, tensor in image_tensors.items():
            img_feat = model.encode_image(tensor.to(device))
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            text_score = (img_feat @ text_feat.T).item()
            ref_score = sum((img_feat @ rf.T).item() for rf in ref_feats.values()) / len(ref_feats)
            scores[file] = {"text": text_score, "image": ref_score}

    best_file = max(scores, key=lambda x: scores[x]['text'])
    return images[best_file], grid, scores