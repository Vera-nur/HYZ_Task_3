import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import cv2
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "facebook/dinov2-base"

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


def load_image(path):
    return Image.open(path).convert("RGB")


def extract_global_features_batch(images):
    inputs = processor(images=images, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)

    feats = outputs.last_hidden_state[:, 0, :]
    feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats.cpu()


def cosine_similarity_matrix(reference_features, query_features):
    return torch.matmul(reference_features, query_features.T)


def generate_grid_regions(image, rows=3, cols=3):
    w, h = image.size
    regions = []

    cell_w = w // cols
    cell_h = h // rows

    for r in range(rows):
        for c in range(cols):
            left = c * cell_w
            top = r * cell_h
            right = w if c == cols - 1 else (c + 1) * cell_w
            bottom = h if r == rows - 1 else (r + 1) * cell_h

            crop = image.crop((left, top, right, bottom))
            regions.append({
                "image": crop,
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom
            })
    return regions


def generate_patches_in_region(region, patch_size=224, stride=112):
    region_img = region["image"]
    rw, rh = region_img.size
    patches = []

    if rw < patch_size or rh < patch_size:
        patches.append({
            "image": region_img.resize((patch_size, patch_size)),
            "left": region["left"],
            "top": region["top"],
            "right": region["right"],
            "bottom": region["bottom"]
        })
        return patches

    for top in range(0, rh - patch_size + 1, stride):
        for left in range(0, rw - patch_size + 1, stride):
            crop = region_img.crop((left, top, left + patch_size, top + patch_size))
            patches.append({
                "image": crop,
                "left": region["left"] + left,
                "top": region["top"] + top,
                "right": region["left"] + left + patch_size,
                "bottom": region["top"] + top + patch_size
            })

    return patches

def pil_to_cv_gray(image: Image.Image):
    img = np.array(image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def orb_match_score(ref_image: Image.Image, cand_image: Image.Image, max_features=500):
    """
    Referans ve aday görüntü arasında ORB tabanlı eşleşme skoru döndürür.
    Basit skor: iyi eşleşme sayısı
    """

    ref_gray = pil_to_cv_gray(ref_image)
    cand_gray = pil_to_cv_gray(cand_image)

    orb = cv2.ORB_create(nfeatures=max_features)

    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(cand_gray, None)

    if des1 is None or des2 is None:
        return 0, 0, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) == 0:
        return 0, 0, 0

    matches = sorted(matches, key=lambda x: x.distance)

    # distance küçük olanlar daha iyi
    good_matches = [m for m in matches if m.distance < 50]

    return len(good_matches), len(kp1), len(kp2)

def pil_to_torch_image(image, device):
    """
    PIL RGB -> torch tensor [1,3,H,W], float32, 0..1
    """
    import torch

    arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device)
    return tensor