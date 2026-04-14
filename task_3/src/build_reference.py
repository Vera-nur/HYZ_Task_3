import os
import torch
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

from utils import load_image, extract_global_features_batch

BASE_DIR = "/Users/veranur/Desktop/task_3"
REF_DIR = os.path.join(BASE_DIR, "data", "reference")
OUT_PATH = os.path.join(BASE_DIR, "outputs", "features", "reference_bank_aug.pt")

os.makedirs(os.path.join(BASE_DIR, "outputs", "features"), exist_ok=True)


def ensure_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB")


def make_edge_rgb(img: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(img)
    edge = gray.filter(ImageFilter.FIND_EDGES)
    return edge.convert("RGB")


def make_variants(img: Image.Image):
    img = ensure_rgb(img)
    variants = {}

    variants["orig"] = img

    gray = ImageOps.grayscale(img).convert("RGB")
    variants["gray"] = gray

    contrast_up = ImageEnhance.Contrast(img).enhance(1.5)
    variants["contrast_up"] = contrast_up

    sharp_up = ImageEnhance.Sharpness(img).enhance(1.8)
    variants["sharp_up"] = sharp_up

    blur = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    variants["blur"] = blur

    edge = make_edge_rgb(img)
    variants["edge"] = edge

    rot_p15 = img.rotate(15, expand=True, fillcolor=(0, 0, 0)).resize(img.size)
    variants["rot_p15"] = rot_p15

    rot_m15 = img.rotate(-15, expand=True, fillcolor=(0, 0, 0)).resize(img.size)
    variants["rot_m15"] = rot_m15

    # hafif zoom-crop
    w, h = img.size
    crop = img.crop((int(0.05*w), int(0.05*h), int(0.95*w), int(0.95*h))).resize((w, h))
    variants["zoom_crop"] = crop

    return variants


ref_files = sorted([
    f for f in os.listdir(REF_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
])

all_variant_images = []
variant_meta = []

for ref_idx, ref_name in enumerate(ref_files):
    ref_path = os.path.join(REF_DIR, ref_name)
    img = load_image(ref_path)

    variants = make_variants(img)

    for variant_name, variant_img in variants.items():
        all_variant_images.append(variant_img)
        variant_meta.append({
            "reference_id": ref_idx,
            "reference_name": ref_name,
            "variant_name": variant_name
        })

print(f"Toplam referans sayısı: {len(ref_files)}")
print(f"Toplam varyant sayısı: {len(all_variant_images)}")

features = extract_global_features_batch(all_variant_images)

torch.save({
    "reference_names": ref_files,
    "variant_meta": variant_meta,
    "variant_features": features
}, OUT_PATH)

print("Kaydedildi:", OUT_PATH)