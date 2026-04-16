import os
import torch
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import cv2
import numpy as np

from utils import load_image, extract_global_features_batch

#birinci nesne için çok iyi olmadı bunun üzerine biraz daha bakıp araştırılabilir.

BASE_DIR = "/Users/veranur/Documents/hyz_task_3/HYZ_Task_3/task_3"
REF_DIR = os.path.join(BASE_DIR, "data", "reference")
OUT_PATH = os.path.join(BASE_DIR, "outputs", "features", "reference_bank_aug_v2.pt")

os.makedirs(os.path.join(BASE_DIR, "outputs", "features"), exist_ok=True)


def ensure_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB")


def make_gray(img: Image.Image) -> Image.Image:
    return ImageOps.grayscale(img).convert("RGB")


def make_contrast(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(factor)


def make_blur(img: Image.Image, radius: float = 1.0) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def make_center_crop(img: Image.Image, crop_ratio: float = 0.85) -> Image.Image:
    w, h = img.size
    nw, nh = int(w * crop_ratio), int(h * crop_ratio)
    left = (w - nw) // 2
    top = (h - nh) // 2
    crop = img.crop((left, top, left + nw, top + nh))
    return crop.resize((w, h), Image.BICUBIC)


def make_zoom_in(img: Image.Image, crop_ratio: float = 0.82) -> Image.Image:
    # Nesne biraz daha yakından görünüyormuş gibi
    return make_center_crop(img, crop_ratio=crop_ratio)


def make_zoom_out(img: Image.Image, scale: float = 0.8) -> Image.Image:
    # Nesne daha uzaktan / daha genişten görünüyormuş gibi
    w, h = img.size
    nw, nh = int(w * scale), int(h * scale)

    small = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (w, h), (0, 0, 0))

    left = (w - nw) // 2
    top = (h - nh) // 2
    canvas.paste(small, (left, top))
    return canvas


def pil_to_cv_rgb(img: Image.Image):
    return np.array(img.convert("RGB"))


def cv_to_pil(img):
    return Image.fromarray(img.astype(np.uint8))


def make_perspective(img: Image.Image, mode: str = "top", strength: float = 0.08) -> Image.Image:
    """
    Hafif perspektif farkı.
    strength küçük tutuldu çünkü aşırı bozmak istemiyoruz.
    """
    src = pil_to_cv_rgb(img)
    h, w = src.shape[:2]

    src_pts = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ])

    margin_w = int(strength * w)
    margin_h = int(strength * h)

    if mode == "left":
        dst_pts = np.float32([
            [margin_w, 0],
            [w - 1, 0],
            [w - 1 - margin_w, h - 1],
            [0, h - 1]
        ])
    elif mode == "right":
        dst_pts = np.float32([
            [0, 0],
            [w - 1 - margin_w, 0],
            [w - 1, h - 1],
            [margin_w, h - 1]
        ])
    elif mode == "top":
        dst_pts = np.float32([
            [0, margin_h],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1 - margin_h]
        ])
    else:
        dst_pts = src_pts.copy()

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(
        src,
        M,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    return cv_to_pil(warped)


# Referansları tipe göre kategorileştiriyoruz
REFERENCE_CATEGORY = {
    "Referans_Nesne_01.jpeg": "vehicle",
    "Referans_Nesne_01.jpg": "vehicle",
    "Referans_Nesne_01.JPG": "vehicle",

    "Referans_Nesne_02.jpeg": "scene_green",
    "Referans_Nesne_02.jpg": "scene_green",
    "Referans_Nesne_02.JPG": "scene_green",

    "Referans_Nesne_03.jpeg": "thin_structure",
    "Referans_Nesne_03.jpg": "thin_structure",
    "Referans_Nesne_03.JPG": "thin_structure",

    "Referans_Nesne_05.jpeg": "thin_structure",
    "Referans_Nesne_05.jpg": "thin_structure",
    "Referans_Nesne_05.JPG": "thin_structure",

    "Referans_Nesne_06.jpeg": "planar_structure_low_angle",
    "Referans_Nesne_06.jpg": "planar_structure_low_angle",
    "Referans_Nesne_06.JPG": "planar_structure_low_angle",

    "Referans_Nesne_04.jpeg": "thermal",
    "Referans_Nesne_04.jpg": "thermal",
    "Referans_Nesne_04.JPG": "thermal",
}


def make_variants(img: Image.Image, category: str):
    img = ensure_rgb(img)
    variants = {}

    # Her zaman orijinal
    variants["orig"] = img

    # Herkese uygulanabilecek güvenli varyantlar
    variants["gray"] = make_gray(img)
    variants["contrast_up"] = make_contrast(img, 1.25)
    variants["contrast_down"] = make_contrast(img, 0.85)
    variants["blur_light"] = make_blur(img, radius=1.0)

    # Kategoriye özel varyantlar
    if category == "vehicle":
        variants["zoom_in"] = make_zoom_in(img, crop_ratio=0.84)

    elif category == "scene_green":
        # Çok yeşil alan içerdiği için fazla serbest bırakmıyoruz
        variants["center_crop"] = make_center_crop(img, crop_ratio=0.78)
        variants["zoom_in"] = make_zoom_in(img, crop_ratio=0.80)
        variants["perspective_top_soft"] = make_perspective(img, mode="top", strength=0.06)

    elif category == "thin_structure":
        # Goalpost gibi ince çizgisel yapılar
        variants["zoom_in"] = make_zoom_in(img, crop_ratio=0.84)
        variants["perspective_left_soft"] = make_perspective(img, mode="left", strength=0.06)
        variants["perspective_right_soft"] = make_perspective(img, mode="right", strength=0.06)

    elif category == "planar_structure_low_angle":
        # Referans 06 gibi alçaktan çekilmiş yapı.
        # Drone daha yukarıdan ve daha uzaktan göreceği için:
        variants["zoom_out"] = make_zoom_out(img, scale=0.82)
        variants["center_crop"] = make_center_crop(img, crop_ratio=0.82)
        variants["perspective_top_soft"] = make_perspective(img, mode="top", strength=0.06)

    elif category == "thermal":
        # Şimdilik çok agresif bir şey yapmıyoruz
        pass

    else:
        variants["center_crop"] = make_center_crop(img, crop_ratio=0.82)

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

    category = REFERENCE_CATEGORY.get(ref_name, "default")
    variants = make_variants(img, category)

    for variant_name, variant_img in variants.items():
        all_variant_images.append(variant_img)
        variant_meta.append({
            "reference_id": ref_idx,
            "reference_name": ref_name,
            "variant_name": variant_name,
            "category": category
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