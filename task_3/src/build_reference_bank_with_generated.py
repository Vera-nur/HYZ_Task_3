import os
import re
import torch
from utils import load_image, extract_global_features_batch

BASE_DIR = "/Users/veranur/Desktop/task_3"
REF_DIR = os.path.join(BASE_DIR, "data", "reference")
GEN_DIR = os.path.join(BASE_DIR, "data", "reference_generated")
OUT_PATH = os.path.join(BASE_DIR, "outputs", "features", "reference_bank_with_generated.pt")

os.makedirs(os.path.join(BASE_DIR, "outputs", "features"), exist_ok=True)

# Ana referanslar
ref_files = sorted([
    f for f in os.listdir(REF_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
])

reference_names = ref_files[:]   # ana referans isimleri sabit liste
reference_id_map = {name: idx for idx, name in enumerate(reference_names)}

all_images = []
variant_meta = []

# 1) Orijinal referansları ekle
for ref_name in reference_names:
    path = os.path.join(REF_DIR, ref_name)
    img = load_image(path)

    all_images.append(img)
    variant_meta.append({
        "reference_id": reference_id_map[ref_name],
        "reference_name": ref_name,
        "variant_name": "orig",
        "variant_source": "original"
    })

# 2) Generated referansları ekle
if os.path.isdir(GEN_DIR):
    gen_files = sorted([
        f for f in os.listdir(GEN_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ])

    for gen_name in gen_files:
        # örnek isim: Referans_Nesne_05_gen_01.png
        m = re.match(r"^(Referans_Nesne_\d+\.(?:jpg|jpeg|png|bmp|webp))_?gen_\d+\.(?:jpg|jpeg|png|bmp|webp)$", gen_name, re.IGNORECASE)

        # daha pratik fallback:
        base_match = None
        for ref_name in reference_names:
            stem = os.path.splitext(ref_name)[0]
            if gen_name.startswith(stem + "_gen_"):
                base_match = ref_name
                break

        if base_match is None:
            print(f"[UYARI] Eşleştirilemeyen generated dosya atlandı: {gen_name}")
            continue

        path = os.path.join(GEN_DIR, gen_name)
        img = load_image(path)

        all_images.append(img)
        variant_meta.append({
            "reference_id": reference_id_map[base_match],
            "reference_name": base_match,
            "variant_name": gen_name,
            "variant_source": "generated"
        })

features = extract_global_features_batch(all_images)

torch.save({
    "reference_names": reference_names,
    "variant_meta": variant_meta,
    "variant_features": features
}, OUT_PATH)

print("Kaydedildi:", OUT_PATH)
print("Ana referans sayısı:", len(reference_names))
print("Toplam varyant sayısı:", len(variant_meta))