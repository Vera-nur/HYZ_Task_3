import os
import csv
import json
import torch

from utils import (
    load_image,
    extract_global_features_batch,
    cosine_similarity_matrix,
    generate_grid_regions,
    generate_patches_in_region,
    orb_match_score
)

BASE_DIR = "/Users/veranur/Desktop/task_3"

FRAMES_DIR = os.path.join(BASE_DIR, "data", "frames")
REF_DIR = os.path.join(BASE_DIR, "data", "reference")   # senin mevcut klasör yapın buysa böyle kalsın
REF_BANK_PATH = os.path.join(BASE_DIR, "outputs", "features", "reference_bank_aug.pt")

CSV_OUT = os.path.join(BASE_DIR, "outputs", "logs", "task3_frame_results_aug_orb.csv")
JSON_OUT = os.path.join(BASE_DIR, "outputs", "json", "task3_frame_results_aug_orb.jsonl")

os.makedirs(os.path.join(BASE_DIR, "outputs", "logs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "outputs", "json"), exist_ok=True)

# -----------------------------
# CONFIG
# -----------------------------
FRAME_LIMIT = None      # test için 100 yapabilirsin
RESIZE_W = 960
RESIZE_H = 540

GRID_ROWS = 3
GRID_COLS = 3

PATCH_SIZE = 224
PATCH_STRIDE = 112
BATCH_SIZE = 32

# DINO eşikleri
COARSE_THRESHOLD = 0.13
FINE_THRESHOLD = 0.16

# margin: en iyi skor ile ikinci en iyi skor farkı
COARSE_MARGIN_THRESHOLD = 0.01
FINE_MARGIN_THRESHOLD = 0.01

# ORB doğrulama eşiği
ORB_MATCH_THRESHOLD = 8


def flatten_variant_scores_to_reference_scores(sim_matrix, variant_meta, num_references):
    """
    sim_matrix: [V, G]
    variant_meta: list of dict
    num_references: R

    Her referans için, varyantlar arasından en iyi region skorunu döndürür.
    Çıktı:
      ref_best_scores: [R]
      ref_best_region_idx: [R]
      ref_best_variant_idx: [R]
      ref_second_best_scores: [R]
    """
    V, G = sim_matrix.shape

    ref_best_scores = torch.full((num_references,), -999.0)
    ref_best_region_idx = torch.full((num_references,), -1, dtype=torch.long)
    ref_best_variant_idx = torch.full((num_references,), -1, dtype=torch.long)
    ref_second_best_scores = torch.full((num_references,), -999.0)

    for v in range(V):
        ref_id = variant_meta[v]["reference_id"]
        scores_v = sim_matrix[v]  # [G]
        best_score_v, best_region_v = torch.max(scores_v, dim=0)

        best_score_v = best_score_v.item()
        best_region_v = best_region_v.item()

        if best_score_v > ref_best_scores[ref_id]:
            ref_second_best_scores[ref_id] = ref_best_scores[ref_id]
            ref_best_scores[ref_id] = best_score_v
            ref_best_region_idx[ref_id] = best_region_v
            ref_best_variant_idx[ref_id] = v
        elif best_score_v > ref_second_best_scores[ref_id]:
            ref_second_best_scores[ref_id] = best_score_v

    return ref_best_scores, ref_best_region_idx, ref_best_variant_idx, ref_second_best_scores


# -----------------------------
# LOAD REFERENCE BANK
# -----------------------------
data = torch.load(REF_BANK_PATH)
reference_names = data["reference_names"]          # [R]
variant_meta = data["variant_meta"]                # len V
variant_features = data["variant_features"]        # [V, D]

num_references = len(reference_names)

# referans görüntülerini de yükle (ORB doğrulama için)
reference_images = []
for ref_name in reference_names:
    ref_path = os.path.join(REF_DIR, ref_name)
    reference_images.append(load_image(ref_path))

# -----------------------------
# LOAD FRAMES
# -----------------------------
frame_files = sorted([
    f for f in os.listdir(FRAMES_DIR)
    if f.lower().endswith(".jpg")
])

if FRAME_LIMIT is not None:
    frame_files = frame_files[:FRAME_LIMIT]

# -----------------------------
# MAIN LOOP
# -----------------------------
with open(CSV_OUT, "w", newline="", encoding="utf-8") as csv_f, \
     open(JSON_OUT, "w", encoding="utf-8") as json_f:

    writer = csv.writer(csv_f)
    writer.writerow([
        "frame_name", "reference_name", "best_variant",
        "coarse_score", "coarse_margin",
        "fine_score", "fine_margin",
        "left", "top", "right", "bottom",
        "orb_good_matches",
        "ref_keypoints",
        "cand_keypoints",
        "detected"
    ])

    for idx, frame_name in enumerate(frame_files):
        frame_path = os.path.join(FRAMES_DIR, frame_name)
        frame_img = load_image(frame_path).resize((RESIZE_W, RESIZE_H))

        # -----------------------------
        # 1) COARSE SEARCH
        # -----------------------------
        regions = generate_grid_regions(frame_img, rows=GRID_ROWS, cols=GRID_COLS)
        region_images = [r["image"] for r in regions]
        region_features = extract_global_features_batch(region_images)   # [G, D]

        # variant x region
        sim_matrix = cosine_similarity_matrix(variant_features, region_features)  # [V, G]

        (
            ref_best_scores,
            ref_best_region_idx,
            ref_best_variant_idx,
            ref_second_best_scores
        ) = flatten_variant_scores_to_reference_scores(
            sim_matrix, variant_meta, num_references
        )

        detected_objects = []

        for ref_idx, ref_name in enumerate(reference_names):
            best_coarse = ref_best_scores[ref_idx].item()
            second_coarse = ref_second_best_scores[ref_idx].item()
            coarse_margin = best_coarse - second_coarse if second_coarse > -998 else best_coarse

            best_region_idx = ref_best_region_idx[ref_idx].item()
            best_variant_idx = ref_best_variant_idx[ref_idx].item()

            best_variant_name = ""
            best_fine = -999
            fine_margin = -999
            best_box = None
            detected = 0

            orb_good_matches = 0
            ref_keypoints = 0
            cand_keypoints = 0

            if best_variant_idx >= 0:
                best_variant_name = variant_meta[best_variant_idx]["variant_name"]

            if (
                best_region_idx >= 0
                and best_coarse >= COARSE_THRESHOLD
                and coarse_margin >= COARSE_MARGIN_THRESHOLD
            ):
                best_region = regions[best_region_idx]
                patches = generate_patches_in_region(
                    best_region,
                    patch_size=PATCH_SIZE,
                    stride=PATCH_STRIDE
                )

                # seçilen referansın seçilen varyant feature'ı
                ref_variant_feature = variant_features[best_variant_idx]  # [D]

                patch_scores_all = []

                for i in range(0, len(patches), BATCH_SIZE):
                    batch = patches[i:i+BATCH_SIZE]
                    batch_imgs = [p["image"] for p in batch]
                    batch_feats = extract_global_features_batch(batch_imgs)  # [B, D]

                    scores = torch.matmul(batch_feats, ref_variant_feature)  # [B]
                    scores = scores.flatten()

                    for j, p in enumerate(batch):
                        s = scores[j].item()
                        patch_scores_all.append((s, p))

                patch_scores_all = sorted(patch_scores_all, key=lambda x: x[0], reverse=True)

                if len(patch_scores_all) > 0:
                    best_fine = patch_scores_all[0][0]
                    best_patch = patch_scores_all[0][1]

                    best_box = (
                        best_patch["left"],
                        best_patch["top"],
                        best_patch["right"],
                        best_patch["bottom"]
                    )

                    if len(patch_scores_all) > 1:
                        fine_margin = patch_scores_all[0][0] - patch_scores_all[1][0]
                    else:
                        fine_margin = best_fine

                    # -----------------------------
                    # 2) ORB VERIFICATION
                    # -----------------------------
                    if (
                        best_fine >= FINE_THRESHOLD
                        and fine_margin >= FINE_MARGIN_THRESHOLD
                    ):
                        ref_img = reference_images[ref_idx]
                        cand_img = best_patch["image"]

                        orb_good_matches, ref_keypoints, cand_keypoints = orb_match_score(
                            ref_img,
                            cand_img
                        )

                        if orb_good_matches >= ORB_MATCH_THRESHOLD:
                            detected = 1
                            detected_objects.append({
                                "object_id": ref_idx,
                                "reference_name": ref_name,
                                "variant_name": best_variant_name,
                                "top_left_x": int(best_box[0]),
                                "top_left_y": int(best_box[1]),
                                "bottom_right_x": int(best_box[2]),
                                "bottom_right_y": int(best_box[3]),
                                "coarse_score": float(best_coarse),
                                "coarse_margin": float(coarse_margin),
                                "fine_score": float(best_fine),
                                "fine_margin": float(fine_margin),
                                "orb_good_matches": int(orb_good_matches)
                            })

            writer.writerow([
                frame_name,
                ref_name,
                best_variant_name,
                float(best_coarse),
                float(coarse_margin),
                float(best_fine) if best_fine != -999 else "",
                float(fine_margin) if fine_margin != -999 else "",
                best_box[0] if best_box else "",
                best_box[1] if best_box else "",
                best_box[2] if best_box else "",
                best_box[3] if best_box else "",
                orb_good_matches,
                ref_keypoints,
                cand_keypoints,
                detected
            ])

        frame_payload = {
            "frame_name": frame_name,
            "detected_undefined_objects": detected_objects
        }
        json_f.write(json.dumps(frame_payload, ensure_ascii=False) + "\n")

        detected_names = [d["reference_name"] for d in detected_objects]
        print(
            f"{idx+1}/{len(frame_files)} işlendi | "
            f"bulunan nesne sayısı: {len(detected_objects)} | "
            f"{detected_names}"
        )

print("CSV yazıldı:", CSV_OUT)
print("JSONL yazıldı:", JSON_OUT)