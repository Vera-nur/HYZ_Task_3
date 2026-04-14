import os
import csv
import json
import torch
import cv2

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
REF_DIR = os.path.join(BASE_DIR, "data", "reference")
REF_BANK_PATH = os.path.join(BASE_DIR, "outputs", "features", "reference_bank_aug.pt")

CSV_OUT = os.path.join(BASE_DIR, "outputs", "logs", "task3_frame_results_aug_orb_track.csv")
JSON_OUT = os.path.join(BASE_DIR, "outputs", "json", "task3_frame_results_aug_orb_track.jsonl")

os.makedirs(os.path.join(BASE_DIR, "outputs", "logs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "outputs", "json"), exist_ok=True)

# -----------------------------
# CONFIG
# -----------------------------
FRAME_LIMIT = None
RESIZE_W = 960
RESIZE_H = 540

GRID_ROWS = 3
GRID_COLS = 3

PATCH_SIZE = 224
PATCH_STRIDE = 112
BATCH_SIZE = 32

COARSE_THRESHOLD = 0.13
FINE_THRESHOLD = 0.16
ORB_MATCH_THRESHOLD = 8

COARSE_MARGIN_THRESHOLD = 0.01
FINE_MARGIN_THRESHOLD = 0.01

# tracking
REDETECT_INTERVAL = 10      # her 10 karede bir SEARCHING / LOST objeler için tekrar ara
TRACKER_MIN_W = 20
TRACKER_MIN_H = 20

# thermal referansı şimdilik dışarıda bırak
EXCLUDED_REFERENCES = {"Referans_Nesne_04.JPG"}


def create_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        return cv2.legacy.TrackerKCF_create()
    if hasattr(cv2, "TrackerMIL_create"):
        return cv2.TrackerMIL_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMIL_create"):
        return cv2.legacy.TrackerMIL_create()
    raise RuntimeError("Uygun OpenCV tracker bulunamadı")


def pil_to_bgr(image):
    import numpy as np
    img = np.array(image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))


def xywh_to_xyxy(box):
    x, y, w, h = box
    return (int(x), int(y), int(x + w), int(y + h))


def flatten_variant_scores_to_reference_scores(sim_matrix, variant_meta, num_references):
    V, G = sim_matrix.shape

    ref_best_scores = torch.full((num_references,), -999.0)
    ref_best_region_idx = torch.full((num_references,), -1, dtype=torch.long)
    ref_best_variant_idx = torch.full((num_references,), -1, dtype=torch.long)
    ref_second_best_scores = torch.full((num_references,), -999.0)

    for v in range(V):
        ref_id = variant_meta[v]["reference_id"]
        scores_v = sim_matrix[v]
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


def detect_reference_with_dino_orb(frame_img_pil, ref_idx, reference_names, reference_images, variant_meta, variant_features):
    """
    Tek referans için DINOv2 + ORB detection yapar.
    Başarılıysa detection dict döner, yoksa None.
    """
    ref_name = reference_names[ref_idx]

    # coarse
    regions = generate_grid_regions(frame_img_pil, rows=GRID_ROWS, cols=GRID_COLS)
    region_images = [r["image"] for r in regions]
    region_features = extract_global_features_batch(region_images)  # [G, D]

    # bu referansa ait varyant indexleri
    variant_indices = [i for i, meta in enumerate(variant_meta) if meta["reference_id"] == ref_idx]
    ref_variant_features = variant_features[variant_indices]   # [V_ref, D]

    sim_matrix = cosine_similarity_matrix(ref_variant_features, region_features)  # [V_ref, G]

    best_coarse = -999
    second_coarse = -999
    best_region_idx = -1
    best_variant_idx_local = -1

    for local_idx in range(sim_matrix.shape[0]):
        scores_v = sim_matrix[local_idx]
        best_score_v, best_region_v = torch.max(scores_v, dim=0)

        best_score_v = best_score_v.item()
        best_region_v = best_region_v.item()

        if best_score_v > best_coarse:
            second_coarse = best_coarse
            best_coarse = best_score_v
            best_region_idx = best_region_v
            best_variant_idx_local = local_idx
        elif best_score_v > second_coarse:
            second_coarse = best_score_v

    coarse_margin = best_coarse - second_coarse if second_coarse > -998 else best_coarse

    if not (
        best_region_idx >= 0
        and best_coarse >= COARSE_THRESHOLD
        and coarse_margin >= COARSE_MARGIN_THRESHOLD
    ):
        return None

    best_region = regions[best_region_idx]
    patches = generate_patches_in_region(
        best_region,
        patch_size=PATCH_SIZE,
        stride=PATCH_STRIDE
    )

    global_variant_idx = variant_indices[best_variant_idx_local]
    ref_variant_feature = variant_features[global_variant_idx]
    best_variant_name = variant_meta[global_variant_idx]["variant_name"]

    patch_scores_all = []

    for i in range(0, len(patches), BATCH_SIZE):
        batch = patches[i:i+BATCH_SIZE]
        batch_imgs = [p["image"] for p in batch]
        batch_feats = extract_global_features_batch(batch_imgs)

        scores = torch.matmul(batch_feats, ref_variant_feature).flatten()

        for j, p in enumerate(batch):
            s = scores[j].item()
            patch_scores_all.append((s, p))

    patch_scores_all = sorted(patch_scores_all, key=lambda x: x[0], reverse=True)

    if len(patch_scores_all) == 0:
        return None

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

    if not (
        best_fine >= FINE_THRESHOLD
        and fine_margin >= FINE_MARGIN_THRESHOLD
    ):
        return None

    ref_img = reference_images[ref_idx]
    cand_img = best_patch["image"]

    orb_good_matches, ref_kp, cand_kp = orb_match_score(ref_img, cand_img)

    if orb_good_matches < ORB_MATCH_THRESHOLD:
        return None

    return {
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
    }


# -----------------------------
# LOAD REFERENCE BANK
# -----------------------------
data = torch.load(REF_BANK_PATH)
reference_names = data["reference_names"]
variant_meta = data["variant_meta"]
variant_features = data["variant_features"]

reference_images = []
for ref_name in reference_names:
    ref_path = os.path.join(REF_DIR, ref_name)
    reference_images.append(load_image(ref_path))

num_references = len(reference_names)

# her referans için state
object_states = {}
for ref_idx, ref_name in enumerate(reference_names):
    if ref_name in EXCLUDED_REFERENCES:
        state = "SKIPPED"
    else:
        state = "SEARCHING"

    object_states[ref_idx] = {
        "reference_name": ref_name,
        "state": state,
        "tracker": None,
        "last_bbox": None,
        "last_seen_frame": -1,
        "last_score": None,
    }

# -----------------------------
# FRAMES
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
        "frame_name",
        "reference_name",
        "state",
        "top_left_x",
        "top_left_y",
        "bottom_right_x",
        "bottom_right_y",
        "coarse_score",
        "fine_score",
        "orb_good_matches",
        "detected"
    ])

    for frame_idx, frame_name in enumerate(frame_files):
        frame_path = os.path.join(FRAMES_DIR, frame_name)
        frame_img_pil = load_image(frame_path).resize((RESIZE_W, RESIZE_H))
        frame_img_bgr = pil_to_bgr(frame_img_pil)

        detected_objects = []

        # -----------------------------
        # 1) TRACK EXISTING OBJECTS
        # -----------------------------
        for ref_idx, st in object_states.items():
            if st["state"] != "TRACKING":
                continue

            tracker = st["tracker"]
            ok, bbox_xywh = tracker.update(frame_img_bgr)

            if ok:
                x, y, w, h = bbox_xywh
                if w >= TRACKER_MIN_W and h >= TRACKER_MIN_H:
                    x1, y1, x2, y2 = xywh_to_xyxy((x, y, w, h))
                    st["last_bbox"] = (x1, y1, x2, y2)
                    st["last_seen_frame"] = frame_idx

                    detected_objects.append({
                        "object_id": ref_idx,
                        "reference_name": st["reference_name"],
                        "variant_name": "TRACKER",
                        "top_left_x": int(x1),
                        "top_left_y": int(y1),
                        "bottom_right_x": int(x2),
                        "bottom_right_y": int(y2),
                        "coarse_score": "",
                        "coarse_margin": "",
                        "fine_score": "",
                        "fine_margin": "",
                        "orb_good_matches": ""
                    })
                else:
                    st["state"] = "LOST"
                    st["tracker"] = None
            else:
                st["state"] = "LOST"
                st["tracker"] = None

        # -----------------------------
        # 2) SEARCH / LOST OBJECTS
        # -----------------------------
        run_redetect = (frame_idx % REDETECT_INTERVAL == 0)

        for ref_idx, st in object_states.items():
            if st["state"] == "SKIPPED":
                continue

            # zaten tracker ile bulunduysa bu karede tekrar arama yapma
            already_detected = any(d["object_id"] == ref_idx for d in detected_objects)
            if already_detected:
                continue

            if st["state"] in ["SEARCHING", "LOST"] and run_redetect:
                det = detect_reference_with_dino_orb(
                    frame_img_pil,
                    ref_idx,
                    reference_names,
                    reference_images,
                    variant_meta,
                    variant_features
                )

                if det is not None:
                    detected_objects.append(det)

                    # tracker başlat
                    try:
                        tracker = create_tracker()
                        bbox_xywh = xyxy_to_xywh((
                            det["top_left_x"],
                            det["top_left_y"],
                            det["bottom_right_x"],
                            det["bottom_right_y"]
                        ))

                        tracker.init(frame_img_bgr, bbox_xywh)

                        st["tracker"] = tracker
                        st["last_bbox"] = (
                            det["top_left_x"],
                            det["top_left_y"],
                            det["bottom_right_x"],
                            det["bottom_right_y"]
                        )
                        st["last_seen_frame"] = frame_idx
                        st["last_score"] = det["fine_score"]
                        st["state"] = "TRACKING"
                    except Exception:
                        st["state"] = "LOST"
                        st["tracker"] = None

        # -----------------------------
        # 3) WRITE CSV ROWS
        # -----------------------------
        detected_by_ref = {d["object_id"]: d for d in detected_objects}

        for ref_idx, st in object_states.items():
            det = detected_by_ref.get(ref_idx, None)

            if det is not None:
                writer.writerow([
                    frame_name,
                    st["reference_name"],
                    st["state"],
                    det["top_left_x"],
                    det["top_left_y"],
                    det["bottom_right_x"],
                    det["bottom_right_y"],
                    det.get("coarse_score", ""),
                    det.get("fine_score", ""),
                    det.get("orb_good_matches", ""),
                    1
                ])
            else:
                writer.writerow([
                    frame_name,
                    st["reference_name"],
                    st["state"],
                    "", "", "", "",
                    "", "", "",
                    0
                ])

        # -----------------------------
        # 4) WRITE JSONL
        # -----------------------------
        frame_payload = {
            "frame_name": frame_name,
            "detected_undefined_objects": [
                {
                    "object_id": d["object_id"],
                    "top_left_x": d["top_left_x"],
                    "top_left_y": d["top_left_y"],
                    "bottom_right_x": d["bottom_right_x"],
                    "bottom_right_y": d["bottom_right_y"]
                }
                for d in detected_objects
            ]
        }
        json_f.write(json.dumps(frame_payload, ensure_ascii=False) + "\n")

        detected_names = [d["reference_name"] for d in detected_objects]
        print(
            f"{frame_idx+1}/{len(frame_files)} işlendi | "
            f"bulunan nesne sayısı: {len(detected_objects)} | "
            f"{detected_names}"
        )

print("CSV yazıldı:", CSV_OUT)
print("JSONL yazıldı:", JSON_OUT)