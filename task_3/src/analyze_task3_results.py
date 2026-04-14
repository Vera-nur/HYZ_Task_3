import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "/Users/veranur/Desktop/task_3"
CSV_PATH = os.path.join(BASE_DIR, "outputs", "logs", "task3_frame_results_aug.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

THERMAL_REF = "Referans_Nesne_04.JPG"  # gerekirse değiştir

df = pd.read_csv(CSV_PATH)

# Sayısal tipler
for col in ["coarse_score", "coarse_margin", "fine_score", "fine_margin", "detected"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("\n=== GENEL ÖZET ===")
print("Toplam satır:", len(df))
print("Toplam frame:", df["frame_name"].nunique())
print("Toplam referans:", df["reference_name"].nunique())
print("Toplam detected=1:", int(df["detected"].fillna(0).sum()))
print("Ortalama detected/frame:", df.groupby("frame_name")["detected"].sum().mean())

print("\n=== REFERANS BAZLI ÖZET ===")
per_ref = df.groupby("reference_name").agg(
    total_rows=("reference_name", "size"),
    detected_count=("detected", "sum"),
    detect_rate=("detected", "mean"),
    coarse_mean=("coarse_score", "mean"),
    fine_mean=("fine_score", "mean"),
).sort_values("detect_rate", ascending=False)
print(per_ref)
per_ref.to_csv(os.path.join(OUT_DIR, "per_reference_summary.csv"))

print("\n=== VARYANT BAZLI ÖZET (sadece detected=1) ===")
if "best_variant" in df.columns:
    variant_counts = df[df["detected"] == 1]["best_variant"].value_counts(dropna=False)
    print(variant_counts)
    variant_counts.to_csv(os.path.join(OUT_DIR, "variant_counts_detected.csv"))

print("\n=== FRAME BAZLI ÖZET ===")
per_frame = df.groupby("frame_name")["detected"].sum().describe()
print(per_frame)
per_frame.to_csv(os.path.join(OUT_DIR, "per_frame_detected_describe.csv"))

# Thermal hariç analiz
df_non_thermal = df[df["reference_name"] != THERMAL_REF].copy()

print("\n=== THERMAL HARİÇ ÖZET ===")
print("Toplam satır:", len(df_non_thermal))
print("Toplam frame:", df_non_thermal["frame_name"].nunique())
print("Toplam referans:", df_non_thermal["reference_name"].nunique())
print("Toplam detected=1:", int(df_non_thermal["detected"].fillna(0).sum()))
print("Ortalama detected/frame:", df_non_thermal.groupby("frame_name")["detected"].sum().mean())

per_ref_non = df_non_thermal.groupby("reference_name").agg(
    total_rows=("reference_name", "size"),
    detected_count=("detected", "sum"),
    detect_rate=("detected", "mean"),
    coarse_mean=("coarse_score", "mean"),
    fine_mean=("fine_score", "mean"),
).sort_values("detect_rate", ascending=False)
print("\n=== THERMAL HARİÇ REFERANS ÖZETİ ===")
print(per_ref_non)
per_ref_non.to_csv(os.path.join(OUT_DIR, "per_reference_summary_non_thermal.csv"))

# Görseller
# 1) Referans bazlı detection rate
plt.figure(figsize=(10, 5))
per_ref["detect_rate"].plot(kind="bar")
plt.title("Reference Detection Rate")
plt.ylabel("Detect Rate")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "reference_detection_rate.png"))
plt.close()

# 2) Thermal hariç referans bazlı detection rate
plt.figure(figsize=(10, 5))
per_ref_non["detect_rate"].plot(kind="bar")
plt.title("Reference Detection Rate (Non-Thermal)")
plt.ylabel("Detect Rate")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "reference_detection_rate_non_thermal.png"))
plt.close()

# 3) Her frame’de kaç referans bulundu?
per_frame_counts = df.groupby("frame_name")["detected"].sum()
plt.figure(figsize=(12, 4))
per_frame_counts.plot()
plt.title("Detected Object Count per Frame")
plt.ylabel("Detected Count")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "detected_count_per_frame.png"))
plt.close()

# 4) Fine score histogram
plt.figure(figsize=(8, 5))
df["fine_score"].dropna().hist(bins=50)
plt.title("Fine Score Distribution")
plt.xlabel("Fine Score")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fine_score_hist.png"))
plt.close()

# 5) Detected=1 olanların fine score histogramı
plt.figure(figsize=(8, 5))
df[df["detected"] == 1]["fine_score"].dropna().hist(bins=50)
plt.title("Fine Score Distribution (Detected=1)")
plt.xlabel("Fine Score")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fine_score_hist_detected.png"))
plt.close()

# 6) En yüksek fine score alan örnekler
top_all = df.sort_values("fine_score", ascending=False).head(50)
top_all.to_csv(os.path.join(OUT_DIR, "top50_by_fine_score.csv"), index=False)

top_non_thermal = df_non_thermal.sort_values("fine_score", ascending=False).head(50)
top_non_thermal.to_csv(os.path.join(OUT_DIR, "top50_by_fine_score_non_thermal.csv"), index=False)

print("\nAnaliz tamamlandı. Çıktılar:", OUT_DIR)