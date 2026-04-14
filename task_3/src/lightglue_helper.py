import torch
from lightglue import LightGlue, SuperPoint
from utils import pil_to_torch_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

extractor = SuperPoint(max_num_keypoints=1024).eval().to(DEVICE)
matcher = LightGlue(features="superpoint").eval().to(DEVICE)


def lightglue_match_score(ref_image, cand_image):
    """
    ref_image, cand_image: PIL Image
    çıktı:
      match_count, keypoints0_count, keypoints1_count
    """
    with torch.no_grad():
        image0 = pil_to_torch_image(ref_image, DEVICE)
        image1 = pil_to_torch_image(cand_image, DEVICE)

        feats0 = extractor.extract(image0)
        feats1 = extractor.extract(image1)

        matches01 = matcher({"image0": feats0, "image1": feats1})

        # batch boyutu 1 olduğu için [0] ile alıyoruz
        keypoints0 = feats0["keypoints"][0]
        keypoints1 = feats1["keypoints"][0]
        matches = matches01["matches"][0]

        match_count = int(matches.shape[0])
        kp0_count = int(keypoints0.shape[0])
        kp1_count = int(keypoints1.shape[0])

    return match_count, kp0_count, kp1_count