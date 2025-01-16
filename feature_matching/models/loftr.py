import kornia as K
import kornia.feature as KF
from kornia_moons.viz import draw_LAF_matches
from kornia.feature import LoFTR

import cv2
import torch

class LoFTRModel:

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.matcher = LoFTR(pretrained="outdoor").to(self.device)

    def forward(self, image1, image2):

        img1 = K.io.load_image(image1, K.io.ImageLoadType.RGB32)[None, ...]
        img2 = K.io.load_image(image2, K.io.ImageLoadType.RGB32)[None, ...]

        img1 = K.geometry.resize(img1, (512, 512), antialias=True).to(self.device)
        img2 = K.geometry.resize(img2, (512, 512), antialias=True).to(self.device)

        input_dict = {
            "image0": K.color.rgb_to_grayscale(img1),
            "image1": K.color.rgb_to_grayscale(img2),
        }

        with torch.inference_mode():
            correspondences = self.matcher(input_dict)

        mkpts0 = correspondences["keypoints0"].cpu().numpy()
        mkpts1 = correspondences["keypoints1"].cpu().numpy()
        Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
        inliers = inliers > 0

        return mkpts0, mkpts1, Fm, inliers


    def viz(self, mkpts0, mkpts1, inliers):

        draw_LAF_matches(
            KF.laf_from_center_scale_ori(
                torch.from_numpy(mkpts0).view(1, -1, 2),
                torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
                torch.ones(mkpts0.shape[0]).view(1, -1, 1),
            ),
            KF.laf_from_center_scale_ori(
                torch.from_numpy(mkpts1).view(1, -1, 2),
                torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
                torch.ones(mkpts1.shape[0]).view(1, -1, 1),
            ),
            torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
            K.tensor_to_image(img1),
            K.tensor_to_image(img2),
            inliers,
            draw_dict={
                "inlier_color": (0.1, 1, 0.1, 0.5),
                "tentative_color": None,
                "feature_color": (0.2, 0.2, 1, 0.5),
                "vertical": False,
            },
        )