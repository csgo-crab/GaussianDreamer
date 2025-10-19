import numpy as np
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from .point_cloud import predictions_to_pointcloud

from dataclasses import dataclass
import torch
from PIL import Image
from torchvision import transforms as TF


@dataclass
class VGGTConfig:
    """VGGT 点云生成参数"""

    images: np.ndarray = None
    model: VGGT = None
    device: str = "cuda"
    conf_thres: float = 50.0  # 过滤低置信度点
    mask_black_bg: bool = False  # 是否去除黑色背景
    mask_white_bg: bool = False  # 是否去除白色背景
    prediction_mode: str = (
        "Predicted Pointmap"  # 模式选择 ("Predicted Pointmap" / "Depthmap")
    )


class VGGTPipeline:
    def __init__(self, cfg: VGGTConfig):
        self.cfg = cfg

    def preprocess_images(self, mode="crop"):
        cfg = self.cfg
        # Check for empty list
        if len(cfg.images) == 0:
            raise ValueError("At least 1 image is required")
        # Validate mode
        if mode not in ["crop", "pad"]:
            raise ValueError("Mode must be either 'crop' or 'pad'")

        images = []
        shapes = set()
        to_tensor = TF.ToTensor()
        target_size = 518

        # First process all images and collect their shapes
        for img in cfg.images:
            width, height = img.size
            if mode == "pad":
                # Make the largest dimension 518px while maintaining aspect ratio
                if width >= height:
                    new_width = target_size
                    new_height = (
                        round(height * (new_width / width) / 14) * 14
                    )  # Make divisible by 14
                else:
                    new_height = target_size
                    new_width = (
                        round(width * (new_height / height) / 14) * 14
                    )  # Make divisible by 14
            else:  # mode == "crop"
                # Original behavior: set width to 518px
                new_width = target_size
                # Calculate height maintaining aspect ratio, divisible by 14
                new_height = round(height * (new_width / width) / 14) * 14

            # Resize with new dimensions (width, height)
            img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
            img = to_tensor(img)  # Convert to tensor (0, 1)

            # Center crop height if it's larger than 518 (only in crop mode)
            if mode == "crop" and new_height > target_size:
                start_y = (new_height - target_size) // 2
                img = img[:, start_y : start_y + target_size, :]

            # For pad mode, pad to make a square of target_size x target_size
            if mode == "pad":
                h_padding = target_size - img.shape[1]
                w_padding = target_size - img.shape[2]

                if h_padding > 0 or w_padding > 0:
                    pad_top = h_padding // 2
                    pad_bottom = h_padding - pad_top
                    pad_left = w_padding // 2
                    pad_right = w_padding - pad_left

                    # Pad with white (value=1.0)
                    img = torch.nn.functional.pad(
                        img,
                        (pad_left, pad_right, pad_top, pad_bottom),
                        mode="constant",
                        value=1.0,
                    )

            shapes.add((img.shape[1], img.shape[2]))
            images.append(img)

        # Check if we have different shapes
        # In theory our model can also work well with different shapes
        if len(shapes) > 1:
            print(f"Warning: Found images with different shapes: {shapes}")
            # Find maximum dimensions
            max_height = max(shape[0] for shape in shapes)
            max_width = max(shape[1] for shape in shapes)

            # Pad images if necessary
            padded_images = []
            for img in images:
                h_padding = max_height - img.shape[1]
                w_padding = max_width - img.shape[2]

                if h_padding > 0 or w_padding > 0:
                    pad_top = h_padding // 2
                    pad_bottom = h_padding - pad_top
                    pad_left = w_padding // 2
                    pad_right = w_padding - pad_left

                    img = torch.nn.functional.pad(
                        img,
                        (pad_left, pad_right, pad_top, pad_bottom),
                        mode="constant",
                        value=1.0,
                    )
                padded_images.append(img)
            images = padded_images

        images = torch.stack(images)  # concatenate images

        # Ensure correct shape when single image
        if len(cfg.images) == 1:
            # Verify shape is (1, C, H, W)
            if images.dim() == 3:
                images = images.unsqueeze(0)
        return images

    def run(self):
        """
        一键运行 VGGT 推理并生成点云 (coords, rgb)。

        Args:
            images (np.ndarray or torch.Tensor): 多视角图片，形状 (N, H, W, 3) 或 (N, 3, H, W)
            model: 加载好的 VGGT 模型
            config (VGGTPointCloudConfig): 推理与点云生成配置
        Returns:
            coords (np.ndarray): 点云坐标 (N_points, 3)
            rgb (np.ndarray): 颜色信息 (N_points, 3)，uint8
        """
        cfg = self.cfg
        model = cfg.model
        model.eval()
        print(f"🚀 开始 VGGT 推理，输入图片数量: {len(cfg.images)}，设备: {cfg.device}")

        # ---- Step 1: 图片预处理 ----
        images_tensor = self.preprocess_images().to(cfg.device)
        print(
            f"预处理后图片形状: {images_tensor.shape} (batch, channels, height, width)"
        )

        # ---- Step 2: 模型推理 ----
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images_tensor)

        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], images_tensor.shape[-2:]
        )
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        # ---- Step 4: 转 numpy ----
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = (
                    predictions[key].cpu().numpy().squeeze(0)
                )  # 移除batch维度

        # ---- Step 5: 深度图 → 点云 ----
        depth_map = predictions["depth"]
        world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
        predictions["world_points_from_depth"] = world_points

        # ---- Step 6: 转换为点云 (coords, rgb) ----
        coords, rgb = predictions_to_pointcloud(
            predictions,
            conf_thres=cfg.conf_thres,
            mask_black_bg=cfg.mask_black_bg,
            mask_white_bg=cfg.mask_white_bg,
            prediction_mode=cfg.prediction_mode,
        )

        torch.cuda.empty_cache()
        print(f"✅ VGGT 点云生成完成，共 {coords.shape[0]} 个点")
        return coords, rgb
