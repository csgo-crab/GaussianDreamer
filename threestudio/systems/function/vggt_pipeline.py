import numpy as np
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

from dataclasses import dataclass
import torch
from PIL import Image
from torchvision import transforms as TF


@dataclass
class VGGTPipelineConfig:
    """VGGT 点云生成参数"""
    model_path: str = "facebook/VGGT-1B"
    device: str = "cuda"
    dtype: str = "float16"  # Changed to float16 to match model weights
    conf_thres: float = 50.0  # 过滤低置信度点
    mask_black_bg: bool = False  # 是否去除黑色背景
    mask_white_bg: bool = False  # 是否去除白色背景
    prediction_mode: str = (
        "Predicted Pointmap"  # 模式选择 ("Predicted Pointmap" / "Depthmap")
    )


class VGGTPipeline:
    def __init__(self, cfg: VGGTPipelineConfig):
        self.cfg = cfg
        # Convert string dtype to torch.dtype
        self.dtype = getattr(torch, cfg.dtype) if hasattr(torch, cfg.dtype) else torch.float16
        self.model = VGGT.from_pretrained(cfg.model_path).to(cfg.device)
        
    def __del__(self):
        del self.model
        torch.cuda.empty_cache()

    def preprocess_images(self, images: np.ndarray = None, mode="crop"):
        cfg = self.cfg
        # Check for empty list
        if len(images) == 0:
            raise ValueError("At least 1 image is required")
        # Validate mode
        if mode not in ["crop", "pad"]:
            raise ValueError("Mode must be either 'crop' or 'pad'")

        result_images = []
        shapes = set()
        to_tensor = TF.ToTensor()
        target_size = 518

        # First process all images and collect their shapes
        for img in images:
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
            result_images.append(img)

        # Check if we have different shapes
        # In theory our model can also work well with different shapes
        if len(shapes) > 1:
            print(f"Warning: Found images with different shapes: {shapes}")
            # Find maximum dimensions
            max_height = max(shape[0] for shape in shapes)
            max_width = max(shape[1] for shape in shapes)

            # Pad images if necessary
            padded_images = []
            for img in result_images:
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
            result_images = padded_images
        result_images = torch.stack(result_images)  # concatenate images

        # Ensure correct shape when single image
        if len(images) == 1:
            # Verify shape is (1, C, H, W)
            if result_images.dim() == 3:
                result_images = result_images.unsqueeze(0)
        return result_images

    def run(self, images: np.ndarray = None):
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
        self.model.eval()
        print(f"🚀 开始 VGGT 推理，输入图片数量: {len(images)}，设备: {cfg.device}")

        # ---- Step 1: 图片预处理 ----
        images_tensor = self.preprocess_images(images).to(cfg.device)
        print(
            f"预处理后图片形状: {images_tensor.shape} (batch, channels, height, width)"
        )

        # ---- Step 2: 模型推理 ----
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                predictions = self.model(images_tensor)

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
        coords, rgb = self._predictions_to_pointcloud(predictions)

        torch.cuda.empty_cache()
        print(f"✅ VGGT 点云生成完成，共 {coords.shape[0]} 个点")
        return coords, rgb
    

    def _predictions_to_pointcloud(self, predictions):
        """提取预测点云的通用函数"""

        cfg = self.cfg
        if not isinstance(predictions, dict):
            raise ValueError("predictions 必须是一个字典")

        if "Pointmap" in cfg.prediction_mode and "world_points" in predictions:
            world_points = predictions["world_points"]
            conf = predictions.get("world_points_conf", np.ones_like(world_points[..., 0]))
        else:
            world_points = predictions["world_points_from_depth"]
            conf = predictions.get("depth_conf", np.ones_like(world_points[..., 0]))

        images = predictions["images"]

        if images.ndim == 4 and images.shape[1] == 3:
            images = np.transpose(images, (0, 2, 3, 1))

        coords = world_points.reshape(-1, 3)
        coords[..., 1:] *= -1
        
        rgb = (images.reshape(-1, 3) * 255).astype(np.uint8)
        conf = conf.reshape(-1)

        conf_threshold = np.percentile(conf, cfg.conf_thres) if cfg.conf_thres > 0 else 0.0
        mask = (conf >= conf_threshold) & (conf > 1e-5)

        if cfg.mask_black_bg:
            black_mask = rgb.sum(axis=1) >= 16
            mask &= black_mask
        if cfg.mask_white_bg:
            white_mask = ~((rgb[:, 0] > 240) & (rgb[:, 1] > 240) & (rgb[:, 2] > 240))
            mask &= white_mask

        coords = coords[mask]
        rgb = rgb[mask]

        if coords.size == 0:
            coords = np.array([[0.0, 0.0, 0.0]])
            rgb = np.array([[255, 255, 255]], dtype=np.uint8)
        return coords, rgb