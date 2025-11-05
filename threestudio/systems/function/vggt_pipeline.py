import numpy as np
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

from dataclasses import dataclass
import torch
from PIL import Image
from torchvision import transforms as TF
from scipy.spatial import cKDTree

from .fusion_point import fusion_point_clouds


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
    multi_reference: bool = True  # 是否使用多参考帧



class VGGTPipeline:
    def __init__(self, cfg: VGGTPipelineConfig):
        self.cfg = cfg
        # Convert string dtype to torch.dtype
        self.dtype = getattr(torch, cfg.dtype) if hasattr(torch, cfg.dtype) else torch.float16
        self.model = VGGT.from_pretrained(cfg.model_path).to(cfg.device)


        #生成 extrinsics
        self.extrinsics = self.create_extrinsics_from_angles([0, 45, 90, 180, 270, 315], radius=1.0, device="cuda")


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
    
    def create_extrinsics_from_angles(self,angles_deg, radius=1.0, device="cuda"):
        """
        根据绕Y轴的角度生成相机外参矩阵 extrinsics: [1, S, 4, 4]
        angles_deg: list 或 numpy 数组，例如 [0, 45, 90, 180, 270, 315]
        radius: 相机到原点的距离
        """
        angles = torch.tensor(angles_deg, dtype=torch.float32, device=device)
        angles_rad = torch.deg2rad(angles)

        S = len(angles)
        extrinsics = torch.eye(4, device=device).repeat(1, S, 1, 1)  # [1,S,4,4]

        for i, theta in enumerate(angles_rad):
            # 绕 Y 轴旋转矩阵（相机绕物体）
            R = torch.tensor([
                [torch.cos(theta), 0, torch.sin(theta)],
                [0, 1, 0],
                [-torch.sin(theta), 0, torch.cos(theta)]
            ], device=device)

            # 平移：相机在圆周上，面向原点
            t = torch.tensor([
                radius * torch.sin(theta),
                0.0,
                radius * torch.cos(theta)
            ], device=device)

            extrinsics[0, i, :3, :3] = R.T     # 注意是从世界到相机 → 取转置
            extrinsics[0, i, :3, 3] = -R.T @ t # 相机中心位置转为外参形式

        return extrinsics  # [1,S,4,4]

    def run(self, images: np.ndarray = None):
        """
        一键运行 VGGT 推理并生成点云 (coords, rgb)。

        支持两种模式：
        - 单次推理模式（cfg.multi_reference=False）：一次性输入所有视角
        - 多参考视角模式（cfg.multi_reference=True）：每张图片依次作为参考视角，独立推理
        """
        cfg = self.cfg
        self.model.eval()
        print(f"🚀 开始 VGGT 推理，输入图片数量: {len(images)}，设备: {cfg.device}")

        # ---- Step 1: 图片预处理 ----
        images_tensor = self.preprocess_images(images).to(cfg.device)
        print(f"预处理后图片形状: {images_tensor.shape} (batch, channels, height, width)")

        # 判断推理模式
        if not getattr(cfg, "multi_reference", False):
            print("🧩 当前为【单次推理模式】")
            all_coords, all_rgb, all_conf = self._run_single_inference(images_tensor)
        else:
            print("🔁 当前为【多参考视角模式】")
            all_coords, all_rgb, all_conf = self._run_multi_reference_inference_fused(images_tensor)

        print(f"✅ VGGT 点云生成完成，总点数: {all_coords.shape[0]}")
        torch.cuda.empty_cache()
        return all_coords, all_rgb, all_conf


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

    def _conf_to_pointcloud_with_conf(self, predictions,use_confidence_color=False):

        """
        将模型输出 predictions 转为点云 + 置信度
        返回:
            coords (N,3), rgb (N,3), conf (N,)
        """
        # 这里假设 Pointmap 分支，conf 默认使用 world_points_conf
        cfg = self.cfg
        if not isinstance(predictions, dict):
            raise ValueError("predictions 必须是一个字典")

        # ---- Step 1: 选择点与置信度来源 ----
        if "Pointmap" in cfg.prediction_mode and "world_points" in predictions:
            world_points = predictions["world_points"]
            conf = predictions.get("world_points_conf", np.ones_like(world_points[..., 0]))
        else:
            world_points = predictions["world_points_from_depth"]
            conf = predictions.get("depth_conf", np.ones_like(world_points[..., 0]))

        images = predictions["images"]

        # ---- Step 2: 图片维度标准化 ----
        if images.ndim == 4 and images.shape[1] == 3:
            images = np.transpose(images, (0, 2, 3, 1))

        coords = world_points.reshape(-1, 3)
        coords[..., 1:] *= -1

        # ---- Step 3: 置信度映射为颜色 (替代 RGB) ----
        conf = conf.reshape(-1)
        conf_norm = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)

        # ✅ 使用 matplotlib colormap 映射为伪彩色（如“turbo”或“viridis”）
        # import matplotlib.cm as cm
        # colormap = cm.get_cmap("turbo")
        # rgb = (colormap(conf_norm)[:, :3] * 255).astype(np.uint8)

        # 如果仍希望保留原始图片颜色，可添加一个配置开关
        if not use_confidence_color:
            rgb = (images.reshape(-1, 3) * 255).astype(np.uint8)
        else:
            import matplotlib.cm as cm
            colormap = cm.get_cmap("turbo")
            rgb = (colormap(conf_norm)[:, :3] * 255).astype(np.uint8)


        # ---- Step 4: 掩码过滤 ----
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
        conf = conf_norm[mask]


        if coords.size == 0:
            coords = np.array([[0.0, 0.0, 0.0]])
            rgb = np.array([[255, 255, 255]], dtype=np.uint8)
        return coords, rgb.astype(np.uint8), conf

    def _conf_to_pointcloud(self, predictions):
        """提取预测点云的通用函数，可选择使用置信度代替 RGB 可视化"""

        cfg = self.cfg
        if not isinstance(predictions, dict):
            raise ValueError("predictions 必须是一个字典")

        # ---- Step 1: 选择点与置信度来源 ----
        if "Pointmap" in cfg.prediction_mode and "world_points" in predictions:
            world_points = predictions["world_points"]
            conf = predictions.get("world_points_conf", np.ones_like(world_points[..., 0]))
            print(f"world_points shape: {world_points.shape}")
            print(f"conf shape: {conf.shape}")

        else:
            world_points = predictions["world_points_from_depth"]
            conf = predictions.get("depth_conf", np.ones_like(world_points[..., 0]))

        images = predictions["images"]

        # ---- Step 2: 图片维度标准化 ----
        if images.ndim == 4 and images.shape[1] == 3:
            images = np.transpose(images, (0, 2, 3, 1))

        coords = world_points.reshape(-1, 3)
        coords[..., 1:] *= -1

        # ---- Step 3: 置信度映射为颜色 (替代 RGB) ----
        conf = conf.reshape(-1)
        conf_norm = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)

        # ✅ 使用 matplotlib colormap 映射为伪彩色（如“turbo”或“viridis”）
        import matplotlib.cm as cm
        colormap = cm.get_cmap("turbo")
        rgb = (colormap(conf_norm)[:, :3] * 255).astype(np.uint8)

        # 如果仍希望保留原始图片颜色，可添加一个配置开关
        # if not cfg.use_confidence_color:
        #     rgb = (images.reshape(-1, 3) * 255).astype(np.uint8)

        # ---- Step 4: 掩码过滤 ----
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

    def _run_single_inference(self, images_tensor: torch.Tensor):
        cfg = self.cfg
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                predictions = self.model(images_tensor,camera_extrinsics=self.extrinsics)


        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], images_tensor.shape[-2:]
        )
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        # 转 numpy
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)

        # 深度图 → 点云
        depth_map = predictions["depth"]
        world_points = unproject_depth_map_to_point_map(
            depth_map, predictions["extrinsic"], predictions["intrinsic"]
        )
        predictions["world_points_from_depth"] = world_points

        # 转换为点云
        coords, rgb, conf = self._conf_to_pointcloud_with_conf(predictions)
        # 将conf设置为全1
        conf = np.ones_like(conf)
        return coords, rgb, conf

    def _run_multi_reference_inference_fused(self, images_tensor: torch.Tensor):
        """
        多参考图推理 + 置信度加权融合点云（统一到世界坐标系，适配3×4外参）
        Args:
            images_tensor (torch.Tensor): 多视角图片 (N, 3, H, W)，N=6
        Returns:
            coords_fused (np.ndarray): 融合后的点云坐标 (M, 3)
            rgb_fused (np.ndarray): 融合后的点云颜色 (M, 3)
        """
        cfg = self.cfg
        num_views = images_tensor.shape[0]  # num_views=6
        all_coords_world = []  # 存储世界坐标系下的点云
        all_rgb = []
        all_conf = []

        # 选择基准坐标系：这里以第0张图的相机坐标系为统一基准
        base_view_idx = 0


        for ref_idx in range(num_views):
            print(f"\n📸 以第 {ref_idx} 张图片作为参考图进行推理...")

            # 重排图片：参考图放第一位，其余顺序不变
            reordered = torch.cat([
                images_tensor[ref_idx:ref_idx+1],
                torch.cat([images_tensor[:ref_idx], images_tensor[ref_idx+1:]], dim=0)
            ], dim=0).unsqueeze(0)  # 输出形状: (1, 6, 3, H, W)

            # 模型推理（禁用梯度和混合精度）
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    # predictions: dict, keys: ['pose_enc', 'depth', 'world_points', 'world_points_conf']
                    # world_points: (6, 518,518, 3)

                    predictions = self.model(reordered)

            # 解析相机参数（外参3×4，内参3×3）
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                predictions["pose_enc"], reordered.shape[-2:]
            )  # extrinsic: (1,6,3,4), intrinsic: (1,6,3,3)
            predictions["extrinsic"] = extrinsic
            predictions["intrinsic"] = intrinsic
            # Tensor转numpy，去除batch维度
            for key in predictions.keys():
                if isinstance(predictions[key], torch.Tensor):
                    predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # 外参变为(6,3,4)，内参变为(6,3,3)

            # --------------------------
            # 关键：统一world_points到基准坐标系
            # --------------------------
            # 1. 获取当前推理中各视角的外参（重排后的顺序）
            #    外参extrinsic格式：[R|t]，表示相机坐标系→世界坐标系的转换（世界坐标系以ref_idx为基准）
            #    即：P_world = R * P_cam + t
            current_extrinsics = predictions["extrinsic"]  # shape: (6, 3, 4)

            # 2. 重排外参回原始顺序（与原始images_tensor的视角对应）
            reorder_indices = [ref_idx] + [i for i in range(num_views) if i != ref_idx]
            inverse_indices = np.argsort(reorder_indices)  # 原始顺序索引
            extrinsics_original_order = current_extrinsics[inverse_indices]  # 恢复为原始视角顺序的外参

            # 3. 计算从“当前参考图坐标系”到“基准坐标系”的转换矩阵
            #    基准坐标系是base_view_idx的相机坐标系，当前世界坐标系是ref_idx的相机坐标系
            #    转换关系：P_base = T * P_current_world
            R_ref, t_ref = extrinsics_original_order[ref_idx, :3, :3], extrinsics_original_order[ref_idx, :3, 3:]  # ref_idx在原始顺序中的外参
            R_base, t_base = extrinsics_original_order[base_view_idx, :3, :3], extrinsics_original_order[base_view_idx, :3, 3:]  # 基准视角的外参

            # 外参的逆：世界坐标系（ref_idx）→ 相机坐标系（ref_idx）
            R_ref_inv = R_ref.T  # 旋转矩阵的逆=转置（正交矩阵）
            t_ref_inv = -R_ref_inv @ t_ref  # 平移的逆

            # 从ref_idx世界坐标系 → base_view_idx相机坐标系的转换矩阵（4x4齐次矩阵）
            T = np.eye(4)
            T[:3, :3] = R_base @ R_ref_inv  # 旋转部分：ref相机坐标系 → base相机坐标系
            T[:3, 3:] = R_base @ t_ref_inv + t_base  # 平移部分

            # 4. 转换当前推理的world_points到基准坐标系
            world_points = predictions["world_points"]  # shape: (6, 518, 518, 3)（6个视角的深度图生成的坐标）
            H, W = world_points.shape[1], world_points.shape[2]
            num_points_per_view = H * W  # 每个视角的点数：518*518

            # 展平为(N_views, N_points, 3)，方便批量转换
            world_points_flat = world_points.reshape(num_views, num_points_per_view, 3)  # (6, 518*518, 3)

            # 齐次坐标转换（添加w=1）
            world_points_hom = np.concatenate([
                world_points_flat,
                np.ones((num_views, num_points_per_view, 1), dtype=np.float32)
            ], axis=-1)  # (6, 518*518, 4)

            # 应用转换矩阵T：P_unified = T @ P_current_world
            world_points_unified_hom = (T @ world_points_hom.transpose(0, 2, 1)).transpose(0, 2, 1)  # (6, 518*518, 4)
            world_points_unified = world_points_unified_hom[..., :3]  # 去除齐次分量，(6, 518*518, 3)

            # 恢复原始形状(6, 518, 518, 3)并存储
            world_points_unified_reshaped = world_points_unified.reshape(num_views, H, W, 3)
            predictions["world_points"] = world_points_unified_reshaped


            # 提取点云、颜色、置信度（相机坐标系下）
            coords_cam, rgb, conf = self._conf_to_pointcloud_with_conf(predictions)  # (M,3), (M,3), (M,)


            x_mean = np.mean(coords_cam[:, 0])
            y_mean = np.mean(coords_cam[:, 1])
            z_mean = np.mean(coords_cam[:, 2])
            coords_cam -= np.array([x_mean, y_mean, z_mean])

            # 用numpy保存coords_cam, rgb, conf数据
            # np.savez_compressed(
            #     f"init_3dgs_{ref_idx}.npz",
            #     coords_cam=coords_cam,
            #     rgb=rgb,
            #     conf=conf
            # )


            # 存储世界坐标系下的点云数据
            all_coords_world.append(coords_cam)
            all_rgb.append(rgb)
            all_conf.append(conf)

            print(f"✅ 第 {ref_idx} 张参考图完成：{coords_cam.shape[0]} 个点（已转世界坐标系）")
            torch.cuda.empty_cache()  # 清理GPU缓存
            del predictions  # 释放内存
        print(len(all_coords_world))

        coords_fused, rgb_fused, conf_fused = fusion_point_clouds(
            all_coords_world,
            all_rgb,
            all_conf,
        )


        print(f"🎯 多参考图融合完成：总点数 {coords_fused.shape[0]}")
        return coords_fused, rgb_fused, conf_fused