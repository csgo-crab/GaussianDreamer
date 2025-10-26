import os
import torch
import numpy as np
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field

from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene import Scene, GaussianModel
from gaussiansplatting.arguments import ModelParams, PipelineParams, get_combined_args,OptimizationParams
from gaussiansplatting.scene.cameras import Camera
from gaussiansplatting.utils.sh_utils import SH2RGB
from gaussiansplatting.scene.gaussian_model import BasicPointCloud

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy
from threestudio.utils.typing import *


@threestudio.register("gaussiandreamer-system")
class GaussianDreamer(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        radius: float = 4
        sh_degree: int = 0
        load_type: int = 0
        load_path: str = "./load/shapes/stand.obj"


    # ================== 初始化相关 ==========================
    cfg: Config
    def configure(self) -> None:
        self.radius = self.cfg.radius
        self.sh_degree =self.cfg.sh_degree
        self.load_type =self.cfg.load_type
        self.load_path = self.cfg.load_path

        self.gaussian = GaussianModel(sh_degree = self.sh_degree)
        bg_color = [1, 1, 1] if False else [0, 0, 0]
        self.background_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    def pcd_init(self) -> BasicPointCloud:
        """加载点云数据, 并处理成3dgs的点云格式"""
        # Since this data set has no colmap data, we start with random points
        if self.load_type== 4: # shap_e
            from threestudio.systems.function.point_cloud import load_from_shape
            coords,rgb = load_from_shape(self.load_path)
        elif self.load_type == 1: # pcd
            from threestudio.systems.function.point_cloud import load_from_pcd
            coords,rgb = load_from_pcd(self.load_path)
        elif self.load_type == 2: # smpl
            from threestudio.systems.function.point_cloud import load_from_smpl
            coords,rgb = load_from_smpl(self.load_path)
        elif self.load_type == 3: # 3dgs
            from threestudio.systems.function.point_cloud import load_from_3dgs
            coords,rgb = load_from_3dgs(self.load_path)
        elif self.load_type == 0: # vggt
            from threestudio.systems.function.point_cloud import load_from_vggt
            save_path = self.get_save_path('instance_images/')
            coords,rgb = load_from_vggt(self.cfg, save_path)
        else:
            raise NotImplementedError(f"load_type {self.load_type} is not implemented, only support [0: shap_e, 1: pcd, 2: smpl, 3: 3dgs]")
        
        bound = self.radius * 0.75
        pcd = BasicPointCloud(points=coords*bound, colors=rgb, normals=np.zeros((coords.shape[0], 3)))
        return pcd
    
    def on_fit_start(self) -> None:
        super().on_fit_start()

        # 如果本地存在instance image, 则可以finetune
        if os.path.exists(os.path.join(self.get_save_dir(), 'instance_images/')):
            # finetune guidance(需要本地下载好模型)
            cmd = [
                "python", "threestudio/systems/function/dreambooth.py",
                "--pretrained_model_name_or_path", self.cfg.guidance.pretrained_model_name_or_path,
                "--enable_xformers_memory_efficient_attention",
                "--with_prior_preservation",
                "--instance_data_dir", self.get_save_path('instance_images/'),
                "--instance_prompt", self.cfg.prompt_processor.prompt,
                "--class_data_dir", self.get_save_path('class_samples/'),
                "--class_prompt", self.cfg.prompt_processor.prompt.replace('<kth>', ''),
                "--validation_prompt", self.cfg.prompt_processor.prompt,
                "--output_dir", self.get_save_path('personalization/'),
                "--max_train_steps", str(4000),
                "--train_batch_size", str(1),
                "--gradient_accumulation_steps", str(4),
                "--mixed_precision", "fp16"
            ]
            # 执行dreambooth训练
            import subprocess
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print("DreamBooth训练成功完成")
                print(result.stdout)
                if result.returncode != 0:
                    raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
            except subprocess.CalledProcessError as e:
                print("DreamBooth训练失败")
                print(f"错误输出: {e.stderr}")
                print(f"返回码: {e.returncode}")
        
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.get_save_path('personalization/')
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.get_save_path('personalization/'))

    def configure_optimizers(self):
        self.parser = ArgumentParser(description="Training script parameters")
        
        opt = OptimizationParams(self.parser)
        point_cloud = self.pcd_init()
        self.cameras_extent = 4.0
        self.gaussian.create_from_pcd(point_cloud, self.cameras_extent)

        # 检查转换出的初始3dgs
        save_path = self.get_save_path(f"init_3dgs.ply")
        self.gaussian.save_ply(save_path)
        # 保存转换到rgb空间的点云
        from threestudio.systems.function.point_cloud import save_ply, load_from_3dgs
        coords, rgb = load_from_3dgs(save_path)
        save_ply(self.get_save_path(f"init-color.ply"), coords, rgb)

        # 准备训练
        self.pipe = PipelineParams(self.parser)
        self.gaussian.training_setup(opt)
        
        ret = {
            "optimizer": self.gaussian.optimizer,
        }
        return ret
    
    
    # ================== 迭代相关 ==========================
    def forward(self, batch: Dict[str, Any],renderbackground = None) -> Dict[str, Any]:
        if renderbackground is None:
            renderbackground = self.background_tensor
        images = []
        depths = []
        self.viewspace_point_list = []
        for id in range(batch['c2w_3dgs'].shape[0]):
            viewpoint_cam  = Camera(c2w = batch['c2w_3dgs'][id],FoVy = batch['fovy'][id],height = batch['height'],width = batch['width'])
            render_pkg = render(viewpoint_cam, self.gaussian, self.pipe, renderbackground)
            image, viewspace_point_tensor, _, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            self.viewspace_point_list.append(viewspace_point_tensor)

            if id == 0:
                self.radii = radii
            else:
                self.radii = torch.max(radii,self.radii)
                
            depth = render_pkg["depth_3dgs"]
            depth =  depth.permute(1, 2, 0)
            
            image =  image.permute(1, 2, 0)
            images.append(image)
            depths.append(depth)
            
        images = torch.stack(images, 0)
        depths = torch.stack(depths, 0)
        self.visibility_filter = self.radii>0.0
        render_pkg["comp_rgb"] = images
        render_pkg["depth"] = depths
        render_pkg["opacity"] = depths / (depths.max() + 1e-5)
        return {
            **render_pkg,
        }
    
    def training_step(self, batch, batch_idx):
        self.gaussian.update_learning_rate(self.true_global_step)
        if self.true_global_step > 500:
            self.guidance.set_min_max_steps(min_step_percent=0.02, max_step_percent=0.55)

        out = self(batch) 

        prompt_utils = self.prompt_processor()
        images = out["comp_rgb"]

        guidance_eval = (self.true_global_step % 200 == 0)        
        guidance_out = self.guidance(
            images, prompt_utils, **batch, rgb_as_latents=False,guidance_eval=guidance_eval
        )
        
        loss = 0.0
        loss = loss + guidance_out['loss_sds'] *self.C(self.cfg.loss['lambda_sds'])
        
        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)
        
        if guidance_eval:
            self.guidance_evaluation_save(
                out["comp_rgb"].detach()[: guidance_out["eval"]["bs"]],
                guidance_out["eval"],
            )
        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))
        return {"loss": loss}

    def on_before_optimizer_step(self, optimizer):
        with torch.no_grad():
            if self.true_global_step < 900: # 15000
                viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                for idx in range(len(self.viewspace_point_list)):
                    viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                # Keep track of max radii in image-space for pruning
                self.gaussian.max_radii2D[self.visibility_filter] = torch.max(self.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])
                
                self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.visibility_filter)

                if self.true_global_step > 300 and self.true_global_step % 100 == 0: # 500 100
                    size_threshold = 20 if self.true_global_step > 500 else None # 3000
                    self.gaussian.densify_and_prune(0.0002 , 0.05, self.cameras_extent, size_threshold) 


    # ================== 验证测试相关 ==========================
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            ),
            name="validation_step",
            step=self.true_global_step,
        )
        # save_path = self.get_save_path(f"it{self.true_global_step}-val.ply")
        # self.gaussian.save_ply(save_path)
        # load_ply(save_path,self.get_save_path(f"it{self.true_global_step}-val-color.ply"))

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        only_rgb = True
        bg_color = [1, 1, 1] if False else [0, 0, 0]

        testbackground_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        out = self(batch,testbackground_tensor)
        if only_rgb:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                ),
                name="test_step",
                step=self.true_global_step,
            )
        else:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "grayscale",
                            "img": out["depth"][0],
                            "kwargs": {},
                        }
                    ]
                    if "depth" in out
                    else []
                )
                + [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ],
                name="test_step",
                step=self.true_global_step,
            )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
        save_path = self.get_save_path(f"last_3dgs.ply")
        self.gaussian.save_ply(save_path)
        # 保存转换到rgb空间的点云
        from threestudio.systems.function.point_cloud import save_ply, load_from_3dgs
        coords, rgb = load_from_3dgs(save_path)
        save_ply(self.get_save_path(f"it{self.true_global_step}-test-color.ply"), coords, rgb)
        