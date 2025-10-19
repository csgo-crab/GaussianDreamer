from dataclasses import dataclass, field
import os
import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from mvadapter.pipelines.pipeline_mvadapter_t2mv_sdxl import MVAdapterT2MVSDXLPipeline
from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from mvadapter.utils import make_image_grid
from mvadapter.utils.geometry import get_plucker_embeds_from_cameras_ortho
from mvadapter.utils.mesh_utils import get_orthogonal_camera


@dataclass
class PipelineConfig:
    # 模型参数
    base_model: str = "load/stabilityai/stable-diffusion-xl-base-1.0"
    vae_model: str = "load/madebyollin/sdxl-vae-fp16-fix"
    unet_model: str | None = None
    scheduler: str | None = None
    lora_model: str | None = None
    adapter_path: str = "load/huanngzh/mv-adapter"

    # 推理参数
    text: str = "a beautiful 3d object"
    negative_prompt: str = "watermark, ugly, deformed, noisy, blurry, low contrast"
    num_views: int = 6
    azimuth_deg: list[int] = field(default_factory=lambda: [0, 45, 90, 180, 270, 315])
    num_inference_steps: int = 50
    guidance_scale: float = 7.0
    seed: int = -1
    lora_scale: list[str] | str = "1.0"

    # 输出参数
    height: int = 768
    width: int = 768
    output: str = "output.png"
    save_alone: bool = False

    # 设备与精度
    device: str = "cuda"
    dtype: torch.dtype = torch.float16


class MVAdapterPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.pipe, self.adapter_name_list = self._prepare_pipeline()

    def _prepare_pipeline(self):
        """加载模型与adapter"""
        cfg = self.cfg
        pipe_kwargs = {}

        # 加载VAE
        if cfg.vae_model:
            if os.path.exists(cfg.vae_model):
                pipe_kwargs["vae"] = AutoencoderKL.from_pretrained(cfg.vae_model, local_files_only=True)
            else:
                pipe_kwargs["vae"] = AutoencoderKL.from_pretrained(cfg.vae_model)

        # 加载UNet
        if cfg.unet_model:
            if os.path.exists(cfg.unet_model):
                pipe_kwargs["unet"] = UNet2DConditionModel.from_pretrained(cfg.unet_model, local_files_only=True)
            else:
                pipe_kwargs["unet"] = UNet2DConditionModel.from_pretrained(cfg.unet_model)

        # 加载主模型
        if os.path.exists(cfg.base_model):
            pipe = MVAdapterT2MVSDXLPipeline.from_pretrained(cfg.base_model, local_files_only=True, **pipe_kwargs)
        else:
            pipe = MVAdapterT2MVSDXLPipeline.from_pretrained(cfg.base_model, **pipe_kwargs)

        # 调度器
        scheduler_class = None
        if cfg.scheduler == "ddpm":
            scheduler_class = DDPMScheduler

        pipe.scheduler = ShiftSNRScheduler.from_scheduler(
            pipe.scheduler,
            shift_mode="interpolated",
            shift_scale=8.0,
            scheduler_class=scheduler_class,
        )

        pipe.init_custom_adapter(num_views=cfg.num_views)
        pipe.load_custom_adapter(cfg.adapter_path, weight_name="mvadapter_t2mv_sdxl.safetensors")

        pipe.to(device=cfg.device, dtype=cfg.dtype)
        pipe.cond_encoder.to(device=cfg.device, dtype=cfg.dtype)

        # 加载LoRA
        adapter_name_list = []
        if cfg.lora_model:
            for lora_model_ in cfg.lora_model.split(","):
                model_, name_ = lora_model_.strip().rsplit("/", 1)
                adapter_name = name_.split(".")[0]
                adapter_name_list.append(adapter_name)
                pipe.load_lora_weights(model_, weight_name=name_, adapter_name=adapter_name)

        pipe.enable_vae_slicing()
        return pipe, adapter_name_list

    def run(self):
        """执行推理"""
        cfg = self.cfg
        pipe = self.pipe

        # 设置 LoRA 权重
        if len(self.adapter_name_list) > 0:
            lora_scale = cfg.lora_scale
            if isinstance(lora_scale, str):
                lora_scale = [lora_scale]
            if len(lora_scale) == 1:
                lora_scale = [lora_scale[0]] * len(self.adapter_name_list)
            else:
                assert len(lora_scale) == len(self.adapter_name_list), \
                    "Number of lora scales must match number of adapters"
            lora_scale = [float(s) for s in lora_scale]
            pipe.set_adapters(self.adapter_name_list, adapter_weights=lora_scale)
            print(f"Loaded {len(self.adapter_name_list)} adapters with scales {lora_scale}")

        # 准备相机
        cameras = get_orthogonal_camera(
            elevation_deg=[0] * cfg.num_views,
            distance=[1.8] * cfg.num_views,
            left=-0.55, right=0.55, bottom=-0.55, top=0.55,
            azimuth_deg=[x - 90 for x in cfg.azimuth_deg],
            device=cfg.device,
        )

        plucker_embeds = get_plucker_embeds_from_cameras_ortho(
            cameras.c2w, [1.1] * cfg.num_views, cfg.width
        )
        control_images = ((plucker_embeds + 1.0) / 2.0).clamp(0, 1)

        pipe_kwargs = {"max_sequence_length": 214}
        if cfg.seed != -1:
            pipe_kwargs["generator"] = torch.Generator(device=cfg.device).manual_seed(cfg.seed)

        # 执行推理
        result = pipe(
            cfg.text,
            height=cfg.height,
            width=cfg.width,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            num_images_per_prompt=cfg.num_views,
            control_image=control_images,
            control_conditioning_scale=1.0,
            negative_prompt=cfg.negative_prompt,
            **pipe_kwargs,
        )
        images = result.images

        # 保存结果
        if cfg.save_alone:
            for i, image in enumerate(images):
                image.save(f"{cfg.output.split('.')[0]}_{i}.png")
        else:
            make_image_grid(images, rows=1).save(cfg.output)
        torch.cuda.empty_cache()
        print(f"✅ 输出已保存到: {cfg.output}")
        return images


if __name__ == "__main__":
    # 示例调用
    cfg = PipelineConfig(
        text="a cute crab wearing sunglasses, high detail",
        output="test_output.png",
        seed=42
    )
    pipeline = MVAdapterPipeline(cfg)
    pipeline.run()
