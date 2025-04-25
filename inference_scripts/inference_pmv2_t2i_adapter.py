# !pip install opencv-python transformers accelerate
import os
import sys

import numpy as np
import torch
from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler, T2IAdapter

from huggingface_hub import hf_hub_download
from controlnet_aux import OpenposeDetector
from photomaker import PhotoMakerStableDiffusionXLAdapterPipeline
from photomaker import FaceAnalysis2, analyze_faces
from style_template import get_style_info

face_detector = FaceAnalysis2(providers=['CUDAExecutionProvider'], allowed_modules=['detection', 'recognition'])
face_detector.prepare(ctx_id=0, det_size=(640, 640))

try:
    if torch.cuda.is_available():
        device = "cuda"
    elif sys.platform == "darwin" and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
except:
    device = "cpu"

torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
if device == "mps":
    torch_dtype = torch.float16

output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)

photomaker_ckpt = hf_hub_download(repo_id="TencentARC/PhotoMaker-V2", filename="photomaker-v2.bin", repo_type="model")

openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
# load adapter
adapter = T2IAdapter.from_pretrained(
  "TencentARC/t2i-adapter-openpose-sdxl-1.0", torch_dtype=torch_dtype,
).to("cuda")

# prompt = "instagram photo, a photo of a woman img, colorful, perfect face, natural skin, hard shadows, film grain, best quality"
# negative_prompt = "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch)"

style_name = "Photographic"
user_prompt = "lean muscular young man img, upper body naked, front view, entire body, posing for instagram, colorful, perfect face, natural skin, hard shadows, film grain, best quality, preserve face and hair details, realistic skin texture"
prompt, negative_prompt = get_style_info(style_name, user_prompt)
print(prompt, negative_prompt)

# download an image
pose_image = load_image(
    "./examples/model.png"
)
pose_image = openpose(pose_image, detect_resolution=512, image_resolution=1024)

# initialize the models and pipeline
adapter_conditioning_scale = 0  # recommended for good generalization
adapter_conditioning_factor = 0.8

### Load base model
pipe = PhotoMakerStableDiffusionXLAdapterPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0", 
    adapter=adapter,
    torch_dtype=torch_dtype,
).to("cuda")

### Load PhotoMaker checkpoint
pipe.load_photomaker_adapter(
    os.path.dirname(photomaker_ckpt),
    subfolder="",
    weight_name=os.path.basename(photomaker_ckpt),
    trigger_word="img"  # define the trigger word
)     
### Also can cooperate with other LoRA modules
# pipe.load_lora_weights(os.path.dirname(lora_path), weight_name=lora_model_name, adapter_name="lcm-lora")
# pipe.set_adapters(["photomaker", "lcm-lora"], adapter_weights=[1.0, 0.5])

pipe.fuse_lora()
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

pipe.enable_model_cpu_offload()

### define the input ID images
input_folder_name = './examples/nikk'
image_basename_list = os.listdir(input_folder_name)
image_path_list = sorted([os.path.join(input_folder_name, basename) for basename in image_basename_list])

input_id_images = []
for image_path in image_path_list:
    input_id_images.append(load_image(image_path))

id_embed_list = []

for img in input_id_images:
    img = np.array(img)
    img = img[:, :, ::-1]
    faces = analyze_faces(face_detector, img)
    if len(faces) > 0:
        id_embed_list.append(torch.from_numpy((faces[0]['embedding'])))

if len(id_embed_list) == 0:
    raise ValueError(f"No face detected in input image pool")

id_embeds = torch.stack(id_embed_list)

style_strength_ratio = 20
num_steps = 50
start_merge_step = int(float(style_strength_ratio) / 100 * num_steps)
if start_merge_step > 30:
    start_merge_step = 30
# generate image
images = pipe(
    prompt, 
    negative_prompt=negative_prompt, 
    width=1024,
    height=1024,
    input_id_images=input_id_images,
    id_embeds=id_embeds,
    adapter_conditioning_scale=adapter_conditioning_scale,
    num_inference_steps=num_steps,
    guidance_scale=5,
    image=pose_image,
    num_images_per_prompt=2,
    start_merge_step=start_merge_step,
    seed=42,
).images

for idx, img in enumerate(images): 
    img.save(os.path.join(output_dir, f"output_pmv2_t2ia_{idx}.jpg"))
