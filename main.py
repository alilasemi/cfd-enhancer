import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image

# Create pipeline
gpu = True
if gpu:
    pipeline = AutoPipelineForImage2Image.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to("cuda")
else:
    pipeline = AutoPipelineForImage2Image.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32, variant="fp32", use_safetensors=True
        ).to("cpu")

# load attention processors
pipeline.unet.load_attn_procs('./')
breakpoint()

# Prepare image
init_image = Image.open("coarse.png")

# Choose prompt
#prompt = "realistic, detailed, 8k"
prompt = "realistic fluid dynamics, turbulent flow"
n_prompt = ""

# Write prompt to file in the output
lines = [f'prompt = "{prompt}"\n', f'n_prompt = "{n_prompt}"']
with open('output/prompt.txt', 'w') as f:
    f.writelines(lines)

#strengths = [.2, .25, .3, .35, .4, .45, .5]
strengths = [.35, .4, .45, .5, .55, .6, .65, .7]
# Loop over strengths
for strength in strengths:
    # pass prompt and image to pipeline
    images = pipeline(prompt, negative_prompt=n_prompt, image=init_image,
            num_images_per_prompt=5, strength=strength).images
            #cross_attention_kwargs={"scale": 1.0}).images
    for i, image in enumerate(images):
        if i < 10:
            num = f'00{i}'
        elif i < 100:
            num = f'0{i}'
        else:
            num = f'{i}'
        image.save(f"output/output_strength{strength}_{num}.png", "PNG")
