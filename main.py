#from diffusers import DiffusionPipeline
#import torch
#
#pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")#, torch_dtype=torch.float16)
#pipeline.to("cpu")
#pipe = pipeline("An image of a squirrel in Picasso style")
#image = pipe.images[0]
#image.save("output.png","PNG")
#breakpoint()



#import torch
#import requests
#from PIL import Image
#from diffusers import StableDiffusionDepth2ImgPipeline
#
## Create pipeline
#pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
#        "stabilityai/stable-diffusion-2-depth",
#        torch_dtype=torch.float16)
#pipe.to("cuda")
##pipe.enable_attention_slicing() # NOTE: This makes the model never even start training
#
## Get input image
#url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#init_image = Image.open(requests.get(url, stream=True).raw)
##init_image = Image.open('cats.jpg')
## Prompt and negative prompt
#prompt = "two tigers"
#n_prompt = "bad, deformed, ugly, bad anotomy"
## Generate
#image = pipe(prompt=prompt, image=init_image, negative_prompt=n_prompt,
#        strength=0.7, height=32, width=32).images[0]
#image.save("output.png","PNG")



import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image

# Create pipeline
gpu = False
if gpu:
    pipeline = AutoPipelineForImage2Image.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to("cuda")
else:
    pipeline = AutoPipelineForImage2Image.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32, variant="fp32", use_safetensors=True
        ).to("cpu")
breakpoint()

# Prepare image
init_image = Image.open("coarse.png")

# Choose prompt
prompt = "realistic, detailed, 8k"
n_prompt = ""

# Write prompt to file in the output
lines = [f'prompt = "{prompt}"\n', f'n_prompt = "{n_prompt}"']
with open('output/prompt.txt', 'w') as f:
    f.writelines(lines)

strengths = [.2, .5, .8]
# Loop over strengths
for strength in strengths:
    # pass prompt and image to pipeline
    images = pipeline(prompt, negative_prompt=n_prompt, image=init_image,
            num_images_per_prompt=5, strength=strength).images
    for i, image in enumerate(images):
        if i < 10:
            num = f'00{i}'
        elif i < 100:
            num = f'0{i}'
        else:
            num = f'{i}'
        image.save(f"output/output_strength{strength}_{num}.png", "PNG")
