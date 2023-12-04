import random
import math

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from PIL import Image

from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers import AutoPipelineForImage2Image
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available



def main():
    seed = 512
    rank = 4
    enable_xformers_memory_efficient_attention = False

    # Create pipeline
    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    weight_dtype = torch.float16
    pipe = AutoPipelineForImage2Image.from_pretrained(
        pretrained_model_name_or_path, torch_dtype=weight_dtype, variant="fp16", use_safetensors=True
        ).to("cuda")

    # Prepare image
    init_image = Image.open("coarse.png")

    # -- Training -- #
    # Load scheduler, tokenizer and models
    noise_scheduler = pipe.scheduler
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet
    # Freeze existing neural network parameters
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # now we will add new LoRA weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
    # => 32 layers

    # Set correct lora layers
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank,
        )

    unet.set_attn_processor(lora_attn_procs)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    lora_layers = AttnProcsLayers(unet.attn_processors)

    # Initialize the optimizer
    optimizer_cls = torch.optim.AdamW

    lr = 1e-4
    adam_beta1 = .9
    adam_beta2 = .999
    adam_weight_decay = 1e-2
    adam_epsilon = 1e-8
    optimizer = optimizer_cls(
        lora_layers.parameters(),
        lr=lr,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    data_files = {}
    #train_data_dir = './'
    #data_files["train"] = os.path.join(train_data_dir, "**")
    data_files["train"] = ['coarse.png', 'metadata.jsonl']
    dataset = load_dataset("imagefolder", data_files=data_files, cache_dir=None)
    # See more about loading custom images at
    # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # -- Preprocessing the datasets -- #
    # Tokenize inputs and targets
    column_names = dataset["train"].column_names
    # Get the column names for input/target
    image_column, caption_column = column_names

    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    resolution = 512
    random_flip = False
    train_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    # Set the training transforms
    train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    # DataLoaders creation:
    train_batch_size = 1
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=train_batch_size,
        num_workers=0,
    )

    # Scheduler and math around the number of training steps.
    num_train_epochs = 100
    gradient_accumulation_steps = 1
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True

    lr_scheduler = 'constant'
    lr_warmup_steps = 500
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if overrode_max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {num_train_epochs}")
    print(f"  Instantaneous batch size per device = {train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    print(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    initial_global_step = 0

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=False
    )

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Convert images to latent space
            latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype).to('cuda')).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch["input_ids"].to('cuda'))[0]

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Predict the noise residual and compute loss
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = loss.repeat(train_batch_size).mean()
            train_loss += avg_loss.item() / gradient_accumulation_steps

            # Backpropagate
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1
            print("train_loss: ", train_loss, "step = ", global_step)
            train_loss = 0.0

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

        validation_prompt = 'Flow over cylinder'
        validation_epochs = 5
        num_validation_images = 4
        # TODO: The validation is commented out for now, it's not really
        # necessary and makes the whole thing a lot slower to run so I figured
        # we can ignore it at the moment. I left the code here commented just in
        # case we decide that we need it.
#        if validation_prompt is not None and epoch % validation_epochs == 0:
#            print(
#                f"Running validation... \n Generating {num_validation_images} images with prompt:"
#                f" {validation_prompt}."
#            )
#            # create pipeline
#            pipeline = DiffusionPipeline.from_pretrained(
#                pretrained_model_name_or_path,
#                unet=unet,
#                revision=None,
#                torch_dtype=weight_dtype,
#            )
#            pipeline = pipeline.to('cuda')
#            pipeline.set_progress_bar_config(disable=True)
#
#            # run inference
#            generator = torch.Generator(device='cuda')
#            if seed is not None:
#                generator = generator.manual_seed(seed)
#            images = []
#            for _ in range(num_validation_images):
#                images.append(
#                    pipeline(validation_prompt, num_inference_steps=30, generator=generator).images[0]
#                )
#            breakpoint()
#
#            for tracker in accelerator.trackers:
#                if tracker.name == "tensorboard":
#                    np_images = np.stack([np.asarray(img) for img in images])
#                    tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
#                if tracker.name == "wandb":
#                    tracker.log(
#                        {
#                            "validation": [
#                                wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
#                                for i, image in enumerate(images)
#                            ]
#                        }
#                    )
#
#            del pipeline
#            torch.cuda.empty_cache()

    # Save the lora layers
    unet = unet.to(torch.float32)
    output_dir = './'
    unet.save_attn_procs(output_dir)

#    # Final inference
#    # Load previous pipeline
#    pipeline = DiffusionPipeline.from_pretrained(
#        pretrained_model_name_or_path, revision=None, torch_dtype=weight_dtype
#    )
#    pipeline = pipeline.to('cuda')
#
#    # load attention processors
#    pipeline.unet.load_attn_procs(output_dir)
#
#    # run inference
#    generator = torch.Generator(device='cuda')
#    generator = generator.manual_seed(seed)
#    images = []
#    for _ in range(num_validation_images):
#        images.append(pipeline(validation_prompt, num_inference_steps=30, generator=generator).images[0])
#
#    for i, image in enumerate(images):
#        image.save(f'output/valid_{i}.png')

if __name__ == "__main__":
    main()
