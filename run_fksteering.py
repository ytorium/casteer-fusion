import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
import sys
sys.path.append('fkd_diffusers')

import torch
from diffusers import DDIMScheduler

# local imports
from launch_eval_runs import do_eval
from fks_utils import get_model

# set args
args = dict(
    seed=0, output_dir="output", eta=1.0, metrics_to_compute="ImageReward",
    prompt_path='./prompt_files/image_rewards_benchmark.json', 
    model_name="stable-diffusion-xl", 
  )

fkd_args = dict(
    lmbda=2.0, num_particles=4, adaptive_resampling=True, resample_frequency=20,
    time_steps=100, potential_type='max', resampling_t_start=20,
    resampling_t_end=50, guidance_reward_fn='ImageReward', use_smc=True,
   )

args = argparse.Namespace(**args, **fkd_args)
print(args)

args.num_inference_steps = fkd_args["time_steps"]
print(fkd_args)

# seed everything
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

pipeline = get_model(args.model_name)
pipeline = pipeline.to("cuda")    
print('pipeline loaded')


# set output directory
cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
output_dir = os.path.join(args.output_dir, cur_time)
os.makedirs(output_dir, exist_ok=False)
arg_path = os.path.join(output_dir, "args.json")
with open(arg_path, "w") as f:
    json.dump(vars(args), f, indent=4)

score_path = os.path.join(output_dir, "scores.jsonl")
images_path = os.path.join(output_dir, "images")
os.makedirs(images_path, exist_ok=False)

metrics_to_compute = args.metrics_to_compute.split("#")

# cache metric fns
do_eval(
    prompt=["test"],
    images=[Image.new("RGB", (224, 224))],
    metrics_to_compute=metrics_to_compute,
    )



# add prompts for generation
prompt_data = [
    {"prompt": "a photo of a brown knife and a blue donut"},
    {"prompt": "a photo of a blue clock and a white cup"},
    {"prompt": "a photo of an orange cow and a purple sandwich"},
    {"prompt": "a photo of a yellow bird and a black motorcycle"},
    {"prompt": "a photo of a green tennis racket and a black dog"},
    {"prompt": "a green stop sign in a red field"},    
]

print('promts: ', len(prompt_data))


show_best_particle = True


with open(score_path, "w") as score_f:
    for prompt_idx, item in enumerate(prompt_data):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)


        prompt = [item['prompt']]*fkd_args['num_particles']
        start_time = datetime.now()

        images = pipeline(prompt, 
                          num_inference_steps=fkd_args["time_steps"], 
                          eta=args.eta,
                          fkd_args=fkd_args)
        end_time = datetime.now()        
        images = images[0]

        time_taken = end_time - start_time

        results = do_eval(prompt=prompt, images=images, metrics_to_compute=metrics_to_compute)
        guidance_reward = np.array(results["ImageReward"]["result"])
        sorted_idx = np.argsort(guidance_reward)[::-1]
        images = [images[i] for i in sorted_idx]

        results['time_taken'] = time_taken.total_seconds()
        results['prompt'] = prompt
        results['prompt_index'] = prompt_idx

        image_fpath = os.path.join(images_path, f"{prompt_idx}.png")
        results['image_path'] = image_fpath

        score_f.write(json.dumps(results) + "\n")

        if show_best_particle:
            _, ax = plt.subplots(1, 1, figsize=(5, 5))            
            ax.imshow(images[0])
            ax.axis("off")
        else:
            _, ax = plt.subplots(1, args.num_particles, figsize=(args.num_particles*5, 5))
            for i, image in enumerate(images):
                ax[i].imshow(image)
                ax[i].axis("off")

        plt.suptitle(prompt[0])
        plt.savefig(image_fpath)
        plt.show()
        plt.close()

print('The End')



