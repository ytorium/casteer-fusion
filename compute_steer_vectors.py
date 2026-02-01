import os
import sys
import argparse
import pickle
import torch

# local imports
import image_utils
import steering
import prompt_catalog
from fks_utils import get_model
from launch_eval_runs import do_eval

sys.path.append('fkd_diffusers')

# parsing arguments
#parser = argparse.ArgumentParser()
#parser.add_argument('--model', type=str, choices=['sdxl', 'sdxl-turbo', 'sdxl-turbo-image'], default="sdxl-turbo")
#arser.add_argument('--mode', type=str, choices=['concrete', 'human-related', 'anime-style'], default="anime-style")
#parser.add_argument('--num_steps', type=int, default=20) #  1 for turbo, 20 for sdxl
#parser.add_argument('--steer_vectors', type=str, default='casteer_vectors') # path to saving steering vectors
#args = parser.parse_args()

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pipeline.to(device)
pipeline.set_progress_bar_config(disable=True)
print('pipeline loaded')

## Hyper Paramters
STEER_TYPE = "default"
INF_STEPS = 20
GUIDE_SCALE = 5.0

# Loads "calibration dataset": dataset from which steering vectors are derived from

steer_prompts = prompt_catalog.ANIME_PROMPT[:20]
'''
if args.mode == "anime-style":
    steer_prompts = prompt_catalog.ANIME_PROMPT[:20]
elif args.mode == "metal":
    steer_prompts = prompt_catalog.METALLIC_SCULPTURE_SET[:20]
elif args.mode == "fine_image":
    steer_prompts = prompt_catalog.FINE_IMAGE_PROMPT[:20]

else:
  raise NotImplementedError(f"Steering prompt mode {args.mode} not implemented")
'''
# add hooks to collect activations and later applies steering vector to intermiate activations
steer_hooks = steering.add_steer_hooks(pipeline, steer_type=STEER_TYPE, save_every=1)

steering_vectors = steering.build_final_steering_vectors(
    pipeline,
    steer_hooks,
    steer_prompts,
    num_inference_steps=INF_STEPS,
    guidance_scale=GUIDE_SCALE
)

arg_model = "sdxl-fkd"
arg_mode = "anime-style"
arg_steer_vectors = "/content/drive/My Drive/castfus_vectors"

# Saving steering vectors:
if not os.path.exists(arg_steer_vectors):
    os.makedirs(arg_steer_vectors)

with open(os.path.join(arg_steer_vectors, '{}_{}.pickle'.format(arg_model, arg_mode)), 'wb') as handle:
    pickle.dump(steering_vectors, handle)

print("steering_vectors shape: ", steering_vectors[0].shape) # (20, 2, 640) 40,1,640

