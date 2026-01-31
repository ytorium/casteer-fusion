import os
import pickle
import torch

# local imports
import image_utils
import steering
import prompt_catalog
from models import get_model

# parsing arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['sdxl', 'sdxl-turbo', 'sdxl-turbo-image'], default="sdxl-turbo")
parser.add_argument('--mode', type=str, choices=['concrete', 'human-related', 'anime-style'], default="anime-style")
parser.add_argument('--num_steps', type=int, default=20) #  1 for turbo, 20 for sdxl
parser.add_argument('--steer_vectors', type=str, default='casteer_vectors') # path to saving steering vectors
args = parser.parse_args()

pipe = get_model(args.model)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pipe.to(device)
pipe.set_progress_bar_config(disable=True)


## Hyper Paramters
STEER_TYPE = "default"
INF_STEPS = 20
GUIDE_SCALE = 5.0

# Loads "calibration dataset": dataset from which steering vectors are derived from
if args.mode == "anime-style":
    steer_prompts = prompt_catalog.ANIME_PROMPT[:20]
elif args.mode == "metal":
    steer_prompts = prompt_catalog.METALLIC_SCULPTURE_SET[:20]
elif args.mode == "fine_image":
    steer_prompts = prompt_catalog.FINE_IMAGE_PROMPT[:20]

else:
  raise NotImplementedError(f"Steering prompt mode {args.mode} not implemented")

# add hooks to collect activations and later applies steering vector to intermiate activations
steer_hooks = steering.add_steer_hooks(pipe, steer_type=STEER_TYPE, save_every=1)

steering_vectors = steering.build_final_steering_vectors(
    pipe,
    steer_hooks,
    steer_prompts,
    num_inference_steps=INF_STEPS,
    guidance_scale=GUIDE_SCALE
)

# Saving steering vectors:
if not os.path.exists(args.steer_vectors):
    os.makedirs(args.steer_vectors)

with open(os.path.join(args.steer_vectors, '{}_{}.pickle'.format(args.model, args.mode)), 'wb') as handle:
    pickle.dump(steering_vectors, handle)

print("steering_vectors shape: ", steering_vectors[0].shape) # (20, 2, 640) 40,1,640

