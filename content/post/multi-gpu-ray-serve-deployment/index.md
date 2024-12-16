---
title: Multi GPU Deployment for a Large AI Model
subtitle: Using ray serve to deploy a large AI model on multiple GPUs as an API endpoint.  

# Summary for listings and search engines
summary: Using ray serve to deploy a large AI model on multiple GPUs as an API endpoint.

# Link this post with a project
projects: []

# Date published
date: "2024-12-15T00:00:00Z"

# Date updated
lastmod: "2024-12-15T00:00:00Z"

# Is this an unpublished draft?
draft: false

# Show this page in the Featured widget?
featured: false

# Featured image
# Place an image named `featured.jpg/png` in this page's folder and customize its options here.
image:
  caption: 'Image credit: [**Anyscale**](https://www.anyscale.com/)'
  focal_point: Smart
  placement: 2
  preview_only: true

authors:
- admin

tags:
- Artificial Intelligence
- Large scale deployment
- Ray Serve

categories:
# - Demo
---

## Introduction (Or rather what makes me qualified enough to write this post)

Generative AI has progressed quite significantly and so has the need to deploy it across a large number of nodes or GPUs. I have deployed various kinds of models for internal usage and rapid prototyping at Adobe and that's how I was introduced to the intersection of AI models and APIs about 2 years ago. I have wanted to write about this for a long time now, but I am a lazy writer and have prioritized other thigns over this. Until now, of course. Another reason that I am writing this post is because it was difficult for me to find an example with this exact format for deploying a model quickly. Hopefully, this will help other people who can use this for their deployments. I have also linked docs at places where people can dive deeper into themselves.

This post does not go too deep into explaining what ray or ray serve has to offer, but rather dives into how you can use ray serve to deploy a model. If you are an engineer, a researcher with some understanding about APIs, or a technical manager, you will definitely learn about something that you can potentially use for your own projects. 

All code lives in this repo! (YET TO CREATE REPO)

## A Bit about Ray Serve

If you already know what [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) is, you can skip reading this section. 

Let's say you have one or more models and you want to deploy them on a server. You can use FastAPI or Flask to load them on a specified CUDA device and access them during an API call. What makes this difficult is that you'd have to specify the device for each model and every time you want to load the same model on multiple GPUs. You'd also have to think about distributing the incoming requests evenly across all the GPUs to ensure good resource utilization. This is doable, yes. But it can get messy and tiresome real quick as your deployments grow larger. This is where Ray comes in.

Ray Serve takes care about all of the aspects (and many more) I mentioned above. All you would have to do is to specify the number of replicas of your model and the number of resources that each replica should consume on a very basic level. There are many other parameters that we can configure, some of which I will cover in the code. For more, you can dive deeper into [their amazing docs](https://docs.ray.io/en/latest/serve/key-concepts.html). 


## Model Loading and Inference

For this post, I will be using the InternVL2-8B model. You can take a look at its inference code in [this repo](https://github.com/OpenGVLab/InternVL). InternVL2 is an Open Source Multimodal LLM. My model class will be generic. Hence, you can replace InternVL2 with any model that you like by just modifying the way you load the model. Let's dive into the code! Starting off with a simple model loading and inference call using a FastAPI. that will later integrate with ray. 

```
from transformers import AutoModel, AutoTokenizer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [*]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class YourBigModel:

    def __init__(self):
        self.load_models()
    
    def load_models(self):
        path = 'OpenGVLab/InternVL2-8B'
        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        self.generation_config = dict(max_new_tokens=1024, do_sample=False)
        self.question = "<image>\nPlease give a detailed caption for the given image that covers all the objects in foreground and background. Please do not start with 'The image'."
    
    def generate_caption(self, image_path:str):
        pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
        caption = self.model.chat(self.tokenizer, pixel_values, self.question, self.generation_config)
        return caption.strip()

model = YourBigModel()

@app.post("/caption")
def get_caption(image_path:str):
    return model.generate(image_path)

```

The model loading is and inference call is taken from [the README](https://github.com/OpenGVLab/InternVL) of the original InternVL repo. The only function I haven't added here is `load_image` as it needs more code and makes this post quite long. Full code can be found in my GitHub repo. 

So far, this is a basic setup for model loading, inference and setting API endpoints. 

## Bringing in Ray Serve

Install ray serve and related dependencies by - `pip install ray ray[serve] ray[train]`.

The best thing about ray serve is that it is based on FastAPI. So there is very little modification that we have to do in our code to add ray! Pasting the same code with some modifications.

```
# caption_model.py
from transformers import AutoModel, AutoTokenizer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ray import serve
import ray.train.torch
from pydantic import BaseModel

app = FastAPI()

origins = [*]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CaptionRequest(BaseModel):
    image_path: str

@serve.deployment() # indicate that this is a ray serve deployment
@serve.ingress(app) # this is what tells ray serve that it has to take in this "app"
class YourBigModel:

    def __init__(self):
        # The generic torch.device() will not work here when you deploy across multiple GPUs. 
        # You need to use that specific device where ray deploys a replica on its own.
        self.device = ray.train.torch.get_device()
        self.load_models()
    
    def load_models(self):
        path = 'OpenGVLab/InternVL2-8B'
        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval().to(device) # send to specific device
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        self.generation_config = dict(max_new_tokens=1024, do_sample=False)
        self.question = "<image>\nPlease give a detailed caption for the given image that covers all the objects in foreground and background. Please do not start with 'The image'."
    
    @app.post("/caption")
    def generate_caption(self, request:CaptionRequest):
        image_path = request.image_path
        pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).to(device) # send to device
        caption = self.model.chat(self.tokenizer, pixel_values, self.question, self.generation_config)
        return caption.strip()

# This is the entry point of ray serve. We will use this in the deployment config.
entry = YourBigModel.bind()
```

There are various ways to start a ray server. I personally find the deployment config to be much cleaner, organized and less error prone when the number of models grow. A deployment config for the above application is below. I will explain the relevant paramters via comments as the explanation alongside the actual parameter makes it faster to understand. 

```
# caption_deploy.yaml
# Sample structure taken from - https://docs.ray.io/en/latest/serve/production-guide/config.html
proxy_location: EveryNode

http_options:
# These are basic HTTP options. You can configure them as you please.
  host: 0.0.0.0
  port: 8000

grpc_options:
  port: 9000
  grpc_servicer_functions: []

logging_config:
  encoding: TEXT
  log_level: INFO
  # This is the directory where your terminal logs will be stored for each run.
  logs_dir: null
  enable_access_log: true

applications:

- name: caption-app # give it any name

  route_prefix: /my-api # this is your API route prefix. This will be a prefix to all of your FastAPI routes.

  import_path: caption_model:entry # tells ray to go caption_mode.py file and look for the bind() method

  # Configure the runtime_env to use a specific python virtual environment.
  runtime_env: {
    "env_vars": {
        "PYTHONPATH": "/opt/venv/bin/python"
    }
  }

  # This is the part where you configure the replicas.
  deployments:

  - name: YourBigModel # This should be same as the class name where we added the ray serve annotation.

    # number of replicas, i.e. parallel deployments. You can have one model instance per GPU.
    # I believe it is possible to have multiple replicas (subject to available GPU memory) on a single GPU, but that often leads to memory issues. You can try and see how it goes.
    num_replicas: 1

    ray_actor_options:
      num_cpus: 1.0 # You do need at least 1 CPU per replica.
      num_gpus: 1.0 # Higher if one instance of your model needs more than 1 GPU.

  # You can add more models under the deployments section. 
  # You can also decouple multiple models across different files and have one config to drive the full deployment together.
  # There are many other configurations that you can dive deeper into here - https://docs.ray.io/en/latest/serve/configure-serve-deployment.html#serve-configure-deployment
```

Deploy your models by - `serve run caption_deploy.yaml`

## Final Notes

This should hopefully be enough for you to have good initial understanding about how you can use ray serve and modify it for your models. Feel free to use the code wherever you'd like. 

Lastly, I am always up for meeting new people online or in the Bay. Feel free to shoot me an email or connect on LinkedIn. In case this post brings good value to you, you can also [buy me a coffee here](https://buymeacoffee.com/jalanshmunshi)!

Happy deploying!