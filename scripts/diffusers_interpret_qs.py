import torch
from diffusers import StableDiffusionPipeline
from diffusers_interpret import StableDiffusionPipelineExplainer

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    use_auth_token=True,
    revision='fp16',
    torch_dtype=torch.float16
).to('cuda')

# optional: reduce memory requirement with a speed trade off 
pipe.enable_attention_slicing()

# pass pipeline to the explainer class
explainer = StableDiffusionPipelineExplainer(pipe)

# generate an image with `explainer`
prompt = "A cute corgi with the Eiffel Tower in the background"
with torch.autocast('cuda'):
    output = explainer(
        prompt, 
        num_inference_steps=15
    )
