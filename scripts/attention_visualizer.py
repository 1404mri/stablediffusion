import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import torch.nn.functional as F
import os

class AttentionVisualizer:
    def __init__(self, model_config="configs/stable-diffusion/v1-inference.yaml", ckpt="checkpoints/sd-v1-4.ckpt"):
        config = OmegaConf.load(model_config)
        model = instantiate_from_config(config.model)
        
        state_dict = torch.load(ckpt)["state_dict"]
        model.load_state_dict(state_dict)
        
        self.model = model.cuda()
        self.model.eval()
        self.sampler = DDIMSampler(self.model)
        
        # Create output directory
        self.output_dir = "attention_maps"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def get_attention_maps(self, prompt, num_inference_steps=50, guidance_scale=7.5):
        # Initialize variables
        height = 512
        width = 512
        channels = 4
        batch_size = 1
        device = "cuda"
        
        # Encode the prompt
        c = self.model.get_learned_conditioning([prompt])
        uc = self.model.get_learned_conditioning([""])
        
        # Initialize sampling
        shape = [channels, height // 8, width // 8]
        self.sampler.make_schedule(ddim_num_steps=num_inference_steps, ddim_eta=0.0, verbose=False)
        
        # Sample with attention map extraction
        x = torch.randn([batch_size] + shape, device=device)
        attention_maps = []
        
        def attention_hook(module, input, output):
            # Extract attention maps from cross attention
            attention_probs = output[1]  # Shape: [batch, heads, sequence_length, sequence_length]
            attention_maps.append(attention_probs.detach().cpu())
        
        # Register hooks for all cross attention layers
        hooks = []
        for name, module in self.model.model.diffusion_model.named_modules():
            if "attn2" in name:  # Cross attention layers
                hooks.append(module.register_forward_hook(attention_hook))
        
        try:
            for i, t in enumerate(self.sampler.ddim_timesteps):
                print(f"Timestep {i}/{len(self.sampler.ddim_timesteps)}")
                
                # Expand the latents for classifier free guidance
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t.unsqueeze(0)] * 2).to(device)
                
                # Predict the noise residual
                noise_pred = self.model.apply_model(x_in, t_in, torch.cat([uc, c]))
                
                # Perform guidance
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                
                # Sample
                x = self.sampler.p_sample_ddim(x, t, condition=c, clip_denoised=True)
                
                # Save attention maps for this timestep
                if len(attention_maps) > 0:
                    self.save_attention_maps(attention_maps, i, prompt)
                attention_maps.clear()
                
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        return x
    
    def save_attention_maps(self, attention_maps, timestep, prompt):
        # Create directory for this timestep
        timestep_dir = os.path.join(self.output_dir, f"timestep_{timestep:03d}")
        os.makedirs(timestep_dir, exist_ok=True)
        
        for layer_idx, attn_map in enumerate(attention_maps):
            # Average attention maps across heads
            avg_attention = attn_map.mean(1)  # Average across heads
            
            # Plot attention map
            plt.figure(figsize=(10, 10))
            plt.imshow(avg_attention[0].numpy(), cmap='viridis')
            plt.title(f"Attention Map - Layer {layer_idx}")
            plt.colorbar()
            plt.savefig(os.path.join(timestep_dir, f"layer_{layer_idx:02d}.png"))
            plt.close()

def main():
    # Initialize visualizer
    visualizer = AttentionVisualizer()
    
    # Generate and visualize attention maps
    prompt = "a beautiful sunset over mountains"
    visualizer.get_attention_maps(prompt, num_inference_steps=50)
    print("Attention maps have been saved to the 'attention_maps' directory")

if __name__ == "__main__":
    main() 