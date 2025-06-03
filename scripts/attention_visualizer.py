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
    def __init__(self, model_config="/fs/nexus-scratch/mrislam/stablediffusion/stablediffusion/configs/stable-diffusion/v2-inference-v.yaml", ckpt="/fs/nexus-scratch/mrislam/v2-1_768-nonema-pruned.ckpt"):
        config = OmegaConf.load(model_config)
        model = instantiate_from_config(config.model)
        
        state_dict = torch.load(ckpt)["state_dict"]
        model.load_state_dict(state_dict)
        
        self.model = model.cuda()
        self.model.eval()
        # Convert model to float32
        self.model = self.model.float()
        for param in self.model.parameters():
            param.data = param.data.float()
        self.sampler = DDIMSampler(self.model)
        
        # Create output directory
        self.output_dir = "attention_maps"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def get_attention_maps(self, prompt, num_inference_steps=50, guidance_scale=7.5, save_frequency=5):
        # Initialize variables
        height = 512
        width = 512
        channels = 4
        batch_size = 1
        device = "cuda"
        
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            # Encode the prompt
            c = self.model.get_learned_conditioning([prompt])
            uc = self.model.get_learned_conditioning([""])
            
            # Initialize sampling
            shape = [channels, height // 8, width // 8]
            self.sampler.make_schedule(ddim_num_steps=num_inference_steps, ddim_eta=0.0, verbose=False)
            
            # Sample with attention map extraction
            x = torch.randn([batch_size] + shape, device=device, dtype=torch.float32)
            attention_maps = []
            
            def attention_hook(module, input, output):
                # The output is the attention tensor itself
                attention_probs = output
                
                # Reshape the attention tensor to get the attention map
                if len(attention_probs.shape) == 3:
                    attention_probs = attention_probs.unsqueeze(1)  # Add head dimension
                
                attention_maps.append(attention_probs.detach().cpu())
            
            # Register hooks for all cross attention layers
            hooks = []
            for name, module in self.model.model.diffusion_model.named_modules():
                if "attn2" in name:  # Cross attention layers
                    hooks.append(module.register_forward_hook(attention_hook))
            
            try:
                for i, t in enumerate(self.sampler.ddim_timesteps):
                    print(f"Timestep {i}/{len(self.sampler.ddim_timesteps)}")
                    
                    # Convert timestep to tensor and ensure int64
                    t = torch.tensor([t], device=device, dtype=torch.int64)
                    
                    # Ensure x is a tensor and not a tuple
                    if isinstance(x, tuple):
                        x = x[0]
                    
                    # Expand the latents for classifier free guidance
                    x_in = torch.cat([x] * 2)
                    t_in = torch.cat([t] * 2)
                    
                    # Ensure inputs are float32
                    x_in = x_in.float()
                    t_in = t_in.float()
                    c = c.float()
                    uc = uc.float()
                    
                    # Predict the noise residual
                    noise_pred = self.model.apply_model(x_in, t_in, torch.cat([uc, c]))
                    
                    # Perform guidance
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    
                    # Sample with proper parameters
                    x = self.sampler.p_sample_ddim(
                        x=x,
                        c=c,
                        t=t,
                        index=i,
                        unconditional_guidance_scale=guidance_scale,
                        unconditional_conditioning=uc
                    )
                    
                    # Ensure x is a tensor after sampling
                    if isinstance(x, tuple):
                        x = x[0]
                    
                    # Save attention maps every save_frequency steps
                    if len(attention_maps) > 0 and i % save_frequency == 0:
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
        
        # Create a single figure for all layers
        n_layers = len(attention_maps)
        n_cols = min(4, n_layers)  # Show at most 4 layers per row
        n_rows = (n_layers + n_cols - 1) // n_cols
        
        plt.figure(figsize=(5*n_cols, 5*n_rows))
        
        for layer_idx, attn_map in enumerate(attention_maps):
            if len(attn_map.shape) != 4:
                continue
                
            # Average attention maps across heads
            avg_attention = attn_map.mean(1)[0]  # [seq_len, seq_len]
            
            # Plot attention map
            plt.subplot(n_rows, n_cols, layer_idx + 1)
            plt.imshow(avg_attention.numpy(), cmap='viridis')
            plt.title(f"Layer {layer_idx}")
            plt.axis('off')
            
            # Save raw attention weights
            np.save(os.path.join(timestep_dir, f"layer_{layer_idx:02d}_raw.npy"), avg_attention.numpy())
        
        plt.suptitle(f"Attention Maps at Timestep {timestep}\nPrompt: {prompt}")
        plt.tight_layout()
        plt.savefig(os.path.join(timestep_dir, "all_layers.png"))
        plt.close()

def main():
    # Initialize visualizer
    visualizer = AttentionVisualizer()
    
    # Generate and visualize attention maps
    prompt = "a beautiful sunset over mountains"
    visualizer.get_attention_maps(prompt, num_inference_steps=50, save_frequency=5)
    print("Attention maps have been saved to the 'attention_maps' directory")

if __name__ == "__main__":
    main() 