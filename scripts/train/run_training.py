import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel
import torch.nn.functional as F
from datasets import load_dataset
from torchvision import transforms
from sklearn.model_selection import KFold

from src.ambiguity_network import AmbiguityPredictor
from src.data_loader import AmbiguityDataset
from src.train import train_ambiguity_network

def prepare_stable_diffusion_inputs(caption, sd_text_encoder):
    """
    Prepare text inputs for Stable Diffusion text encoder
    
    Args:
        caption (str): Input text caption
        sd_text_encoder (CLIPTextModel): Stable Diffusion text encoder
    
    Returns:
        torch.Tensor: Text encoding
    """
    # Tokenize and encode caption
    inputs = sd_text_encoder.tokenizer(
        caption, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    )
    text_embeddings = sd_text_encoder(inputs.input_ids)[0]
    return text_embeddings

def compute_reconstruction_loss(sd_pipeline, text_embeddings, input_image, num_inference_steps=50):
    """
    Compute reconstruction loss across denoising timesteps
    
    Args:
        sd_pipeline (StableDiffusionPipeline): Stable Diffusion pipeline
        text_embeddings (torch.Tensor): Text embeddings
        input_image (torch.Tensor): Input image tensor to reconstruct
        num_inference_steps (int): Number of denoising steps
    
    Returns:
        torch.Tensor: Reconstruction losses for each timestep
    """
    reconstruction_losses = []
    
    # Convert input image to latent space
    with torch.no_grad():
        latents = sd_pipeline.vae.encode(input_image).latent_dist.sample()
        latents = latents * 0.18215  # Scale latents as per SD preprocessing
    
    # Simulate denoising process
    for t in range(num_inference_steps):
        # Add noise to the latents based on the current timestep
        noise = torch.randn_like(latents)
        timestep = torch.tensor([t], dtype=torch.long)
        noisy_latents = sd_pipeline.scheduler.add_noise(latents, noise, timestep)
        
        # Predict noise
        noise_pred = sd_pipeline.unet(
            sample=noisy_latents,
            timestep=t,
            encoder_hidden_states=text_embeddings
        ).sample
        
        # Compute reconstruction loss (MSE between predicted and actual noise)
        reconstruction_loss = F.mse_loss(noise_pred, noise)
        reconstruction_losses.append(reconstruction_loss)
    
    return torch.stack(reconstruction_losses)

def load_laion_dataset(split='train', max_samples=1000):
    """
    Load LAION dataset and preprocess images and captions
    
    Args:
        split (str): Dataset split to load
        max_samples (int): Maximum number of samples to load
    
    Returns:
        torch.utils.data.Dataset: Preprocessed LAION dataset with images and captions
    """
    # Load LAION dataset (replace with appropriate dataset name)
    dataset = load_dataset("laion/laion2B-en", split=split)
    
    # Image preprocessing transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),  # Resize to stable diffusion input size
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1] range
    ])
    
    # Limit dataset size
    dataset = dataset.select(range(min(len(dataset), max_samples)))
    
    def preprocess_sample(example):
        """Preprocess image and caption from LAION dataset"""
        # Adjust these keys based on the exact LAION dataset structure
        image = example['image']
        caption = example.get('caption', 'No caption available')
        
        return {
            'image': transform(image),
            'caption': caption
        }
    
    # Apply preprocessing
    dataset = dataset.map(preprocess_sample, remove_columns=dataset.column_names)
    
    return dataset

def load_flickr30k_dataset(split="train", n_splits=5, max_samples=None):
    """Load and preprocess Flickr30k dataset with cross-validation support.

    Args:
        split (str): 'train', 'val', or 'test'
        n_splits (int): Number of cross-validation folds
        max_samples (int, optional): Maximum number of samples to use

    Returns:
        torch.utils.data.Dataset: Preprocessed Flickr30k dataset with images and captions
    """
    # Load the entire test dataset
    full_dataset = load_dataset("nlphuji/flickr30k", split="test")
    
    # Print column names and first example for debugging
    print("Dataset columns:", full_dataset.column_names)
    print("First example:", full_dataset[0])
    
    # Limit samples if specified
    if max_samples:
        full_dataset = full_dataset.select(range(min(len(full_dataset), max_samples)))
    
    # Prepare transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Preprocess function
    def preprocess_sample(example):
        # Dynamically handle different dataset structures
        image = example['image']
        
        # Try different ways to extract caption
        if 'sentences' in example and example['sentences']:
            caption = example['sentences'][0]['raw']
        elif 'caption' in example:
            caption = example['caption']
        else:
            caption = "No caption"
        
        return {
            'image': transform(image),
            'caption': caption
        }
    
    # Perform cross-validation split
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    indices = list(range(len(full_dataset)))
    
    # Determine which fold to use based on split
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices), 1):
        if split == "train" and fold != n_splits:
            # Use all but the last fold for training
            dataset = full_dataset.select(train_idx)
            break
        elif split == "val" and fold == n_splits:
            # Use the last fold for validation
            dataset = full_dataset.select(val_idx)
            break
        elif split == "test":
            # If test is requested, return the full dataset
            dataset = full_dataset
            break
    
    # Apply preprocessing
    dataset = dataset.map(preprocess_sample, remove_columns=dataset.column_names)
    
    return dataset

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pre-trained Stable Diffusion components
    sd_pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
    sd_text_encoder = sd_pipeline.text_encoder.to(device)
    
    # Initialize models
    ambiguity_model = AmbiguityPredictor().to(device)
    
    # Optimizer and loss function
    optimizer = optim.Adam(ambiguity_model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    
    # # Load LAION dataset
    # laion_dataset = load_laion_dataset()
    # dataloader = DataLoader(laion_dataset, batch_size=8, shuffle=True)
    
    # Load Flickr30k dataset (smaller than LAION)
    flickr_dataset = load_flickr30k_dataset()
    dataloader = DataLoader(flickr_dataset, batch_size=8, shuffle=True)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for batch in dataloader:
            input_images = batch['image']  # Images from LAION dataset
            captions = batch['caption']  # Use actual captions from the dataset
            
            for caption, input_image in zip(captions, input_images):
                # Prepare text embeddings
                text_embeddings = prepare_stable_diffusion_inputs(caption, sd_text_encoder)
                
                # Compute reconstruction losses across timesteps
                reconstruction_losses = compute_reconstruction_loss(
                    sd_pipeline, 
                    text_embeddings, 
                    input_image.unsqueeze(0)  # Add batch dimension
                )
                
                # Prepare time embeddings (normalized timesteps)
                time_embeddings = torch.linspace(0, 1, reconstruction_losses.shape[0]).unsqueeze(1)
                
                # Create dataset
                dataset = AmbiguityDataset(
                    text_encodings=text_embeddings.repeat(reconstruction_losses.shape[0], 1),
                    time_embeddings=time_embeddings,
                    ambiguity_scores=reconstruction_losses.unsqueeze(1)
                )
                dataloader_ambiguity = DataLoader(dataset, batch_size=8, shuffle=True)
                
                # Train ambiguity network
                avg_loss = train_ambiguity_network(
                    ambiguity_model, 
                    dataloader_ambiguity, 
                    criterion, 
                    optimizer, 
                    device
                )
                epoch_losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {torch.mean(torch.tensor(epoch_losses))}")
    
    # Save trained model
    torch.save(ambiguity_model.state_dict(), 'ambiguity_model.pth')

if __name__ == '__main__':
    main()