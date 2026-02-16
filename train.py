import os
import argparse
import torch
from torch.utils.data import DataLoader
from dataset_loaders.multisite_2_5D_dataset_loader import MultiSite_2_5D_Dataset
from utils import collate_multisite, training_loop
from models.model_2_5D import MURD_2_5D, MURDTrainer

# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():    
    parser = argparse.ArgumentParser(description='MURD 2.5D Inference')
    parser.add_argument('--experiment_name', type=str, required=True, help='Name of the experiment for logging')
    parser.add_argument('--num-sites', type=int, default=2, help='Total number of sites')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--checkpoint-save-frequency', type=int, default=10, help='Frequency of saving checkpoints (epochs)')
    
    args = parser.parse_args()

    ## DEFINE DATA PATHS HERE
    # Define sites/modalities data folders    
    folder_paths = {
        0: 'data/3T/train',
        1: 'data/7T/train'
    }
    
    image_paths = {}
    for site_idx, folder in folder_paths.items():
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder for site {site_idx} not found: {folder}")
        image_paths[site_idx] = [os.path.join(folder, path) for path in os.listdir(folder) if path.endswith('.nii.gz')]

    assert len(image_paths) == num_sites, "Number of sites in image_paths must match num_sites"

    # THIS IS TO BE CHANGED IN CAS WE EVELOP NEW MODELS
    # Configuration
    experiment_name = args.experiment_name
    num_sites = args.num_sites
    num_epochs = args.epochs
    batch_size = args.batch_size
    save_freq = args.checkpoint_save_frequency
    
    # Create model
    model = MURD_2_5D(
        in_channels=3,  # 2.5D input
        num_sites=num_sites,
        style_dim=64,
        latent_dim=16,
        base_channels=64
    )
    
    # Create trainer
    trainer = MURDTrainer(
        experiment_name=experiment_name,
        model=model,
        epochs=num_epochs,
        num_sites=num_sites,
        lr=1e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    

    dataset = MultiSite_2_5D_Dataset(image_paths, image_size=256)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_multisite,
        num_workers=4
    )
    
    training_loop(num_epochs=num_epochs, experiment_name=experiment_name, save_freq=save_freq, trainer=trainer, dataloader=dataloader)


if __name__ == '__main__':
    main()