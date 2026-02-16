import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
import csv
import json
from datetime import datetime

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns


class LossLogger:
    """
    Logger for tracking and saving training losses
    """
    def __init__(self, log_dir: str, experiment_name: str = None):
        """
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment (timestamp if None)
        """
        self.log_dir = os.path.join(log_dir, experiment_name) if experiment_name else log_dir
        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        
        # Initialize storage
        self.batch_losses = []  # List of dicts for each batch
        self.epoch_losses = []  # List of dicts for each epoch
        
        # CSV files
        self.batch_csv = self.log_dir / f"batch_losses.csv"
        self.epoch_csv = self.log_dir / f"epoch_losses.csv"
        
        # Initialize CSV files with headers
        self.batch_csv_initialized = False
        self.epoch_csv_initialized = False
        
        # Config file
        self.config_file = self.log_dir / f"config.json"
        
        print(f"✓ Loss logger initialized")
        print(f"  Batch log: {self.batch_csv}")
        print(f"  Epoch log: {self.epoch_csv}")
    
    def log_batch(self, epoch: int, batch_idx: int, losses: Dict[str, float]):
        """Log losses for a single batch"""
        log_entry = {
            'epoch': epoch,
            'batch': batch_idx,
            'global_step': len(self.batch_losses),
            **losses
        }
        
        self.batch_losses.append(log_entry)
        
        # Write to CSV
        if not self.batch_csv_initialized:
            # Write header
            with open(self.batch_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_entry.keys())
                writer.writeheader()
                writer.writerow(log_entry)
            self.batch_csv_initialized = True
        else:
            # Append row
            with open(self.batch_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_entry.keys())
                writer.writerow(log_entry)
    
    def log_epoch(self, epoch: int, avg_losses: Dict[str, float], 
                  epoch_time: float = None, lr: float = None):
        """Log average losses for an epoch"""
        log_entry = {
            'epoch': epoch,
            **avg_losses
        }
        
        if epoch_time is not None:
            log_entry['epoch_time'] = epoch_time
        if lr is not None:
            log_entry['learning_rate'] = lr
        
        self.epoch_losses.append(log_entry)
        
        # Write to CSV
        if not self.epoch_csv_initialized:
            with open(self.epoch_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_entry.keys())
                writer.writeheader()
                writer.writerow(log_entry)
            self.epoch_csv_initialized = True
        else:
            with open(self.epoch_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_entry.keys())
                writer.writerow(log_entry)
    
    def save_config(self, config: dict):
        """Save training configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Saved config to {self.config_file}")
    
    def get_batch_dataframe(self) -> pd.DataFrame:
        """Get batch losses as pandas DataFrame"""
        return pd.read_csv(self.batch_csv)
    
    def get_epoch_dataframe(self) -> pd.DataFrame:
        """Get epoch losses as pandas DataFrame"""
        return pd.read_csv(self.epoch_csv)
    
    def plot_losses(self, save_path: str = None, show: bool = True):
        """
        Create comprehensive loss plots
        
        Args:
            save_path: Path to save figure (optional)
            show: Whether to display the plot
        """
        df_epoch = self.get_epoch_dataframe()
        
        # Identify loss columns (exclude epoch, time, lr)
        loss_cols = [col for col in df_epoch.columns 
                     if col not in ['epoch', 'epoch_time', 'learning_rate']]
        
        # Separate generator and discriminator losses
        gen_losses = [col for col in loss_cols if col.startswith('gen_')]
        disc_losses = [col for col in loss_cols if col.startswith('disc')]
        
        # Create figure
        n_plots = 2 + len(gen_losses)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4*n_plots))
        
        # Plot 1: Total generator loss
        if 'gen_total' in df_epoch.columns:
            axes[0].plot(df_epoch['epoch'], df_epoch['gen_total'], 
                        linewidth=2, color='blue', label='Generator Total')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Generator Total Loss', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
        
        # Plot 2: Discriminator loss
        if 'disc' in df_epoch.columns:
            axes[1].plot(df_epoch['epoch'], df_epoch['disc'], 
                        linewidth=2, color='red', label='Discriminator')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Discriminator Loss', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
        
        # Plot 3+: Individual generator losses
        colors = plt.cm.tab10(np.linspace(0, 1, len(gen_losses)))
        for idx, (loss_name, color) in enumerate(zip(gen_losses, colors)):
            if loss_name == 'gen_total':
                continue
            if loss_name in df_epoch.columns:
                ax = axes[2 + idx] if idx < len(axes) - 2 else axes[-1]
                ax.plot(df_epoch['epoch'], df_epoch[loss_name], 
                       linewidth=2, color=color, label=loss_name)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title(f'{loss_name.replace("gen_", "").upper()} Loss', 
                           fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_training_losses(csv_path: str, save_path: str = None, style: str = 'darkgrid'):
    """
    Create publication-quality loss plots using seaborn
    
    Args:
        csv_path: Path to epoch losses CSV file
        save_path: Path to save figure
        style: Seaborn style ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks')
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Set style
    sns.set_style(style)
    sns.set_context("paper", font_scale=1.5)
    
    # Identify loss columns
    loss_cols = [col for col in df.columns 
                 if col not in ['epoch', 'epoch_time', 'learning_rate']]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: All Generator Losses
    ax1 = fig.add_subplot(gs[0, :])
    gen_cols = [col for col in loss_cols if col.startswith('gen_') and col != 'gen_total']
    for col in gen_cols:
        if col in df.columns:
            label = col.replace('gen_', '').upper()
            ax1.plot(df['epoch'], df[col], linewidth=2, label=label, marker='o', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax1.set_title('Generator Loss Components', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Total Generator Loss
    ax2 = fig.add_subplot(gs[1, 0])
    if 'gen_total' in df.columns:
        sns.lineplot(data=df, x='epoch', y='gen_total', ax=ax2, 
                    linewidth=2.5, color='#2E86AB', marker='o')
        ax2.fill_between(df['epoch'], df['gen_total'], alpha=0.3, color='#2E86AB')
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Generator Total Loss', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Discriminator Loss
    ax3 = fig.add_subplot(gs[1, 1])
    if 'disc' in df.columns:
        sns.lineplot(data=df, x='epoch', y='disc', ax=ax3, 
                    linewidth=2.5, color='#A23B72', marker='o')
        ax3.fill_between(df['epoch'], df['disc'], alpha=0.3, color='#A23B72')
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax3.set_title('Discriminator Loss', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Key Losses (Content, Style, Identity)
    ax4 = fig.add_subplot(gs[2, 0])
    key_losses = ['gen_cont', 'gen_sty', 'gen_id']
    for loss in key_losses:
        if loss in df.columns:
            label = loss.replace('gen_', '').upper()
            ax4.plot(df['epoch'], df[loss], linewidth=2, label=label, marker='o', markersize=4)
    ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax4.set_title('Key Reconstruction Losses', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Adversarial Loss
    ax5 = fig.add_subplot(gs[2, 1])
    if 'gen_adv' in df.columns:
        sns.lineplot(data=df, x='epoch', y='gen_adv', ax=ax5, 
                    linewidth=2.5, color='#F18F01', marker='o')
    ax5.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax5.set_title('Adversarial Loss', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    plt.suptitle('MURD Training Loss Progression', fontsize=18, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")
    
    plt.show()
    return fig


def plot_loss_comparison(csv_paths: List[str], labels: List[str], 
                         loss_name: str = 'gen_total', save_path: str = None):
    """
    Compare losses across multiple experiments
    
    Args:
        csv_paths: List of paths to epoch loss CSV files
        labels: Labels for each experiment
        loss_name: Name of loss to compare
        save_path: Path to save figure
    """
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for csv_path, label in zip(csv_paths, labels):
        df = pd.read_csv(csv_path)
        if loss_name in df.columns:
            ax.plot(df['epoch'], df[loss_name], linewidth=2.5, 
                   label=label, marker='o', markersize=4, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax.set_title(f'{loss_name.replace("_", " ").title()} Comparison', 
                fontsize=16, fontweight='bold')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison plot to {save_path}")
    
    plt.show()
    return fig


def plot_loss_heatmap(csv_path: str, save_path: str = None):
    """
    Create a heatmap of loss correlations
    
    Args:
        csv_path: Path to epoch losses CSV
        save_path: Path to save figure
    """
    df = pd.read_csv(csv_path)
    
    # Select only loss columns
    loss_cols = [col for col in df.columns 
                 if col not in ['epoch', 'epoch_time', 'learning_rate']]
    loss_df = df[loss_cols]
    
    # Compute correlation matrix
    corr = loss_df.corr()
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, 
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Loss Correlation Matrix', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved heatmap to {save_path}")
    
    plt.show()
    return fig


def create_training_report(log_dir: str, experiment_name: str):
    """
    Create a comprehensive training report with multiple visualizations
    
    Args:
        log_dir: Directory containing logs
        experiment_name: Name of the experiment
    """
    log_dir = os.path.join(log_dir, experiment_name)
    epoch_csv = Path(log_dir) / f"epoch_losses.csv"
    
    if not epoch_csv.exists():
        print(f"Error: {epoch_csv} not found!")
        return
    
    # Load data
    df = pd.read_csv(epoch_csv)
    
    # Create output directory for plots
    plot_dir = Path(log_dir) / "plots"
    plot_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Creating Training Report for: {experiment_name}")
    print(f"{'='*60}\n")
    
    # 1. Main training plot
    print("1. Generating main training plot...")
    plot_training_losses(
        str(epoch_csv),
        save_path=str(plot_dir / "training_losses.png")
    )
    
    # 2. Loss correlation heatmap
    print("2. Generating loss correlation heatmap...")
    plot_loss_heatmap(
        str(epoch_csv),
        save_path=str(plot_dir / "loss_correlations.png")
    )
    
    # 3. Summary statistics
    print("3. Computing summary statistics...")
    summary_file = plot_dir / "summary_statistics.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Training Summary for {experiment_name}\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total Epochs: {len(df)}\n")
        if 'epoch_time' in df.columns:
            f.write(f"Total Training Time: {df['epoch_time'].sum()/3600:.2f} hours\n")
            f.write(f"Average Epoch Time: {df['epoch_time'].mean():.2f} seconds\n\n")
        
        f.write("Final Loss Values:\n")
        f.write("-"*60 + "\n")
        last_row = df.iloc[-1]
        for col in df.columns:
            if col not in ['epoch', 'epoch_time', 'learning_rate']:
                f.write(f"  {col}: {last_row[col]:.6f}\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"✓ Saved summary statistics to {summary_file}")
    print(f"\n✓ Training report complete! Plots saved to {plot_dir}/")