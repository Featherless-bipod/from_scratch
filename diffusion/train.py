import os
import copy
import argparse
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
from torchvision.utils import save_image

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

from model import MicroDiT, Discriminator
from frameworks_flow import FlowMatcher
from frameworks_bridge import SchrodingerBridge
from frameworks_score import StandardDiffusion
from data import get_dataloader

def get_args():
    parser = argparse.ArgumentParser(description="Generative BME Training Hub")
    parser.add_argument("--mode", type=str, default="flow", 
                        choices=["flow", "gan", "bridge", "diffusion"], 
                        help="Choose the mathematical framework.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    # Bridge specific
    parser.add_argument("--source_class", type=int, default=3, help="Source class for Bridge")
    parser.add_argument("--target_class", type=int, default=5, help="Target class for Bridge")
    return parser.parse_args()

def main():
    args = get_args()
    print(f"--- Booting up in {args.mode.upper()} mode ---")

    # 1. hardware
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Apple Silicon (MPS) detected. Engaging unified memory.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("NVIDIA GPU (CUDA) detected. Engaging unified memory.")
    else:
        device = torch.device("cpu")
        print("Warning: MPS not found. Running on CPU.")

    # 2. data loading
    if args.mode == "bridge":
        dataloader = get_dataloader(batch_size=args.batch_size, 
                                    source_class=args.source_class, 
                                    target_class=args.target_class)
    else:
        dataloader = get_dataloader(batch_size=args.batch_size)

    # 3. model + framework
    model = MicroDiT().to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    opt_G = AdamW(model.parameters(), lr=args.lr, betas=(0.5, 0.999))

    if args.mode == "bridge":
        framework = SchrodingerBridge()
    elif args.mode == "diffusion":
        framework = StandardDiffusion()
    else:
        framework = FlowMatcher()

    if args.mode == "gan":
        discriminator = Discriminator().to(device)
        # TTUR: Discriminator trains significantly slower than Generator
        opt_D = AdamW(discriminator.parameters(), lr=args.lr / 4, betas=(0.5, 0.999))
    else:
        discriminator = None
        opt_D = None

    ema = EMA(model) if args.mode != "gan" else None
    scheduler = OneCycleLR(opt_G, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(dataloader))
    scheduler_D = OneCycleLR(opt_D, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(dataloader)) if args.mode == "gan" else None

    # checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 4. training loop
    loss_history_G = []
    loss_history_D = []

    for epoch in range(args.epochs):
        model.train()
        total_loss_G = 0
        total_loss_D = 0

        for batch_idx, batch in enumerate(dataloader):
            # mode based data
            if args.mode == "bridge":
                x_input, targets = batch[0].to(device), batch[1].to(device)
                labels = torch.zeros(x_input.shape[0], dtype=torch.long, device=device)
            else:
                x_input, targets = batch[0].to(device), batch[1].to(device)
                labels = targets

            # calculate losses
            if args.mode == "flow":
                loss_G = framework.compute_loss_flow(model, x_input, labels) 
                loss_D = None

            elif args.mode == "gan":
                loss_G, loss_D = framework.compute_loss_adversarial(model, discriminator, x_input, labels)

            elif args.mode == "bridge":
                loss_G = framework.compute_loss_ot_sinkhorn(model, x_input, targets, labels)
                loss_D = None

            elif args.mode == "diffusion":
                loss_G = framework.compute_loss(model, x_input, labels)
                loss_D = None

            # optimization steps
            
            opt_G.zero_grad()
            loss_G.backward()

            if loss_D is not None:
                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()
                total_loss_D += loss_D.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt_G.step()
            
            if ema is not None:
                ema.update(model)
            scheduler.step()
            if scheduler_D is not None:
                scheduler_D.step()
            
            total_loss_G += loss_G.item()

        # logging + checkpointing
        avg_loss_G = total_loss_G / len(dataloader)
        
        loss_history_G.append(avg_loss_G)
        if args.mode == "gan":
            avg_loss_D = total_loss_D / len(dataloader)
            loss_history_D.append(avg_loss_D)
            print(f"Epoch [{epoch+1}/{args.epochs}] | Gen Loss: {avg_loss_G:.4f} | Disc Loss: {avg_loss_D:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{args.epochs}] | Loss: {avg_loss_G:.4f}")

        # save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"checkpoints/model_{args.mode}_epoch_{epoch+1}.pt"
            state_dict_to_save = ema.ema_model.state_dict() if ema else model.state_dict()
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': {k: v.cpu() for k, v in state_dict_to_save.items()},
                'optimizer_state_dict': opt_G.state_dict(),
                'loss': avg_loss_G,
            }, checkpoint_path)
            print(f"--> Checkpoint saved: {checkpoint_path}")

        # plotting and saving image grids
        if (epoch + 1) % 5 == 0 or epoch == 0 or (epoch + 1) == args.epochs:
            # 1. Save Loss Chart
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, epoch + 2), loss_history_G, label="Gen Loss")
            if args.mode == "gan":
                plt.plot(range(1, epoch + 2), loss_history_D, label="Disc Loss", color='orange')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Training Loss ({args.mode.upper()})")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"outputs/loss_chart_{args.mode}.png")
            plt.close()

            # 2. Generate and Save Test Images
            print("--> Generating sample images to monitor capabilities...")
            gen_model = ema.ema_model if ema else model
            sample_labels = (torch.arange(16) % 10).to(device) # fixed 16 labels [0,1,2,..,9,0,1,2,3,4,5]
            
            with torch.no_grad():
                try:
                    if args.mode == "diffusion":
                        samples = framework.sample_simple(gen_model, num_images=16, device=device, class_labels=sample_labels)
                    elif args.mode == "flow":
                        samples = framework.sample_euler(gen_model, num_images=16, device=device, class_labels=sample_labels)
                    elif args.mode == "bridge":
                        num_bridge_samples = min(16, x_input.shape[0])
                        samples = framework.sample_euler(gen_model, x_start=x_input[:num_bridge_samples], class_labels=sample_labels[:num_bridge_samples])
                    elif hasattr(framework, 'sample_euler'): # fallback for bridge or other
                        samples = framework.sample_euler(gen_model, num_images=16, device=device, class_labels=sample_labels)
                    else:
                        samples = None
                        
                    if samples is not None:
                        save_image(samples, f"outputs/samples_{args.mode}_epoch_{epoch+1}.png", nrow=4, normalize=True, value_range=(-1, 1))
                except Exception as e:
                    print(f"Could not generate visual samples: {e}")

if __name__ == "__main__":
    main()