import torch
import argparse
import os
from torchvision.utils import save_image

from model import MicroDiT, Discriminator
from frameworks_score import StandardDiffusion
from frameworks_flow import FlowMatcher
from frameworks_bridge import SchrodingerBridge
from data import get_dataloader

def get_args():
    parser = argparse.ArgumentParser(description="Generative BME Inference Engine")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to your saved .pt file")
    parser.add_argument("--mode", type=str, default="flow", choices=["flow", "bridge", "diffusion"], 
                        help="Note: GAN-trained models just use 'flow' mode for sampling.")
    parser.add_argument("--sampler", type=str, default="euler", choices=["euler", "pc", "aln"],
                        help="Choose the ODE or SDE solver.")
    parser.add_argument("--num_images", type=int, default=16, help="How many images to generate")
    
    # bridge specific
    parser.add_argument("--source_class", type=int, default=3, 
                        help="The starting class if using Schrödinger Bridge (e.g., 'Healthy' class)")
    parser.add_argument("--steps", type=int, default=50, help="Number of integration steps")
    return parser.parse_args()

def main():
    args = get_args()
    print(f"--- Loading {args.mode.upper()} model using {args.sampler.upper()} sampler ---")

    # 1. device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 2. weights
    model = MicroDiT().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # CRITICAL: Turns off Dropout and BatchNorm layers for inference
    print(f"Loaded weights from Epoch {checkpoint['epoch']} (Loss: {checkpoint['loss']:.4f})")

    # 3. output
    os.makedirs("outputs", exist_ok=True)

    # 4. framework and sampling choice
    labels = torch.randint(0, 10, (args.num_images,), device=device) #random classes for conditional generation

    with torch.no_grad(): #disable gradients to save memory and speed up inference
        if args.mode == "flow":
            framework = FlowMatcher(num_sampling_steps=args.steps)
            
            # flow matching samplers expect just the model and the number of images
            if args.sampler == "euler":
                samples = framework.sample_euler(model, args.num_images, device, labels)
            elif args.sampler == "pc":
                samples = framework.sample_pc(model, args.num_images, device, labels)
            elif args.sampler == "aln":
                samples = framework.sample_aln(model, args.num_images, device, labels)

        elif args.mode == "bridge":
            framework = SchrodingerBridge(num_sampling_steps=args.steps)
            
            # get real starting image
            print(f"Fetching source images (Class {args.source_class})...")
            dataloader = get_dataloader(batch_size=args.num_images, source_class=args.source_class, target_class=args.source_class) # Dummy target just to pass init
            x_start, _ = next(iter(dataloader))
            x_start = x_start.to(device)
            
            # bridge samplers expect the starting state tensor
            if args.sampler == "euler":
                samples = framework.sample_euler(model, x_start, labels)
            elif args.sampler == "pc":
                samples = framework.sample_pc(model, x_start, labels)
            elif args.sampler == "aln":
                samples = framework.sample_aln(model, x_start, labels)

        elif args.mode == "diffusion":
            framework = StandardDiffusion(num_sampling_steps=args.steps)
            
            # diff matching samplers expect just the model and the number of images
            if args.sampler == "euler":
                samples = framework.sample_simple(model, args.num_images, device, labels)
            elif args.sampler == "pc":
                samples = framework.sample_pc(model, args.num_images, device, labels)
            elif args.sampler == "aln":
                samples = framework.sample_ALN(model, args.num_images, device, labels)

    # 5. un-normalize and save
    output_path = f"outputs/sample_{args.mode}_{args.sampler}.png"
    
    # save side by side comparison if bridge
    if args.mode == "bridge":
        comparison = torch.cat([x_start, samples], dim=0) # Stacks original on top of generated
        save_image(comparison, output_path, nrow=args.num_images, normalize=True, value_range=(-1, 1))
    else:
        save_image(samples, output_path, nrow=4, normalize=True, value_range=(-1, 1))
        
    print(f"--> Images successfully saved to {output_path}")

if __name__ == "__main__":
    main()