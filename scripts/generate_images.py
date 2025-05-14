#!/usr/bin/env python

import os
import sys
import argparse
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

# Add parent directory to path to import from pallets
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import pallets modules
from pallets import (
    images as I,
    datasets as DS,
    models as M,
    logging as L,
    paths as P
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate images from trained models")
    
    parser.add_argument(
        "--model", "-m", 
        required=True,
        help="Name of the model to load (without file extension)"
    )
    
    parser.add_argument(
        "--count", "-c", 
        type=int, 
        default=1,
        help="Number of images to generate"
    )
    
    parser.add_argument(
        "--output", "-o", 
        default="generated",
        help="Output directory for generated images"
    )
    
    parser.add_argument(
        "--latent-dim", "-l", 
        type=int, 
        default=32,
        help="Dimension of the latent space"
    )
    
    parser.add_argument(
        "--seed", "-s", 
        type=int, 
        default=None,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--interpolate", "-i", 
        action="store_true",
        help="Generate interpolations between random points"
    )
    
    parser.add_argument(
        "--interpolation-steps", 
        type=int, 
        default=10,
        help="Number of steps for interpolation"
    )
    
    parser.add_argument(
        "--gpu", "-g", 
        action="store_true",
        help="Use GPU if available"
    )
    
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()


def setup_environment(args):
    """Setup the environment for image generation."""
    # Initialize logger
    log_level = "INFO" if args.verbose else "WARNING"
    logger = L.init_logger(level=log_level)
    
    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")
    
    # Set device
    device = M.get_device(require_gpu=args.gpu)
    logger.info(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    logger.info(f"Output directory: {args.output}")
    
    return logger, device


def load_model(model_name, device):
    """Load the model from the saved directory."""
    try:
        model, train_losses, test_losses = M.load(model_name, device)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_name}': {e}")


def get_model_type(model):
    """Determine the type of the model for generation."""
    model_name = model.__class__.__name__
    
    if "VAE" in model_name:
        if "Gumbel" in model_name:
            return "gumbel"
        elif "CVAE" in model_name:
            return "cvae"
        else:
            return "vae"
    elif "AE" in model_name:
        return "ae"
    else:
        return "unknown"


def setup_color_mapper():
    """Setup the color mapper for one-hot encoding."""
    all_colors = I.get_punk_colors()
    mapper = DS.ColorOneHotMapper(all_colors)
    return mapper


def generate_random_latent(batch_size, latent_dim, device):
    """Generate random latent vectors."""
    return torch.randn(batch_size, latent_dim).to(device)


def generate_interpolation(start_point, end_point, steps):
    """Generate interpolation between two points in latent space."""
    alphas = torch.linspace(0, 1, steps).unsqueeze(1)
    interpolations = start_point * (1 - alphas) + end_point * alphas
    return interpolations


def decode_image(model, latent_vector, mapper, model_type):
    """Decode a latent vector into an image."""
    with torch.no_grad():
        model.eval()
        
        if model_type == "ae":
            # For standard autoencoders
            output = model.decode(latent_vector)
        elif model_type == "vae":
            # For VAEs
            output = model.decode(latent_vector)
        elif model_type == "cvae":
            # For CVAEs, we need to handle conditional input
            # This is a simplification; adapt based on your model architecture
            output = model.decode(latent_vector)
        elif model_type == "gumbel":
            # For Gumbel-Softmax models
            output = model.decoder(latent_vector)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    # For models with one-hot output (most models in this project)
    if output.dim() == 4 and output.shape[1] == 222:  # One-hot channels
        # Get the first image in batch
        onehot_image = output[0].view(222, 24, 24)
        # Convert from one-hot to RGBA
        rgba_image = DS.onehot.one_hot_to_rgba(onehot_image, mapper)
        return rgba_image
    elif output.dim() == 4 and output.shape[1] == 4:  # RGBA channels
        # Already in RGBA format
        return output[0]
    else:
        raise ValueError(f"Unexpected output shape: {output.shape}")


def save_image(image_tensor, output_dir, index, prefix=""):
    """Save image tensor as PNG file."""
    pil_image = to_pil_image(image_tensor)
    filename = os.path.join(output_dir, f"{prefix}{index:04d}.png")
    pil_image.save(filename)
    return filename


def generate_images(args, logger, device):
    """Generate images based on command line arguments."""
    # Load model
    model = load_model(args.model, device)
    model_type = get_model_type(model)
    logger.info(f"Model type detected: {model_type}")
    
    # Get color mapper
    mapper = setup_color_mapper()
    
    # Determine actual latent dimension (may be different from args.latent_dim)
    try:
        if hasattr(model, 'latent_dim'):
            latent_dim = model.latent_dim
            logger.info(f"Using model's latent dimension: {latent_dim}")
        else:
            latent_dim = args.latent_dim
            logger.info(f"Using provided latent dimension: {latent_dim}")
    except:
        latent_dim = args.latent_dim
        logger.info(f"Falling back to provided latent dimension: {latent_dim}")
    
    # Handle special case for Gumbel models
    if model_type == "gumbel" and "Labeled" in model.__class__.__name__:
        # Labeled Gumbel models might need different latent dimensions
        latent_dim = latent_dim * 92  # Typical dimension for labeled models
        logger.info(f"Adjusted latent dimension for labeled model: {latent_dim}")
    
    generated_files = []
    
    if args.interpolate:
        # Generate interpolation between two random points
        logger.info("Generating interpolation images")
        start_point = generate_random_latent(1, latent_dim, device)
        end_point = generate_random_latent(1, latent_dim, device)
        
        interpolations = generate_interpolation(
            start_point, end_point, args.interpolation_steps
        )
        
        for i, latent in enumerate(interpolations):
            image = decode_image(model, latent.unsqueeze(0), mapper, model_type)
            filename = save_image(image, args.output, i, prefix="interp_")
            generated_files.append(filename)
            logger.info(f"Generated interpolation image {i+1}/{args.interpolation_steps}: {filename}")
    else:
        # Generate individual random images
        logger.info(f"Generating {args.count} random images")
        for i in range(args.count):
            latent = generate_random_latent(1, latent_dim, device)
            image = decode_image(model, latent, mapper, model_type)
            filename = save_image(image, args.output, i)
            generated_files.append(filename)
            logger.info(f"Generated image {i+1}/{args.count}: {filename}")
    
    return generated_files


def main():
    """Main entry point."""
    args = parse_args()
    logger, device = setup_environment(args)
    
    try:
        generated_files = generate_images(args, logger, device)
        logger.info(f"Successfully generated {len(generated_files)} images")
        
        # Print the first few generated files for user convenience
        if generated_files:
            print("Generated images:")
            for i, f in enumerate(generated_files[:5]):
                print(f"  {f}")
            if len(generated_files) > 5:
                print(f"  ... and {len(generated_files) - 5} more")
    
    except Exception as e:
        logger.error(f"Error generating images: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()