import os
import sys
import clip
import torch
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
import warnings
import logging

# Configure logging
logging.basicConfig(filename='clip_text_features.log',
                    filemode='w',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress warnings
warnings.filterwarnings("ignore")

def get_categories_list(csv_file_path):
    """
    Reads the CSV file header and returns a list of category names.

    Args:
        csv_file_path (str): Path to the CSV file.

    Returns:
        list: List of category names.
    """
    df = pd.read_csv(csv_file_path, nrows=0)
    columns = df.columns.tolist()[1:]  # Exclude the first column ('path')
    return columns

def main():
    # Configuration
    # Define the comprehensive list of CLIP models
    clip_models = [
        'RN50',
        'RN101',
        'RN50x4',
        'RN50x16',
        'RN50x64',
        'ViT-B/32',
        'ViT-B/16',
        'ViT-L/14',
        'ViT-L/14@336px'
    ]

    # Define paths
    csv_filepath = '/mnt/disk/sxyy/code_proj/trash_class/csv/all.csv'
    npy_save_path = '/mnt/disk/sxyy/code_proj/trash_class/text_features'
    os.makedirs(npy_save_path, exist_ok=True)  # Create the save directory if it doesn't exist

    # Read the CSV file headers to get categories
    try:
        categories = get_categories_list(csv_filepath)
        logging.info(f"Categories detected: {categories}")
        print(f"Categories detected: {categories}")
    except Exception as e:
        logging.error(f"Error reading CSV file headers at {csv_filepath}: {e}")
        sys.exit(f"Error reading CSV file headers at {csv_filepath}: {e}")

    # Map each category to its description for text prompts
    category_descriptions = {
        "Harmful_waste": "batteries, expired medication inner package",
        "Recyclable_waste": "pull-tab cans, bottles",
        "Food_waste": "small potatoes, white radishes, carrots",
        "Other_waste": "porcelain shards, pebbles, bricks"
    }

    # Verify that all categories have descriptions
    for category in categories:
        if category not in category_descriptions:
            logging.error(f"No description provided for category: {category}")
            sys.exit(f"No description provided for category: {category}")

    # Define text prompts based on categories
    text_prompts = [
        f"a photo of {category.replace('_', ' ')}: {desc}"
        for category, desc in category_descriptions.items()
    ]

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Device: {device}')
    logging.info(f'Device selected for computation: {device}')

    # Iterate over each CLIP model to compute and save text features
    for model_name in clip_models:
        print(f'\nProcessing text features with CLIP model: {model_name}')
        logging.info(f'Processing text features with CLIP model: {model_name}')

        try:
            # Load the CLIP model and preprocessing function
            model, preprocess = clip.load(model_name, device=device)
            logging.info(f'Successfully loaded model: {model_name}')
        except Exception as e:
            logging.error(f'Error loading model {model_name}: {e}')
            print(f'Error loading model {model_name}: {e}')
            continue  # Skip to the next model

        try:
            # Tokenize the text prompts
            text_tokens = clip.tokenize(text_prompts).to(device)

            # Encode text features
            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)

            # Convert text features to CPU and NumPy
            text_features_np = text_features.cpu().numpy()  # Shape: (num_categories, feature_dim)

            # Define a unique save name for each model
            save_filename = f'clip_{model_name.replace("/", "-")}_text_features.npy'
            save_filepath = os.path.join(npy_save_path, save_filename)

            # Save the text features to a .npy file
            np.save(save_filepath, text_features_np)
            logging.info(f'Saved text features to {save_filepath}')
            print(f'Saved text features to {save_filepath}')

        except Exception as e:
            logging.error(f'Error processing text for model {model_name}: {e}')
            print(f'Error processing text for model {model_name}: {e}')
            continue  # Skip to the next model

        # Clear CUDA cache to free memory
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\nText features extraction and saving completed successfully!")
    logging.info("Text features extraction and saving completed successfully!")

if __name__ == "__main__":
    main()
