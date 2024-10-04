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
logging.basicConfig(filename='clip_inference.log',
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
        # 'RN50',
        # 'RN101',
        # 'RN50x4',
        # 'RN50x16',
        # 'RN50x64',
        'ViT-B/32',
        'ViT-B/16',
        'ViT-L/14',
        'ViT-L/14@336px'
    ]

    # Define paths
    csv_filepath = '/mnt/disk/sxyy/code_proj/trash_class/csv/all.csv'
    npy_load_path = '/mnt/disk/sxyy/code_proj/trash_class/text_features'
    npy_save_path = '/mnt/disk/sxyy/code_proj/trash_class/zeroshot'
    os.makedirs(npy_save_path, exist_ok=True)  # Create the save directory if it doesn't exist

    # Read the CSV file
    try:
        true_labels_df = pd.read_csv(csv_filepath)
        logging.info(f"Successfully read CSV file from {csv_filepath}")
    except Exception as e:
        logging.error(f"Error reading CSV file at {csv_filepath}: {e}")
        sys.exit(f"Error reading CSV file at {csv_filepath}: {e}")

    # Ensure that the CSV has the necessary columns
    required_columns = ['path']  # Assuming 'path' is the first column
    for col in required_columns:
        if col not in true_labels_df.columns:
            logging.error(f"CSV file is missing the required column: {col}")
            sys.exit(f"CSV file is missing the required column: {col}")

    # Get category names
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

    # Prepare true labels
    # Assuming single-label classification: only one category is 1 per image
    # Create a list of true labels based on the category columns
    true_labels = []
    for index, row in true_labels_df.iterrows():
        label = None
        for category in categories:
            if row[category] == 1:
                label = category
                break
        if label is None:
            logging.warning(f"No label found for image: {row['path']}")
            label = "Unknown"
        true_labels.append(label)

    true_labels_df['True_Label'] = true_labels

    # Initialize a list to store accuracy results
    results = []

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Device: {device}')
    logging.info(f'Device selected for computation: {device}')

    # Iterate over each CLIP model
    for model_name in clip_models:
        print(f'\nProcessing inference with CLIP model: {model_name}')
        logging.info(f'Processing inference with CLIP model: {model_name}')

        try:
            # Load the CLIP model and preprocessing function
            model, preprocess = clip.load(model_name, device=device)
            model = model.visual.float()
            logging.info(f'Successfully loaded model: {model_name}')
        except Exception as e:
            logging.error(f'Error loading model {model_name}: {e}')
            print(f'Error loading model {model_name}: {e}')
            continue  # Skip to the next model

        # Load precomputed text features
        save_filename = f'clip_{model_name.replace("/", "-")}_text_features.npy'
        save_filepath = os.path.join(npy_load_path, save_filename)
        if not os.path.exists(save_filepath):
            logging.error(f'Text features file not found for model {model_name}: {save_filepath}')
            print(f'Text features file not found for model {model_name}: {save_filepath}')
            continue  # Skip to the next model

        try:
            text_features_np = np.load(save_filepath)  # Shape: (num_categories, feature_dim)
            text_features = torch.tensor(text_features_np).to(device).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            logging.info(f'Loaded and normalized text features from {save_filepath}')
        except Exception as e:
            logging.error(f'Error loading text features from {save_filepath}: {e}')
            print(f'Error loading text features from {save_filepath}: {e}')
            continue  # Skip to the next model

        # Initialize a list to store prediction probabilities
        pred_probs = []

        # Iterate over each image
        for idx, row in tqdm(true_labels_df.iterrows(), total=true_labels_df.shape[0],
                             desc=f'CLIP Inference Images with {model_name}', leave=False):
            image_path = row['path']
            try:
                image = preprocess(Image.open(image_path)).unsqueeze(0).to(device).float()
            except Exception as e:
                logging.error(f"Error opening image {image_path}: {e}")
                print(f"Error opening image {image_path}: {e}")
                # Append a zero vector if image fails to open
                pred_probs.append(np.zeros(len(categories)))
                continue

            try:
                with torch.no_grad():
                    image_features = model(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    similarity = (image_features @ text_features.T) * 100.0
                    probs = similarity.softmax(dim=-1).cpu().numpy()
                pred_probs.append(probs.flatten())
            except Exception as e:
                logging.error(f"Error encoding image {image_path} with model {model_name}: {e}")
                print(f"Error encoding image {image_path} with model {model_name}: {e}")
                pred_probs.append(np.zeros(len(categories)))
                continue

        # Convert prediction probabilities to a NumPy array
        pred_probs_array = np.array(pred_probs)  # Shape: (num_images, num_categories)
        print(f'Prediction probabilities shape for {model_name}: {pred_probs_array.shape}')
        logging.info(f'Prediction probabilities shape for {model_name}: {pred_probs_array.shape}')

        # Save the prediction probabilities to a .npy file
        # Define a unique save name for each model
        pred_probs_filename = f'clip_{model_name.replace("/", "-")}_pred_probs.npy'
        pred_probs_filepath = os.path.join(npy_save_path, pred_probs_filename)
        try:
            np.save(pred_probs_filepath, pred_probs_array)
            logging.info(f'Saved prediction probabilities to {pred_probs_filepath}')
            print(f'Saved prediction probabilities to {pred_probs_filepath}')
        except Exception as e:
            logging.error(f'Error saving prediction probabilities to {pred_probs_filepath}: {e}')
            print(f'Error saving prediction probabilities to {pred_probs_filepath}: {e}')
            continue  # Skip accuracy computation if saving failed

        # Determine predicted labels by taking the argmax
        predicted_indices = pred_probs_array.argmax(axis=1)
        predicted_labels = [categories[idx] if idx < len(categories) else "Unknown" for idx in predicted_indices]

        # Add predicted labels to the DataFrame
        true_labels_df[f'Predicted_{model_name.replace("/", "_")}'] = predicted_labels

        # Compute accuracy
        correct_predictions = sum(true_labels_df['True_Label'] == predicted_labels)
        total_predictions = len(predicted_labels)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        accuracy_percentage = accuracy * 100

        # Log and print accuracy
        logging.info(f'Accuracy for model {model_name}: {accuracy_percentage:.2f}%')
        print(f'Accuracy for model {model_name}: {accuracy_percentage:.2f}%')

        # Append the result to the results list
        results.append({'Model': model_name, 'Accuracy': accuracy_percentage})

        # Clear CUDA cache to free memory
        if device == "cuda":
            torch.cuda.empty_cache()

    # Create a DataFrame from the results list
    results_df = pd.DataFrame(results)

    # Save the accuracy results to a CSV file
    accuracy_csv_path = os.path.join(npy_save_path, 'clip_models_accuracy.csv')
    try:
        results_df.to_csv(accuracy_csv_path, index=False)
        logging.info(f'Saved accuracy results to {accuracy_csv_path}')
        print(f'\nAccuracy results saved to {accuracy_csv_path}')
    except Exception as e:
        logging.error(f'Error saving accuracy results to {accuracy_csv_path}: {e}')
        print(f'Error saving accuracy results to {accuracy_csv_path}: {e}')

    # Optionally, save the true_labels_df with all predicted labels
    detailed_results_csv = os.path.join(npy_save_path, 'detailed_predictions.csv')
    try:
        true_labels_df.to_csv(detailed_results_csv, index=False)
        logging.info(f'Saved detailed predictions to {detailed_results_csv}')
        print(f'Detailed predictions saved to {detailed_results_csv}')
    except Exception as e:
        logging.error(f'Error saving detailed predictions to {detailed_results_csv}: {e}')
        print(f'Error saving detailed predictions to {detailed_results_csv}: {e}')

    print("\nInference using precomputed text features completed successfully!")
    logging.info("Inference using precomputed text features completed successfully!")

if __name__ == "__main__":
    main()
