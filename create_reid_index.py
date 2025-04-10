#python create_reid_index.py --dataset D:\AI_league_compeation\code\drive-download-20250409T013908Z-001\database\players_database --index my_reid_index.bin --names my_player_names.pkl
# Example command (update paths based on your config.py):
# python create_reid_index.py --dataset /path/to/database --index path/to/faiss_index_resnest50_2048d.bin --names path/to/player_names.pkl

import torch
import torchreid # Ensure this and dependencies like resnest are installed
import faiss
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import pickle
import argparse
from tqdm import tqdm

# --- Import Configuration ---
try:
    import config # Try to import the main config file
except ImportError:
    print("❌ Error: config.py not found. Cannot determine ReID settings.")
    print("Please ensure config.py exists in the same directory or Python path.")
    exit()
except KeyError as e:
    print(f"❌ Error: Missing key {e} in config.py. Please ensure REID section is complete.")
    exit()

# --- Read Settings from Config ---
try:
    REID_MODEL_NAME = config.REID['model_name']
    REID_INPUT_SIZE = config.REID['input_size'] # Expects tuple (H, W)
    EXPECTED_EMBEDDING_DIM = config.REID['embedding_dim'] # CRITICAL for FAISS index dimension
    # Get default paths from config for argparse defaults
    DEFAULT_INDEX_PATH = config.REID['faiss_index_path']
    DEFAULT_NAMES_PATH = config.REID['player_names_path']
    # Dataset path is usually better as an argument, but could be read from config too
    DEFAULT_DATASET_PATH = "players_database" # Or read from config if preferred
except KeyError as e:
    print(f"❌ Error: Required key '{e}' not found in config.REID dictionary.")
    print("Please ensure 'model_name', 'input_size', 'embedding_dim', 'faiss_index_path', and 'player_names_path' are defined.")
    exit()


print(f"--- Index Creation Configuration ---")
print(f"Model Name:         {REID_MODEL_NAME}")
print(f"Input Size (H, W):  {REID_INPUT_SIZE}")
print(f"Embedding Dim:      {EXPECTED_EMBEDDING_DIM}")
print(f"Default Index Path: {DEFAULT_INDEX_PATH}")
print(f"Default Names Path: {DEFAULT_NAMES_PATH}")
print(f"---------------------------------")


# --- Image Transform ---
# Uses REID_INPUT_SIZE read from config
try:
    transform = transforms.Compose([
        transforms.Resize(REID_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
except Exception as e:
    print(f"❌ Error creating image transform (check REID_INPUT_SIZE format in config): {e}")
    exit()

# --- Helper Functions ---

def load_reid_model():
    """Loads the torchreid model specified in config."""
    print(f"⏳ Loading Re-ID model: {REID_MODEL_NAME}...")
    try:
        model = torchreid.models.build_model(
            name=REID_MODEL_NAME,
            num_classes=1000, # Placeholder
            loss='softmax',   # Placeholder
            pretrained=True
        )
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"✅ Loaded Re-ID model '{REID_MODEL_NAME}' on {device}")
        return model, device
    except Exception as e:
        print(f"❌ Error loading Re-ID model '{REID_MODEL_NAME}': {e}")
        print("Make sure the model name is correct and dependencies (like 'resnest') are installed.")
        exit()


@torch.no_grad()
def extract_feature(model, image_path, device):
    """Extracts feature vector from a single image file."""
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        feature = model(image_tensor)
        feature = feature.cpu().numpy().flatten()
        if np.isnan(feature).any() or np.isinf(feature).any():
            feature = np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
        # --- Dimension Check ---
        if feature.shape[0] != EXPECTED_EMBEDDING_DIM:
             print(f"⚠️ Warning: Extracted feature dim ({feature.shape[0]}) for {os.path.basename(image_path)} "
                   f"does not match expected dim ({EXPECTED_EMBEDDING_DIM}) from config. Skipping image.")
             return None # Skip if dimension is wrong
        return feature
    except Exception as e:
        print(f"❌ Error processing image {image_path}: {e}")
        return None

def extract_all_features(model, dataset_path, device):
    """Extracts and averages features for each player in the dataset."""
    player_embeddings = {}
    print(f"⏳ Extracting features from dataset: {dataset_path}")
    if not os.path.isdir(dataset_path): print(f"❌ Error: Dataset path '{dataset_path}' not found."); return None
    player_folders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    if not player_folders: print(f"❌ Error: No player subdirectories found in '{dataset_path}'."); return None
    print(f"Found {len(player_folders)} potential player folders.")

    successful_players = 0
    for player in tqdm(player_folders, desc="Processing Players"):
        player_folder = os.path.join(dataset_path, player)
        features = []
        image_files = [f for f in os.listdir(player_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not image_files: continue # Skip if no images

        for img_name in image_files:
            img_path = os.path.join(player_folder, img_name)
            feature = extract_feature(model, img_path, device)
            if feature is not None: features.append(feature) # Only add if extraction succeeded and dim matched

        if features:
            # Check if all extracted features have the correct dimension before averaging
            valid_features = [f for f in features if f.shape[0] == EXPECTED_EMBEDDING_DIM]
            if valid_features:
                if len(valid_features) > 1: avg_feature = np.mean(valid_features, axis=0)
                else: avg_feature = valid_features[0]
                player_embeddings[player] = avg_feature
                successful_players += 1
            else:
                 print(f"⚠️ Warning: No features with correct dimension ({EXPECTED_EMBEDDING_DIM}) found for player '{player}'. Skipping.")
        # else: No valid features extracted for this player (already handled by extract_feature prints)

    if not player_embeddings: print("❌ Error: No features were extracted successfully."); return None
    print(f"\n✅ Extracted average features for {successful_players} players.")
    return player_embeddings

def create_faiss_index(player_embeddings):
    """Creates a FAISS index using the dimension specified in config."""
    if not player_embeddings: print("❌ Cannot create FAISS index: No embeddings."); return None, None
    player_names = list(player_embeddings.keys())
    features = np.array(list(player_embeddings.values()), dtype=np.float32)
    if features.size == 0: print("❌ Cannot create FAISS index: Feature array empty."); return None, None

    # --- Final Dimension Verification ---
    actual_dim = features.shape[1]
    if actual_dim != EXPECTED_EMBEDDING_DIM:
        print(f"❌❌ CRITICAL ERROR: Dimension of features going into FAISS ({actual_dim}) "
              f"does not match expected dimension from config ({EXPECTED_EMBEDDING_DIM}).")
        print("This likely means feature extraction yielded inconsistent dimensions. Index creation aborted.")
        return None, None

    print(f"Normalizing {features.shape[0]} feature vectors (Dim: {actual_dim})...")
    faiss.normalize_L2(features) # Normalize for Cosine Similarity search

    print(f"Building FAISS index (IndexFlatIP, Dim: {actual_dim})...")
    index = faiss.IndexFlatIP(actual_dim) # Use the verified dimension
    index.add(features)

    print(f"✅ Created FAISS index with {index.ntotal} players.")
    return index, player_names

def save_faiss_index(index, player_names, index_path, names_path):
    """Saves the FAISS index and player names list."""
    if index is None or player_names is None: print("❌ Cannot save: Index or names missing."); return
    print(f"💾 Saving FAISS index to: {index_path}")
    faiss.write_index(index, index_path)
    print(f"💾 Saving player names to: {names_path}")
    with open(names_path, "wb") as f: pickle.dump(player_names, f)
    print(f"✅ FAISS Index & Names Saved!")

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Create Re-ID Feature Index using FAISS")
    # Use defaults read from config file
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH,
                        help=f"Path to the player image dataset root directory")
    parser.add_argument("--index", type=str, default=DEFAULT_INDEX_PATH,
                        help=f"Output path for the FAISS index file")
    parser.add_argument("--names", type=str, default=DEFAULT_NAMES_PATH,
                        help=f"Output path for the player names pickle file")
    args = parser.parse_args()

    # Step 1: Load Model (Uses config internally)
    model, device = load_reid_model()

    # Step 2: Extract Features (Uses config internally for dim check)
    player_embeddings = extract_all_features(model, args.dataset, device)
    if player_embeddings is None: print("❌ Feature extraction failed. Exiting."); return

    # Step 3: Create FAISS Index (Uses config internally for dim check)
    faiss_index, player_names = create_faiss_index(player_embeddings)
    if faiss_index is None: print("❌ FAISS index creation failed. Exiting."); return

    # Step 4: Save the Index and Names (Uses paths from args, which default to config)
    save_faiss_index(faiss_index, player_names, args.index, args.names)

if __name__ == "__main__":
    main()