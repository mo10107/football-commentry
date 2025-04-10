"""
Configuration settings for Arabic Football Commentary System
"""

import os
from pathlib import Path
# Base paths
BASE_DIR = Path().resolve()
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Detection model settings
DETECTION = {
    "model_path": os.path.join(MODELS_DIR, "yolov8l-640-football-players.pt"),
    "conf_threshold": 0.1,
    "classes": {0: "Ball", 1: "Player"},
    "device": "cpu" # Options: "cpu", "cuda"
}

# Tracking settings
TRACKING = {
    "tracker_type": "ByteTrack",  # Options: "ByteTrack", "BoTSORT"
    "track_threshold": 0.6,
    "track_buffer": 30,
    "match_threshold": 0.8,
    "frame_rate": 30
}

# Re-identification settings
# In config.py:

REID = {
    "enabled": True,
    # --- Point 1: Use a different model ---
    "model_name": "resnet50",  # <<< CHANGED (Example: ResNeSt50)
    "input_size": (256, 128),   # <<< CHECK if resnest50 needs different (e.g., 224, 224), adjust if needed
    "embedding_dim": 2048,      # <<< CHANGED (ResNeSt50 typically outputs 2048) - CRITICAL for FAISS
    # --- Point 3: Tracklet configuration ---
    "use_tracklet": True,      # <<< ADDED: Enable tracklet logic
    "tracklet_size": 16,        # <<< ADDED: Number of features per tracklet (tune this)
    "tracklet_aggregation": "mean", # <<< ADDED: Aggregation method ('mean' or 'max')

    "faiss_index_path": os.path.join(DATA_DIR, "my_reid_index.bin"), # <<< CHANGED Filename (RECOMMENDED)
    "player_names_path": os.path.join(DATA_DIR, "my_player_names.pkl"), # <<< KEEP same name or change if needed

    "similarity_threshold": 0.70, # <<< TUNE this threshold for the new model/logic
    "use_averaging": False,      # <<< DISABLED (Optional: Tracklet aggregation replaces query averaging)
    "avg_window_size": 5,       # (Not used if use_averaging is False)
    "target_class_id": 2
}
# Event recognition settings
EVENT = {
    "model_path": os.path.join(MODELS_DIR, "timesformer_football.pth"),
    "frame_buffer_size": 16,
    "event_confidence_threshold": 0.7,
    "event_classes": {
        0: "goal",
        1: "pass",
        2: "shot",
        3: "save",
        4: "tackle",
        5: "foul",
        6: "corner",
        7: "free_kick",
        8: "offside"
    }
}

# Commentary generation settings
COMMENTARY = {
    "api_endpoint": "https://api.example.com/commentary",
    "api_key": os.environ.get("COMMENTARY_API_KEY", ""),
    "max_length": 100,
    "language": "arabic",
    "templates_file": os.path.join(DATA_DIR, "commentary_templates.json")
}

# Text-to-Speech settings
TTS = {
    "model_path": os.path.join(MODELS_DIR, "arabic_tts_model.pth"),
    "output_dir": os.path.join(OUTPUT_DIR, "audio"),
    "rate": 1.0,
    "pitch_factor": 1.0
}

# Knowledge base settings
KNOWLEDGE = {
    "players_db": os.path.join(DATA_DIR, "players.json"),
    "teams_db": os.path.join(DATA_DIR, "teams.json"),
    "phrases_db": os.path.join(DATA_DIR, "arabic_phrases.json")
}

# Video settings
VIDEO = {
    "input_source": os.path.join(DATA_DIR, "video.mp4"), 
    "output_path": os.path.join(OUTPUT_DIR, "output_video.mp4"),
    "display": True,
    "fps": 30,
    "resolution": (1280, 720)
}