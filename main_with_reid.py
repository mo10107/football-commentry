#python main_with_reid.py --video \data\football_video.mp4 --output processed_video.mp4  
# ==============================================================================
# main_with_reid.py - Updated for Tracklet Re-ID and New Model Option
# ==============================================================================

# ==============================================================================
# main_with_reid.py
#
# Main script for Football Player Detection, Tracking, and Re-Identification
# Features:
# - YOLO Detection
# - ByteTrack Tracking
# - Re-Identification using models from torchreid (configurable via config.py)
# - Tracklet-based feature aggregation for robust Re-ID matching
# - Peak score logic for stable identity assignment over time
# - Diagnostic printing of top-k matches
# - Configuration driven by config.py
# ==============================================================================

import os
import cv2
import time
import argparse # Ensure argparse is imported
import numpy as np
import supervision as sv # Use supervision >= 0.18.0 for BoxAnnotator
from ultralytics import YOLO
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# --- Re-ID Specific Imports ---
import torch
import torchreid # Ensure this and potentially dependencies like resnest are installed
import faiss
from torchvision import transforms
from PIL import Image
import pickle

# --- Import Configuration ---
try:
    import config
except ImportError:
    print("❌ Error: config.py not found. Please ensure it exists and is configured.")
    exit()
except KeyError as e:
    # Basic check if essential sections exist, more specific checks later
    if 'REID' not in dir(config) or 'DETECTION' not in dir(config) or 'VIDEO' not in dir(config):
         print(f"❌ Error: Missing essential section (DETECTION, REID, VIDEO) in config.py.")
         exit()
    print(f"⚠️ Warning: Potentially missing key {e} during initial config check.")


# --- Global Re-ID Variables ---
REID_MODEL = None
REID_DEVICE = None
FAISS_INDEX = None
PLAYER_NAMES = None
REID_TRANSFORM = None

# --- State Management ---
# Peak Score (long-term memory per track)
TRACK_BEST_MATCHES = {} # {tracker_id: {"name": str, "score": float}}
# Tracklet Feature Storage (short-term buffer for aggregation)
TRACKLET_FEATURES = defaultdict(list) # {tracker_id: [emb1, emb2, ...]}

# --- Initialize Models and Annotators ---
# Moved initialization inside main() or helper functions to ensure config is loaded
yolo_model = None
tracker = None
box_annotator = None
label_annotator = None


# --- Re-ID Helper Functions ---

def load_reid_resources():
    """Loads the Re-ID model, FAISS index, and player names based on config."""
    global REID_MODEL, REID_DEVICE, FAISS_INDEX, PLAYER_NAMES, REID_TRANSFORM

    # Check if ReID is enabled in config
    if not config.REID.get("enabled", False):
        print("ℹ️ Re-ID is disabled in config.")
        return False

    try:
        # --- Model Loading ---
        model_name = config.REID['model_name']
        print(f"⏳ Loading Re-ID model: {model_name}...")
        # Build model, pretrained=True loads weights. num_classes is placeholder.
        REID_MODEL = torchreid.models.build_model(
            name=model_name,
            num_classes=1000, # Placeholder, weights overwrite classifier
            pretrained=True
        )
        REID_MODEL.eval() # Set to evaluation mode
        REID_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        REID_MODEL = REID_MODEL.to(REID_DEVICE)
        print(f"✅ Re-ID model '{model_name}' loaded on {REID_DEVICE}")

        # --- Transform ---
        # Ensure input_size is a tuple (H, W)
        input_size_config = config.REID["input_size"]
        if not isinstance(input_size_config, (tuple, list)) or len(input_size_config) != 2:
             raise ValueError(f"config.REID['input_size'] must be a tuple or list of (Height, Width), got: {input_size_config}")
        h, w = input_size_config
        REID_TRANSFORM = transforms.Compose([
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        print(f"✅ Re-ID Transform configured for input size: ({h}, {w})")

        # --- FAISS Index ---
        index_path = config.REID["faiss_index_path"]
        print(f"⏳ Loading FAISS index from: {index_path}")
        if not os.path.exists(index_path):
             raise FileNotFoundError(f"FAISS index file not found: {index_path}")
        FAISS_INDEX = faiss.read_index(index_path)

        # --- CRITICAL Check: Ensure Index Dimension Matches Config ---
        expected_dim = config.REID["embedding_dim"]
        if FAISS_INDEX.d != expected_dim:
             raise ValueError(f"❌ FAISS index dimension ({FAISS_INDEX.d}) does not match config embedding_dim ({expected_dim}) "
                              f"for model '{model_name}'. Please recreate the index using 'create_reid_index.py' "
                              f"after ensuring config.py and the index script use the correct dimension.")
        print(f"✅ FAISS index loaded (Dim: {FAISS_INDEX.d}, Entries: {FAISS_INDEX.ntotal}).")

        # --- Player Names ---
        names_path = config.REID["player_names_path"]
        print(f"⏳ Loading player names from: {names_path}")
        if not os.path.exists(names_path):
             raise FileNotFoundError(f"Player names file not found: {names_path}")
        with open(names_path, "rb") as f:
            PLAYER_NAMES = pickle.load(f)
        print(f"✅ Player names loaded ({len(PLAYER_NAMES)} names).")

        # --- Sanity Check: Index Size vs Names Count ---
        if FAISS_INDEX.ntotal != len(PLAYER_NAMES):
            print(f"⚠️ Warning: FAISS index size ({FAISS_INDEX.ntotal}) does not match number of player names ({len(PLAYER_NAMES)}). "
                  "This can lead to errors or incorrect matches. Ensure the index and names files correspond.")
            # Depending on severity, you might want to disable Re-ID or raise an error here
            # config.REID["enabled"] = False
            # return False

        return True # Indicate successful loading

    except FileNotFoundError as e:
        print(f"❌ Error loading Re-ID resources: {e}")
        print("ℹ️ Re-ID will be disabled.")
        config.REID["enabled"] = False # Disable Re-ID if essential files are missing
        return False
    except KeyError as e:
         print(f"❌ Error: Missing required key '{e}' in config.REID dictionary.")
         print("ℹ️ Re-ID will be disabled.")
         config.REID["enabled"] = False
         return False
    except ValueError as e: # Catch specific errors like dimension mismatch
         print(f"❌ Error loading Re-ID resources: {e}")
         print("ℹ️ Re-ID will be disabled.")
         config.REID["enabled"] = False
         return False
    except Exception as e: # Catch-all for other unexpected errors
        print(f"❌ An unexpected error occurred during Re-ID resource loading: {e}")
        import traceback
        traceback.print_exc()
        config.REID["enabled"] = False # Disable Re-ID on other errors
        return False

def preprocess_for_reid(crop: np.ndarray) -> Optional[torch.Tensor]:
    """Preprocesses a NumPy crop (BGR) for the Re-ID model."""
    global REID_TRANSFORM
    if REID_TRANSFORM is None:
        # This check prevents errors if ReID failed to load but somehow process_frame is called
        # print("⚠️ ReID Transform not initialized in preprocess_for_reid.") # Reduce noise
        return None
    try:
        # Convert NumPy array (BGR assumed from OpenCV) to PIL Image (RGB)
        pil_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        # Apply the transform and add batch dimension
        tensor = REID_TRANSFORM(pil_image).unsqueeze(0)
        return tensor
    except Exception as e:
        # Log error less frequently to avoid console spam
        # if np.random.rand() < 0.01: print(f"⚠️ Error during ReID preprocessing: {e}")
        return None

@torch.no_grad()
def extract_feature_vector(processed_tensor: torch.Tensor) -> Optional[np.ndarray]:
    """Extracts feature vector using the loaded Re-ID model."""
    global REID_MODEL, REID_DEVICE
    if REID_MODEL is None or processed_tensor is None:
        return None
    try:
        processed_tensor = processed_tensor.to(REID_DEVICE)
        embedding = REID_MODEL(processed_tensor) # Get feature vector from model
        feature = embedding.cpu().numpy().flatten() # Convert to NumPy array

        # Handle potential NaN or Inf values, replace with 0
        # Check only occasionally to reduce performance impact unless it's frequent
        if np.isnan(feature).any() or np.isinf(feature).any():
            # print("⚠️ Warning: NaN/Inf detected in extracted feature. Replacing with zeros.") # Reduce noise
            feature = np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Optional: Dimension check ---
        # expected_dim = config.REID["embedding_dim"]
        # if feature.shape[0] != expected_dim:
        #     print(f"⚠️ Extracted feature dim {feature.shape[0]} != expected {expected_dim}")
        #     return None

        return feature
    except Exception as e:
        # if np.random.rand() < 0.01: print(f"⚠️ Error during ReID inference: {e}") # Reduce noise
        return None

def find_top_k_matches(query_embedding: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
    """Compares query embedding against the FAISS index and returns top K matches."""
    global FAISS_INDEX, PLAYER_NAMES
    matches = []
    # Ensure resources are loaded and query is valid
    if FAISS_INDEX is None or query_embedding is None or not PLAYER_NAMES:
        return matches
    try:
        # Ensure correct dtype and shape for FAISS
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)

        # Normalize the query vector (L2 norm) - crucial for cosine similarity with IndexFlatIP
        faiss.normalize_L2(query_embedding)

        # Search the index for the top K nearest neighbors
        scores, indices = FAISS_INDEX.search(query_embedding, k)

        # Process the results
        for i in range(min(k, len(indices[0]))): # Handle cases where k > index size
            score = scores[0][i]
            idx = indices[0][i]

            # Clamp score: Cosine similarity should ideally be <= 1.0
            # Floating point errors might make it slightly higher.
            score = min(score, 1.0)

            # Check if the index is valid before accessing PLAYER_NAMES
            if 0 <= idx < len(PLAYER_NAMES):
                player_name = PLAYER_NAMES[idx]
                matches.append((player_name, score))
            # else: FAISS returned an invalid index (should not happen with IndexFlatIP)

        return matches # List of (name, score) tuples

    except Exception as e:
        # if np.random.rand() < 0.01: print(f"⚠️ Error during FAISS top-k search: {e}") # Reduce noise
        return matches # Return empty list on error

# --- Argument Parsing Function ---
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Football Detection, Tracking, and Re-ID System")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    # Use default from config if available and key exists, otherwise use hardcoded default
    default_output = "output.mp4"
    if 'VIDEO' in dir(config) and isinstance(config.VIDEO, dict):
        default_output = config.VIDEO.get("output_path", default_output)
    parser.add_argument("--output", type=str, default=default_output,
                        help="Path to output video file")
    parser.add_argument("--display", action="store_true",
                        help="Display video with visualizations (overrides config)")
    parser.add_argument("--no-display", action="store_true",
                        help="Force disable display (overrides config)")
    parser.add_argument("--no-reid", action="store_true", help="Disable Re-ID even if enabled in config")
    parser.add_argument("--reid-thresh", type=float, default=None, help="Override ReID similarity threshold from config")

    # Parse arguments
    args = parser.parse_args()

    # Handle display logic based on args and config (ensure config.VIDEO exists)
    if 'VIDEO' in dir(config) and isinstance(config.VIDEO, dict):
        if args.display:
            config.VIDEO["display"] = True
        elif args.no_display:
            config.VIDEO["display"] = False
        # If neither arg is provided, config.VIDEO["display"] remains as defined in config.py
    elif args.display and not args.no_display:
         print("⚠️ Warning: --display used, but config.VIDEO dictionary not found. Display may not work as expected.")
         # Attempt to set a temporary display flag if needed by later logic
         # config.VIDEO = {"display": True} # Or handle appropriately

    return args


# --- Main Frame Processing Function ---
def process_frame(frame: np.ndarray, frame_idx: int) -> np.ndarray:
    """ Process a single frame: Detect -> Track -> Re-ID (Tracklet + Peak Score) -> Annotate """
    global TRACK_BEST_MATCHES, TRACKLET_FEATURES, yolo_model, tracker # Allow access

    # 1. YOLO Detection
    # Check if yolo_model is initialized
    if yolo_model is None:
         print("❌ Error in process_frame: YOLO model not initialized.")
         return frame # Return original frame if detection fails
    results = yolo_model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Filter Detections by Class ID specified in config
    target_class_id = config.REID.get("target_class_id") if config.REID else None
    if target_class_id is not None:
        detections = detections[detections.class_id == target_class_id]

    # 2. Update Tracker
    # Check if tracker is initialized
    if tracker is None:
         print("❌ Error in process_frame: Tracker not initialized.")
         # Cannot proceed without tracking if ReID is enabled
         return frame # Or handle error differently
    detections = tracker.update_with_detections(detections)

    # Prepare lists for labels and debug messages
    labels = []
    debug_prints = []

    # --- Re-ID Logic ---
    if config.REID.get("enabled", False) and len(detections) > 0:
        # Get IDs present in the current frame
        current_track_ids = set(detections.tracker_id)

        # --- Process each tracked detection ---
        for i in range(len(detections)):
            xyxy = detections.xyxy[i]
            tracker_id = detections.tracker_id[i]
            # Default label if no peak match is found
            current_label = f"ID:{tracker_id} ?"

            # --- Feature Extraction for Current Frame ---
            x1, y1, x2, y2 = map(int, xyxy)
            h_frame, w_frame, _ = frame.shape
            # Ensure coordinates are valid and within frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_frame, x2), min(h_frame, y2)

            if x1 < x2 and y1 < y2: # Proceed only if crop dimensions are valid
                crop = frame[y1:y2, x1:x2]
                # Preprocess crop for Re-ID model
                processed_tensor = preprocess_for_reid(crop)
                # Extract feature vector (embedding)
                current_embedding = extract_feature_vector(processed_tensor)

                if current_embedding is not None:
                    # --- Tracklet Logic ---
                    if config.REID.get("use_tracklet", False):
                        # Append current embedding to this track's buffer
                        TRACKLET_FEATURES[tracker_id].append(current_embedding)
                        tracklet_list = TRACKLET_FEATURES[tracker_id]
                        tracklet_current_size = len(tracklet_list)
                        tracklet_target_size = config.REID.get("tracklet_size", 16) # Default if missing

                        # Check if the tracklet buffer is full enough
                        if tracklet_current_size >= tracklet_target_size:
                            # --- Aggregate features in the tracklet ---
                            aggregation_method = config.REID.get("tracklet_aggregation", "mean")
                            aggregated_embedding = None
                            try:
                                if len(tracklet_list) > 0: # Ensure list is not empty
                                    if aggregation_method == "mean":
                                        aggregated_embedding = np.mean(tracklet_list, axis=0)
                                    elif aggregation_method == "max":
                                        aggregated_embedding = np.max(tracklet_list, axis=0)
                                    else: # Default to mean for invalid method
                                        aggregated_embedding = np.mean(tracklet_list, axis=0)
                                        # Log warning occasionally
                                        # if frame_idx % 100 == 1: print(f"⚠️ Invalid tracklet agg: {aggregation_method}. Using 'mean'.")
                                else:
                                     # if frame_idx % 100 == 1: print(f"⚠️ Tracklet list empty for ID {tracker_id} at agg step.")
                                     pass # Cannot aggregate empty list

                            except Exception as agg_e:
                                 # if frame_idx % 100 == 1: print(f"⚠️ Error during tracklet agg for ID {tracker_id}: {agg_e}")
                                 pass # Avoid crashing, proceed without aggregation result

                            if aggregated_embedding is not None:
                                # --- Match the aggregated tracklet embedding ---
                                top_matches = find_top_k_matches(aggregated_embedding, k=3)

                                if top_matches:
                                    tracklet_best_name, tracklet_best_score = top_matches[0]
                                    # --- Update Peak Score based on Tracklet Result ---
                                    similarity_threshold = config.REID.get("similarity_threshold", 0.65)
                                    if tracklet_best_score >= similarity_threshold:
                                        existing_peak = TRACK_BEST_MATCHES.get(tracker_id)
                                        # Update if no peak exists OR tracklet score is higher
                                        if existing_peak is None or tracklet_best_score > existing_peak["score"]:
                                            TRACK_BEST_MATCHES[tracker_id] = {"name": tracklet_best_name, "score": tracklet_best_score}
                                            # Print when a new peak is set via tracklet
                                            print(f"+++ F:{frame_idx} TRACKLET PEAK ID {tracker_id} -> {tracklet_best_name} ({tracklet_best_score:.3f}) +++")

                                    # --- Collect Diagnostic Info for Printing ---
                                    match_str = ", ".join([f"{name}({score:.2f})" for name, score in top_matches])
                                    debug_prints.append(f"  [F:{frame_idx}, ID:{tracker_id}, Trklt] Matches:[{match_str}]")
                                # else: No matches found for aggregated embedding

                            # --- Clear the buffer for the next tracklet ---
                            TRACKLET_FEATURES[tracker_id] = [] # Reset buffer after processing

                    # --- End Tracklet Logic ---

            # --- Generate Label based on TRACK_BEST_MATCHES history ---
            # Check if a peak score has ever been recorded for this track ID
            best_match_info = TRACK_BEST_MATCHES.get(tracker_id)
            if best_match_info:
                # Use the stored peak information for the label
                player_name = best_match_info["name"]
                peak_score = best_match_info["score"]
                current_label = f"ID:{tracker_id} {player_name} (Peak:{peak_score:.2f})"
            # else: label remains the default "ID:XXX ?"

            labels.append(current_label)

        # --- Clean up state for tracks that disappeared ---
        lost_track_ids = (set(TRACK_BEST_MATCHES.keys()) | set(TRACKLET_FEATURES.keys())) - current_track_ids
        for tid in lost_track_ids:
            # Clear tracklet buffer immediately
            if tid in TRACKLET_FEATURES: del TRACKLET_FEATURES[tid]
            # Optionally clear peak score history after a delay or keep it
            # if tid in TRACK_BEST_MATCHES: del TRACK_BEST_MATCHES[tid]

    else: # Re-ID disabled or no (filtered) detections in this frame
        labels = [f"ID:{tid}" for tid in detections.tracker_id] if len(detections) > 0 else []


    # --- Print collected debug info periodically ---
    # Adjust frequency (e.g., % 10, % 50) or condition (e.g., if len(debug_prints) > 0)
    if frame_idx % 30 == 1 and debug_prints: # Print roughly every second at 30fps
        print(f"--- Debug Info Frame {frame_idx} ---")
        for msg in debug_prints:
            print(msg)
        print(f"-----------------------------")


    # --- Annotation ---
    annotated_frame = frame.copy() # Work on a copy
    # Check if annotators are initialized
    if box_annotator is None or label_annotator is None:
         print("❌ Error in process_frame: Annotators not initialized.")
         return frame # Return unannotated frame

    # Annotate boxes
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame,
        detections=detections
    )
    # Annotate labels, ensuring list lengths match
    if len(labels) == len(detections):
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )
    elif len(detections) > 0: # Mismatch occurred but there are detections
         # Fallback: Annotate with just tracker IDs if label list is wrong length
         fallback_labels = [f"ID:{tid}" for tid in detections.tracker_id]
         annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=fallback_labels
         )
    # If no detections, no labels are annotated anyway

    return annotated_frame

# --- Main Execution Function ---
def main():
    global TRACK_BEST_MATCHES, TRACKLET_FEATURES # Allow modification
    global yolo_model, tracker, box_annotator, label_annotator # Allow initialization

    # Clear state from previous runs if script is re-run in same session (e.g., in Jupyter)
    TRACK_BEST_MATCHES = {}
    TRACKLET_FEATURES.clear()

    # --- Argument Parsing and Config Update ---
    args = parse_args() # Parse arguments and update config display settings

    # Apply command-line overrides to config
    if args.no_reid:
        if config.REID and config.REID.get("enabled", False): # Check if it was enabled
            print("ℹ️ Re-ID explicitly disabled via --no-reid argument.")
            config.REID["enabled"] = False
    if args.reid_thresh is not None:
         if config.REID and config.REID.get("enabled", False): # Only override if ReID is enabled
             print(f"ℹ️ Overriding ReID similarity threshold from config with: {args.reid_thresh}")
             config.REID["similarity_threshold"] = args.reid_thresh
         elif args.reid_thresh is not None: # User provided arg but ReID is off
              print("⚠️ Warning: --reid-thresh provided, but Re-ID is disabled. Threshold not changed.")

    # --- Initialize Models and Annotators ---
    print("⏳ Initializing models and annotators...")
    try:
        # Ensure DETECTION config exists and has the path key
        if not ('DETECTION' in dir(config) and isinstance(config.DETECTION, dict) and 'model_path' in config.DETECTION):
             raise KeyError("config.DETECTION['model_path'] is missing or invalid.")
        yolo_model = YOLO(config.DETECTION["model_path"])
        print(f"✅ YOLO model loaded: {config.DETECTION['model_path']}")

        # Initialize tracker (consider parameters from config if needed)
        tracker = sv.ByteTrack() # Default settings

        # Initialize annotators (using updated Supervision names)
        box_annotator = sv.BoxAnnotator(thickness=1)
        label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER, text_scale=0.4, text_thickness=1, text_padding=2)
        print("✅ Tracker and Annotators initialized.")

    except KeyError as e:
        print(f"❌ Fatal Error: Missing configuration key for model initialization: {e}")
        return
    except Exception as e:
        print(f"❌ Fatal Error: Could not initialize models/annotators: {e}")
        import traceback
        traceback.print_exc()
        return # Exit if essential components fail

    # --- Load Re-ID Resources (if enabled in config) ---
    if config.REID and config.REID.get("enabled", False):
        load_reid_resources() # This function handles internal disabling if loading fails

    # --- Video Input Setup ---
    video_path = args.video
    print(f"⏳ Opening video source: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Fatal Error: Could not open video source: {video_path}")
        return
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    try: # Frame count might not be available
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: total_frames = -1 # Indicate unknown length
    except:
        total_frames = -1
        print("⚠️ Warning: Could not determine total frame count from video source.")
    print(f"🎬 Video Info: {width}x{height} @ {fps:.2f} FPS, {total_frames if total_frames > 0 else 'Unknown'} frames")

    # --- Video Output Setup ---
    writer = None
    output_path = args.output
    if output_path:
        print(f"⏳ Setting up video writer for: {output_path}")
        # Ensure output directory exists (optional, prevents errors if dir is missing)
        # os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 (common)
        output_fps = fps if fps > 0 else 30.0 # Use source FPS or default to 30
        try:
            writer = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
            if not writer.isOpened():
                print(f"❌ Error: Failed to open video writer for {output_path}. Output disabled.")
                writer = None
            else:
                 print(f"✅ Video writer configured for {output_path}")
        except Exception as e:
             print(f"❌ Error initializing video writer: {e}. Output disabled.")
             writer = None


    # --- Processing Loop ---
    frame_count = 0
    start_time = time.time()
    print("\n🚀 Starting video processing loop...")
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            # If frame reading fails (end of video or error), break the loop
            if not ret:
                if frame_count > 0: # Check if we processed any frames before ending
                     print("\n🏁 End of video stream reached.")
                else:
                     print("\n❌ Error: Could not read first frame from video.")
                break

            frame_count += 1
            loop_start_time = time.time()

            # --- Process the frame ---
            annotated_frame = process_frame(frame, frame_count)

            # --- Write frame to output video file ---
            if writer:
                try:
                    writer.write(annotated_frame)
                except Exception as write_e:
                    print(f"❌ Error writing frame {frame_count} to video: {write_e}")
                    # Optionally disable writer after first error: writer = None
                    break # Stop processing if writing fails consistently

            # --- Display the frame ---
            display_enabled = False # Default
            if 'VIDEO' in dir(config) and isinstance(config.VIDEO, dict):
                 display_enabled = config.VIDEO.get("display", False)

            if display_enabled:
                cv2.imshow("Football Tracking & Re-ID (Tracklet + Peak)", annotated_frame)
                # Check for 'q' key press to quit (waitKey(1) allows frame processing)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n🛑 User requested quit via display window.")
                    break

            # --- Print progress periodically ---
            # Only calculate/print if loop isn't instantaneous
            loop_time = time.time() - loop_start_time
            if frame_count % 50 == 0 and loop_time > 1e-6: # Adjust frequency as needed
                processing_fps = 1.0 / loop_time
                elapsed_time = time.time() - start_time
                avg_fps = frame_count / elapsed_time if elapsed_time > 0 else float('inf')
                progress_str = f"Frame {frame_count}"
                if total_frames > 0:
                    progress = (frame_count / total_frames) * 100
                    progress_str = f"{progress:.1f}% ({frame_count}/{total_frames})"
                print(f"⏱️  Processed: {progress_str}, Current FPS: {processing_fps:.1f}, Avg FPS: {avg_fps:.1f}")

    except KeyboardInterrupt:
        print("\n🛑 Processing interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during the processing loop:")
        import traceback
        traceback.print_exc()
    finally:
        # --- Cleanup ---
        print("\n🧹 Cleaning up resources...")
        if cap and cap.isOpened():
            cap.release()
            print("- Video capture released.")
        if writer and writer.isOpened():
            print(f"- Releasing video writer for {output_path}...")
            writer.release()
            print("- Video writer released.")
        # Check display flag again in case it was set temporarily
        display_enabled = False
        if 'VIDEO' in dir(config) and isinstance(config.VIDEO, dict):
             display_enabled = config.VIDEO.get("display", False)
        if display_enabled:
            cv2.destroyAllWindows()
            print("- Display windows closed.")

        # --- Final Summary ---
        end_time = time.time()
        total_time = end_time - start_time

        # Print summary of final identifications stored
        print("\n--- Final Identifications (Peak Scores) ---")
        identified_count = 0
        if TRACK_BEST_MATCHES:
            # Sort by Track ID for cleaner output
            for tid in sorted(TRACK_BEST_MATCHES.keys()):
                data = TRACK_BEST_MATCHES[tid]
                print(f"  Track ID {tid}: {data['name']} (Peak Score: {data['score']:.3f})")
                identified_count +=1
            print(f"Total tracks with an identification recorded: {identified_count}")
        else:
            print("  No identifications recorded.")
        print("-------------------------------------------")

        # Print overall processing summary
        print("\n--- Processing Summary ---")
        print(f"Total frames processed: {frame_count}")
        print(f"Total time taken: {total_time:.2f} seconds")
        if frame_count > 0 and total_time > 0:
            overall_avg_fps = frame_count / total_time
            print(f"Overall Average FPS: {overall_avg_fps:.2f}")
        # Check if output file exists as a confirmation of writing success
        if output_path and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
             print(f"💾 Output video likely saved successfully to: {os.path.abspath(output_path)}")
        elif output_path:
             print(f"⚠️ Output video file may be empty or not saved correctly: {output_path}")
        print("✅ Processing complete!")

# --- Script Execution Entry Point ---
if __name__ == "__main__":
    main()
