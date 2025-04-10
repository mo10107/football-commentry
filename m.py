import os
import cv2
import time
import argparse
import numpy as np
import supervision as sv
from ultralytics import YOLO

# Import configuration
import config

# Initialize YOLO Model
model = YOLO(config.DETECTION["model_path"])  # Load YOLOv8 Model
tracker = sv.ByteTrack()  # Initialize ByteTrack Tracker
box_annotator = sv.BoundingBoxAnnotator()  # Visualization Tool

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Football Detection and Tracking System")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default=config.VIDEO["output_path"], help="Path to output video file")
    parser.add_argument("--display", action="store_true", help="Display video with visualizations")
    return parser.parse_args()

def process_frame(frame: np.ndarray, _: int) -> np.ndarray:
    """
    Process a single frame for detection, tracking, and annotation.
    
    Args:
        frame: Input video frame (NumPy array)
        _: Frame index (ignored)
    
    Returns:
        Annotated frame
    """
    # Run YOLOv8 Detection
    results = model(frame)[0]

    # Convert detections to Supervision format
    detections = sv.Detections.from_ultralytics(results)

    # Update Tracker with Detections
    detections = tracker.update_with_detections(detections)

    # Annotate frame
    return box_annotator.annotate(frame.copy(), detections=detections)

def main():
    args = parse_args()

    # Update config with args
    config.VIDEO["input_source"] = args.video
    config.VIDEO["output_path"] = args.output
    config.VIDEO["display"] = args.display
    
    # Set up video capture
    cap = cv2.VideoCapture(config.VIDEO["input_source"])
    if not cap.isOpened():
        print(f"Error: Could not open video source {config.VIDEO['input_source']}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Initialize video writer
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process Frame with YOLO & Tracking
            annotated_frame = process_frame(frame, frame_count)
            
            # Write to output video
            if args.output:
                out.write(annotated_frame)
            
            # Display if requested
            if args.display:
                cv2.imshow("Football Detection and Tracking", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Print progress
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"Processing: {progress:.1f}% complete ({frame_count}/{total_frames})")
    
    except KeyboardInterrupt:
        print("Processing interrupted by user.")
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        # Clean up
        cap.release()
        if args.output:
            out.release()
        if args.display:
            cv2.destroyAllWindows()
        
        print("Processing complete!")

if __name__ == "__main__":
    main()
