"""
YOLOv12m model wrapper for football object detection
"""

import torch
from ultralytics import YOLO

class YOLOModel:
    def __init__(self, model_path, device="cuda"):
        """
        Initialize YOLOv12m model
        
        Args:
            model_path: Path to model weights
            device: Device to run model on (cuda/cpu)
        """
        self.device = device
        
        # Check if CUDA is available when device is set to cuda
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = "cpu"
            
        try:
            # Load YOLOv12m model
            self.model = YOLO(model_path)
            self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Model info
            print(f"YOLOv12m model loaded: {model_path}")
            print(f"Running on: {self.device}")
            print(f"Input size: {self.model.stride}")
            
        except Exception as e:
            print(f"Error loading YOLOv12m model: {e}")
            raise
    
    def predict(self, frame, conf=None):
        """
        Run inference on a frame
        
        Args:
            frame: Input video frame
            conf: Optional confidence threshold override
            
        Returns:
            list: List of detections
        """
        # Set confidence threshold
        if conf is None:
            conf = 0.25  # Default confidence threshold
        
        # Run inference
        with torch.no_grad():
            results = self.model(frame, conf=conf)
        
        # Return the results
        return results