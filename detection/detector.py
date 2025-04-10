"""
Football object detector using YOLOv12m model
"""

import cv2
import torch
import numpy as np
from .yolo_model import YOLOModel

class FootballDetector:
    def __init__(self, model_path, conf_threshold=0.5, device="cuda"):
        """
        Initialize FootballDetector with YOLOv12m model trained on ball and player classes
        
        Args:
            model_path: Path to YOLOv12m model weights
            conf_threshold: Confidence threshold for detections
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.conf_threshold = conf_threshold
        self.device = device
        self.model = YOLOModel(model_path, device)
        
        # Classes from config: {0: "Ball", 1: "Player"}
        self.classes = {0: "Ball", 1: "Player"}
        
        print(f"Football detector initialized with model: {model_path}")
        
    def detect(self, frame):
        """
        Detect football objects (players and ball) in a frame
        
        Args:
            frame: Input video frame
            
        Returns:
            dict: Dictionary with 'players' and 'ball' detections
        """
        # Prepare results container
        results = {
            'players': [],
            'ball': None
        }
        
        # Run inference
        detections = self.model.predict(frame)
        
        # Process detections
        if detections:
            for detection in detections:
                # Extract class, confidence, and bounding box
                class_id = int(detection.cls[0].item())
                confidence = detection.conf[0].item()
                x1, y1, x2, y2 = [int(coord) for coord in detection.xyxy[0]]
                
                # Skip if below confidence threshold
                if confidence < self.conf_threshold:
                    continue
                
                # Process based on class
                if class_id == 1:  # Player
                    results['players'].append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class': 'player'
                    })
                elif class_id == 0:  # Ball
                    # Only update ball if confidence is higher than current
                    if results['ball'] is None or confidence > results['ball']['confidence']:
                        results['ball'] = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'class': 'ball'
                        }
        
        return results
    
    def _calculate_center(self, bbox):
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) // 2, (y1 + y2) // 2]