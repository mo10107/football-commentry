"""
Football object tracker using Supervision library
"""

import numpy as np
import supervision as sv
from .kalman_filter import KalmanTracker

class FootballTracker:
    def __init__(self, tracker_type="ByteTrack", track_threshold=0.6, track_buffer=30, 
                 match_threshold=0.8, frame_rate=30):
        """
        Initialize football object tracker
        
        Args:
            tracker_type: Type of tracker to use ('ByteTrack' or 'BoTSORT')
            track_threshold: Detection confidence threshold for tracking
            track_buffer: Number of frames to keep track of lost objects
            match_threshold: Threshold for matching detections to tracks
            frame_rate: Video frame rate
        """
        self.track_threshold = track_threshold
        
        # Initialize tracker based on type
        if tracker_type == "ByteTrack":
            self.tracker = sv.ByteTrack(
                track_activation_threshold=track_threshold,
                lost_track_buffer=track_buffer,
                minimum_matching_threshold=match_threshold,
                frame_rate=frame_rate
            )
            """        
        elif tracker_type == "BoTSORT":
            self.tracker = sv.BoTSORT(
                track_thresh=track_threshold,
                track_buffer=track_buffer,
                match_thresh=match_threshold,
                frame_rate=frame_rate
            )
            """
        else:
            raise ValueError(f"Unknown tracker type: {tracker_type}")
        
        # Initialize ball tracker with Kalman filter
        self.ball_tracker = KalmanTracker()
        
        # ID counters
        self.next_track_id = 1
        
        print(f"Football tracker initialized: {tracker_type}")
    
    def update(self, detections, frame):
        """
        Update tracks with new detections
        
        Args:
            detections: Dict with 'players' and 'ball' detections from detector
            frame: Current video frame
            
        Returns:
            tuple: (player_tracks, ball_track)
        """
        # Extract player detections
        player_detections = detections['players']
        
        # Format player detections for supervision
        if player_detections:
            # Convert to numpy arrays for Supervision
            boxes = np.array([det['bbox'] for det in player_detections])
            confidences = np.array([det['confidence'] for det in player_detections])
            class_ids = np.ones(len(player_detections))  # All are class 1 (player)
            
            # Create Supervision Detections object
            sv_detections = sv.Detections(
                xyxy=boxes,
                confidence=confidences,
                class_id=class_ids
            )
            
            # Update tracker
            sv_tracks = self.tracker.update(detections=sv_detections)
            
            # Convert to our format
            player_tracks = []
            for i in range(len(sv_tracks.xyxy)):
                track_id = sv_tracks.tracker_id[i]
                bbox = sv_tracks.xyxy[i]
                confidence = sv_tracks.confidence[i] if hasattr(sv_tracks, 'confidence') else 1.0
                
                player_tracks.append({
                    'track_id': int(track_id),
                    'bbox': bbox.tolist(),
                    'confidence': float(confidence),
                    'center': [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                })
        else:
            player_tracks = []
        
        # Handle ball tracking
        ball_track = None
        if detections['ball'] is not None:
            ball_bbox = detections['ball']['bbox']
            ball_center = [(ball_bbox[0] + ball_bbox[2]) / 2, (ball_bbox[1] + ball_bbox[3]) / 2]
            
            # Update ball tracker with Kalman filter
            ball_state = self.ball_tracker.update(np.array(ball_center))
            
            # Create ball track info
            ball_track = {
                'bbox': ball_bbox,
                'confidence': detections['ball']['confidence'],
                'center': ball_center,
                'velocity': ball_state[1:3].tolist() if ball_state is not None else [0, 0]
            }
        
        return player_tracks, ball_track