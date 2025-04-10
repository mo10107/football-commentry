"""
Kalman filter implementation for smoother tracking
"""

import numpy as np
from filterpy.kalman import KalmanFilter

class KalmanTracker:
    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        """
        Initialize Kalman filter for tracking
        
        Args:
            process_noise: Process noise parameter (Q)
            measurement_noise: Measurement noise parameter (R)
        """
        # Initialize Kalman filter
        # State vector: [x, vx, y, vy]
        # Measurement vector: [x, y]
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 1, 0, 0],  # x = x + vx
            [0, 1, 0, 0],  # vx = vx
            [0, 0, 1, 1],  # y = y + vy
            [0, 0, 0, 1]   # vy = vy
        ])
        
        # Measurement function (we only measure position, not velocity)
        self.kf.H = np.array([
            [1, 0, 0, 0],  # measure x
            [0, 0, 1, 0]   # measure y
        ])
        
        # Measurement noise
        self.kf.R = np.eye(2) * measurement_noise
        
        # Process noise
        self.kf.Q = np.eye(4) * process_noise
        
        # Initial state covariance
        self.kf.P = np.eye(4) * 100
        
        # Tracker state
        self.initialized = False
        self.missed_updates = 0
        self.max_missed_updates = 10  # Reset after N missed updates
    
    def update(self, measurement):
        """
        Update Kalman filter with new measurement
        
        Args:
            measurement: [x, y] position of object
            
        Returns:
            numpy.ndarray: Current state estimate [x, vx, y, vy]
        """
        # Convert to numpy array if not already
        measurement = np.asarray(measurement)
        
        # Initialize if first measurement
        if not self.initialized:
            self.kf.x = np.array([measurement[0], 0, measurement[1], 0])
            self.initialized = True
            return self.kf.x
        
        # If measurement is None (detection lost), predict without update
        if measurement is None:
            self.missed_updates += 1
            if self.missed_updates > self.max_missed_updates:
                self.initialized = False
                return None
            
            self.kf.predict()
            return self.kf.x
        
        # Reset missed update counter
        self.missed_updates = 0
        
        # Predict and update
        self.kf.predict()
        self.kf.update(measurement)
        
        return self.kf.x
    
    def get_state(self):
        """
        Get current state estimate
        
        Returns:
            numpy.ndarray: Current state estimate [x, vx, y, vy]
        """
        if not self.initialized:
            return None
        
        return self.kf.x
    
    def reset(self):
        """Reset the Kalman filter"""
        self.initialized = False
        self.missed_updates = 0
        self.kf.P = np.eye(4) * 100