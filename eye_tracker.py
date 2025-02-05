import cv2
import numpy as np
import base64
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RoboflowGazeTracker:
    def __init__(self):
        self.DISTANCE_TO_OBJECT = 1000  # mm
        self.HEIGHT_OF_HUMAN_FACE = 250  # mm
        self.GAZE_DETECTION_URL = "http://127.0.0.1:9001/gaze/gaze_detection"
        self.api_key = os.getenv("ROBOFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("ROBOFLOW_API_KEY environment variable not set")
        self.debug = False  # Debug visualization flag
        
        # History buffer for smoothing
        self.gaze_history_length = 5
        self.gaze_history = []
        
    def detect_gazes(self, frame):
        img_encode = cv2.imencode(".jpg", frame)[1]
        img_base64 = base64.b64encode(img_encode)
        
        try:
            resp = requests.post(
                self.GAZE_DETECTION_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "image": {"type": "base64", "value": img_base64.decode("utf-8")},
                },
                timeout=5
            )
            
            if resp.status_code != 200:
                print(f"Server error: Status {resp.status_code} - {resp.text}")
                return []
                
            result = resp.json()
            if not result:
                print("No response data from server")
                return []
                
            gazes = result[0]["predictions"]
            if not gazes:
                print("No gazes detected in frame")
            return gazes
            
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error: Could not connect to server at {self.GAZE_DETECTION_URL}")
            print(f"Make sure the Roboflow server is running and accessible")
            return []
        except requests.exceptions.Timeout:
            print("Request timed out. Server took too long to respond")
            return []
        except Exception as e:
            print(f"Error in gaze detection: {str(e)}")
            return []

    def get_gaze_point(self, frame, gaze):
        image_height, image_width = frame.shape[:2]
        
        # Calculate pixels to real-world units ratio
        length_per_pixel = self.HEIGHT_OF_HUMAN_FACE / gaze["face"]["height"]
        
        # Calculate screen coordinates from pitch and yaw
        dx = -self.DISTANCE_TO_OBJECT * np.tan(gaze['yaw']) / length_per_pixel
        dx = dx if not np.isnan(dx) else 0
        
        dy = -self.DISTANCE_TO_OBJECT * np.arccos(gaze['yaw']) * np.tan(gaze['pitch']) / length_per_pixel
        dy = dy if not np.isnan(dy) else 0
        
        # Get screen point
        x = int(image_width / 2 + dx)
        y = int(image_height / 2 + dy)
        
        # Clip to screen bounds
        x = np.clip(x, 0, image_width)
        y = np.clip(y, 0, image_height)
        
        return (x, y)

    def get_smoothed_gaze(self, current_point):
        # Add current point to history
        self.gaze_history.append(current_point)
        
        # Keep only last N points
        if len(self.gaze_history) > self.gaze_history_length:
            self.gaze_history.pop(0)
        
        # Calculate smoothed position
        if len(self.gaze_history) > 0:
            x = int(np.mean([p[0] for p in self.gaze_history]))
            y = int(np.mean([p[1] for p in self.gaze_history]))
            return (x, y)
        
        return current_point

    def draw_debug(self, frame, gaze, gaze_point):
        # Draw face bounding box
        face = gaze["face"]
        x_min = int(face["x"] - face["width"] / 2)
        x_max = int(face["x"] + face["width"] / 2)
        y_min = int(face["y"] - face["height"] / 2)
        y_max = int(face["y"] + face["height"] / 2)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        
        # Draw gaze direction arrow
        _, imgW = frame.shape[:2]
        arrow_length = imgW / 4
        dx = -arrow_length * np.sin(gaze["yaw"]) * np.cos(gaze["pitch"])
        dy = -arrow_length * np.sin(gaze["pitch"])
        cv2.arrowedLine(
            frame,
            (int(face["x"]), int(face["y"])),
            (int(face["x"] + dx), int(face["y"] + dy)),
            (0, 0, 255),
            2,
            cv2.LINE_AA,
            tipLength=0.2,
        )
        
        # Draw gaze point
        cv2.circle(frame, gaze_point, 10, (0, 255, 0), -1)
        
        return frame

    def get_gaze(self, frame):
        """Main method to get gaze point from a frame"""
        gazes = self.detect_gazes(frame)
        
        if not gazes:
            return None, frame
            
        gaze = gazes[0]  # Use first detected face
        raw_point = self.get_gaze_point(frame, gaze)
        smooth_point = self.get_smoothed_gaze(raw_point)
        
        if self.debug:
            frame = self.draw_debug(frame, gaze, smooth_point)
            
        return smooth_point, frame