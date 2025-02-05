import cv2
from eye_tracker import RoboflowGazeTracker

def main():
    print("Starting gaze tracking...")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
        
    # Initialize tracker with debug mode
    tracker = RoboflowGazeTracker()
    tracker.debug = True  # Enable debug visualization
    
    print("Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
                
            # Get gaze point and debug frame
            gaze_point, debug_frame = tracker.get_gaze(frame)
            
            if gaze_point is not None:
                print(f"Gaze point detected at: {gaze_point}")
            
            # Show the frame
            cv2.imshow('Gaze Tracking', debug_frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nGaze tracking stopped by user")
    except Exception as e:
        print(f"Error during gaze tracking: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 