import cv2
import numpy as np
from collections import deque
import time
import os
from pathlib import Path

class FireDetector:
    def __init__(self, sensitivity=0.6, min_fire_area=500):
        """
        Initialize the Fire Detection System
        
        Args:
            sensitivity: Fire detection sensitivity (0.3-1.0, higher = more sensitive)
            min_fire_area: Minimum contour area to be considered as fire (in pixels)
        """
        self.sensitivity = sensitivity
        self.min_fire_area = min_fire_area
        self.fire_detected_frames = deque(maxlen=5)
        self.alarm_triggered = False
        
    def detect_fire_by_color(self, frame):
        """
        Detect fire using color-based HSV thresholding
        Fire typically has Red, Orange, Yellow colors
        """
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color range for fire (Red, Orange, Yellow)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Orange-Yellow range
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([25, 255, 255])
        
        # Create masks for each color range
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        
        # Combine all masks
        fire_mask = cv2.bitwise_or(mask_red1, mask_red2)
        fire_mask = cv2.bitwise_or(fire_mask, mask_orange)
        
        # Calculate fire percentage
        total_pixels = frame.shape[0] * frame.shape[1]
        fire_pixels = cv2.countNonZero(fire_mask)
        fire_percentage = (fire_pixels / total_pixels) * 100
        
        return fire_mask, fire_percentage
    
    def detect_fire_by_edge(self, frame):
        """
        Detect fire using edge detection (flickering effect of flames)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Detect edges
        edges = cv2.Canny(filtered, 50, 150)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        return edges
    
    def detect_fire_by_motion(self, frame, prev_frame):
        """
        Detect fire by analyzing frame differences (flickering)
        """
        if prev_frame is None:
            return None
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Apply threshold
        _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)
        
        return motion_mask
    
    def draw_fire_alert(self, frame, fire_detected, confidence):
        """
        Draw fire alert on the frame with visual warnings
        
        Args:
            frame: Current frame
            fire_detected: Boolean indicating if fire is detected
            confidence: Confidence score (0-100)
            
        Returns:
            frame: Frame with alert drawn
        """
        h, w = frame.shape[:2]
        
        if fire_detected:
            # Draw thick red border around frame
            cv2.rectangle(frame, (5, 5), (w-5, h-5), (0, 0, 255), 8)
            
            # Create semi-transparent red overlay for alert effect
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Add large FIRE ALERT text at top
            alert_text = "FIRE ALERT!"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.0
            thickness = 4
            text_size = cv2.getTextSize(alert_text, font, font_scale, thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = 80
            
            # Add background rectangle for text
            cv2.rectangle(frame, (text_x - 20, text_y - 50), (text_x + text_size[0] + 20, text_y + 10),
                         (0, 0, 200), -1)
            cv2.putText(frame, alert_text, (text_x, text_y),
                       font, font_scale, (0, 255, 255), thickness)
            
            # Add flashing effect with timestamp
            current_time = time.time()
            if int(current_time * 2) % 2 == 0:  # Flash every 500ms
                cv2.rectangle(frame, (10, 10), (w-10, 100), (0, 0, 255), 5)
            
            # Add confidence bar at bottom
            bar_height = 40
            bar_width = int((confidence / 100) * (w - 20))
            cv2.rectangle(frame, (10, h - bar_height - 15), (10 + bar_width, h - 15),
                         (0, 0, 255), -1)
            cv2.rectangle(frame, (10, h - bar_height - 15), (w - 10, h - 15),
                         (255, 255, 255), 3)
            cv2.putText(frame, f"ALERT LEVEL: {confidence:.1f}%", (20, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Add blinking warning triangles
            triangle_y = h // 2
            pts1 = np.array([[50, triangle_y - 30], [20, triangle_y + 30], [80, triangle_y + 30]], np.int32)
            pts2 = np.array([[w - 50, triangle_y - 30], [w - 80, triangle_y + 30], [w - 20, triangle_y + 30]], np.int32)
            cv2.polylines(frame, [pts1], True, (0, 0, 255), 4)
            cv2.polylines(frame, [pts2], True, (0, 0, 255), 4)
            
        else:
            # Normal display when no fire
            cv2.rectangle(frame, (5, 5), (w-5, h-5), (0, 255, 0), 3)
            cv2.putText(frame, "STATUS: SAFE", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            # Confidence bar (green)
            bar_height = 40
            bar_width = int((confidence / 100) * (w - 20))
            cv2.rectangle(frame, (10, h - bar_height - 15), (10 + bar_width, h - 15),
                         (0, 255, 0), -1)
            cv2.rectangle(frame, (10, h - bar_height - 15), (w - 10, h - 15),
                         (255, 255, 255), 3)
            cv2.putText(frame, f"CONFIDENCE: {confidence:.1f}%", (20, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return frame
    
    def analyze_frame(self, frame, prev_frame=None):
        """
        Comprehensive fire detection using multiple methods
        """
        # Method 1: Color-based detection
        color_mask, fire_percentage = self.detect_fire_by_color(frame)
        
        # Method 2: Edge detection
        edge_mask = self.detect_fire_by_edge(frame)
        
        # Method 3: Motion detection
        motion_mask = None
        if prev_frame is not None:
            motion_mask = self.detect_fire_by_motion(frame, prev_frame)
        
        # Combine masks
        combined_mask = color_mask.copy()
        if motion_mask is not None:
            combined_mask = cv2.bitwise_and(combined_mask, motion_mask)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by minimum area
        fire_contours = [c for c in contours if cv2.contourArea(c) > self.min_fire_area]
        
        # Calculate confidence score
        contour_score = min(len(fire_contours) * 20, 50)
        color_score = min(fire_percentage * self.sensitivity, 50)
        confidence = contour_score + color_score
        
        # Determine if fire is detected
        fire_detected = confidence > (100 - (self.sensitivity * 100))
        
        return fire_detected, confidence, combined_mask, fire_contours
    
    def process_camera_feed(self, camera_index=0, display=True):
        """
        Process live camera feed for fire detection
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("✗ Error: Could not open camera")
            return
        
        print("=" * 60)
        print("FIRE DETECTION SYSTEM - LIVE CAMERA")
        print("=" * 60)
        print("Press 'q' to quit, 'c' to capture screenshot")
        print("-" * 60)
        
        prev_frame = None
        frame_count = 0
        fire_frame_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("✗ Error: Failed to read frame")
                break
            
            frame_count += 1
            
            # Analyze frame
            fire_detected, confidence, fire_mask, fire_contours = self.analyze_frame(frame, prev_frame)
            
            if fire_detected:
                fire_frame_count += 1
            
            # Prepare status
            status = "🔥 FIRE DETECTED!" if fire_detected else "✓ NO FIRE"
            status_color = (0, 0, 255) if fire_detected else (0, 255, 0)
            
            # Print analysis
            print(f"Frame: {frame_count} | Confidence: {confidence:.1f}/100 | Status: {status}")
            
            if display:
                display_frame = frame.copy()
                h, w = display_frame.shape[:2]
                
                # Draw fire alert on frame
                display_frame = self.draw_fire_alert(display_frame, fire_detected, confidence)
                
                # Add sensitivity info
                cv2.putText(display_frame, f"Sensitivity: {self.sensitivity:.1f}", (10, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Draw fire contours
                cv2.drawContours(display_frame, fire_contours, -1, (0, 255, 255), 2)
                
                # Create visualization with masks
                fire_mask_colored = cv2.cvtColor(fire_mask, cv2.COLOR_GRAY2BGR)
                display_combined = np.hstack([display_frame, fire_mask_colored])
                
                cv2.imshow("Fire Detection System", display_combined)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"fire_detection_{timestamp}.jpg", frame)
                print(f"✓ Screenshot saved: fire_detection_{timestamp}.jpg")
            
            prev_frame = frame.copy()
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("-" * 60)
        print(f"Total frames: {frame_count}")
        print(f"Fire detected in: {fire_frame_count} frames")
        print("=" * 60)
    
    def process_video_file(self, video_path, display=True):
        """
        Analyze a video file for fire detection
        """
        if not os.path.exists(video_path):
            print(f"✗ Error: Video file not found: {video_path}")
            return
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"✗ Error: Could not open video {video_path}")
            return
        
        print("=" * 60)
        print(f"Analyzing video: {video_path}")
        print("=" * 60)
        print("Press 'q' to stop playback")
        print("-" * 60)
        
        prev_frame = None
        frame_count = 0
        fire_frame_count = 0
        fire_detections = []
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video Info: {width}x{height} @ {fps} fps, Total frames: {total_frames}")
        print("-" * 60)
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Analyze frame
            fire_detected, confidence, fire_mask, fire_contours = self.analyze_frame(frame, prev_frame)
            
            if fire_detected:
                fire_frame_count += 1
                fire_detections.append({
                    'frame': frame_count,
                    'confidence': confidence
                })
            
            # Display status every 10 frames
            if frame_count % 10 == 0:
                status = "🔥 FIRE!" if fire_detected else "NO FIRE"
                print(f"Frame {frame_count}/{total_frames}: {status} (Confidence: {confidence:.1f}%)")
            
            if display:
                display_frame = frame.copy()
                
                # Draw fire alert on frame
                display_frame = self.draw_fire_alert(display_frame, fire_detected, confidence)
                
                # Add frame counter
                cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames}", (10, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Draw contours
                cv2.drawContours(display_frame, fire_contours, -1, (0, 255, 255), 2)
                
                # Show frame
                fire_mask_colored = cv2.cvtColor(fire_mask, cv2.COLOR_GRAY2BGR)
                display_combined = np.hstack([display_frame, fire_mask_colored])
                
                cv2.imshow("Fire Detection - Video", display_combined)
                
                # Handle key press
                key = cv2.waitKey(int(1000 / fps)) & 0xFF
                if key == ord('q'):
                    print("[*] Video playback stopped by user")
                    break
            
            prev_frame = frame.copy()
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("-" * 60)
        print(f"Total frames: {frame_count}")
        print(f"Fire detected in: {fire_frame_count} frames ({(fire_frame_count/frame_count)*100:.1f}%)")
        if fire_detections:
            print(f"\nFire detection events (first 5):")
            for detection in fire_detections[:5]:
                print(f"  Frame {detection['frame']}: Confidence {detection['confidence']:.1f}%")
        print("=" * 60)
    
    def process_image(self, image_path, display=True):
        """
        Analyze a single image for fire
        """
        if not os.path.exists(image_path):
            print(f"✗ Error: Image file not found: {image_path}")
            return
        
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"✗ Error: Could not read image {image_path}")
            return
        
        fire_detected, confidence, fire_mask, fire_contours = self.analyze_frame(frame)
        
        status = "🔥 FIRE DETECTED!" if fire_detected else "✓ NO FIRE"
        
        print("=" * 60)
        print(f"Image Analysis: {image_path}")
        print(f"Confidence: {confidence:.1f}/100")
        print(f"Status: {status}")
        print("=" * 60)
        print("Press any key to close the window...")
        print("-" * 60)
        
        if display:
            display_frame = frame.copy()
            
            # Draw fire alert on frame
            display_frame = self.draw_fire_alert(display_frame, fire_detected, confidence)
            
            # Draw contours
            cv2.drawContours(display_frame, fire_contours, -1, (0, 255, 255), 2)
            
            # Show result
            fire_mask_3channel = cv2.cvtColor(fire_mask, cv2.COLOR_GRAY2BGR)
            result = np.hstack([display_frame, fire_mask_3channel])
            
            cv2.imshow("Fire Detection Result", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def display_menu():
    """Display main menu"""
    print("\n" + "=" * 60)
    print("🔥 FIRE DETECTION SYSTEM - MAIN MENU 🔥")
    print("=" * 60)
    print("1. Live Camera Feed (Press ENTER for default)")
    print("2. Analyze Image File")
    print("3. Analyze Video File")
    print("4. Settings")
    print("5. Exit")
    print("=" * 60)


def display_settings_menu():
    """Display settings menu"""
    print("\n" + "-" * 60)
    print("⚙️  SETTINGS MENU")
    print("-" * 60)
    print("1. Adjust Sensitivity (Current: 0.6)")
    print("2. Adjust Minimum Fire Area (Current: 500px)")
    print("3. Back to Main Menu")
    print("-" * 60)


def get_file_path(file_type):
    """Get file path from user with validation"""
    while True:
        print(f"\nEnter the path to your {file_type} file:")
        print("(Drag and drop the file here, or type the full path)")
        file_path = input("Path: ").strip()
        
        # Remove quotes if user dragged file
        if file_path.startswith('"') and file_path.endswith('"'):
            file_path = file_path[1:-1]
        
        # Check if file exists
        if os.path.exists(file_path):
            return file_path
        else:
            print(f"✗ File not found: {file_path}")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry != 'y':
                return None


def main():
    """Main program loop"""
    detector = FireDetector(sensitivity=0.6, min_fire_area=500)
    
    print("\n" + "█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + "  🔥 WELCOME TO FIRE DETECTION SYSTEM 🔥".center(58) + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60)
    
    while True:
        display_menu()
        
        choice = input("\nEnter your choice (1-5) or press ENTER for camera: ").strip()
        
        # Default to camera feed if user presses ENTER
        if choice == "" or choice == "1":
            print("\n[*] Starting camera feed...")
            time.sleep(1)
            detector.process_camera_feed(camera_index=0, display=True)
        
        elif choice == "2":
            print("\n[*] IMAGE ANALYSIS")
            image_path = get_file_path("image")
            if image_path:
                detector.process_image(image_path, display=True)
            else:
                print("✗ Operation cancelled")
        
        elif choice == "3":
            print("\n[*] VIDEO ANALYSIS")
            video_path = get_file_path("video")
            if video_path:
                detector.process_video_file(video_path, display=True)
            else:
                print("✗ Operation cancelled")
        
        elif choice == "4":
            while True:
                display_settings_menu()
                settings_choice = input("\nEnter your choice (1-3): ").strip()
                
                if settings_choice == "1":
                    try:
                        sensitivity = float(input("Enter sensitivity (0.3-1.0): "))
                        if 0.3 <= sensitivity <= 1.0:
                            detector.sensitivity = sensitivity
                            print(f"✓ Sensitivity updated to {sensitivity}")
                        else:
                            print("✗ Sensitivity must be between 0.3 and 1.0")
                    except ValueError:
                        print("✗ Invalid input. Please enter a number.")
                
                elif settings_choice == "2":
                    try:
                        min_area = int(input("Enter minimum fire area in pixels (100-5000): "))
                        if 100 <= min_area <= 5000:
                            detector.min_fire_area = min_area
                            print(f"✓ Minimum fire area updated to {min_area}px")
                        else:
                            print("✗ Area must be between 100 and 5000 pixels")
                    except ValueError:
                        print("✗ Invalid input. Please enter a number.")
                
                elif settings_choice == "3":
                    break
                
                else:
                    print("✗ Invalid choice. Please try again.")
        
        elif choice == "5":
            print("\n" + "=" * 60)
            print("👋 Thank you for using Fire Detection System!")
            print("=" * 60 + "\n")
            break
        
        else:
            print("✗ Invalid choice. Please enter 1-5 or press ENTER.")


# Main execution
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[*] Program interrupted by user")
        print("Exiting Fire Detection System...\n")