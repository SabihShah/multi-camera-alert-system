import cv2
import numpy as np
import threading
import time
import pygame
from collections import deque
import math
import queue
import signal
import os
from typing import Optional, List, Dict, Tuple

class MultipleCameraBoundaryAlarmSystem:
    def __init__(self, camera_sources=[]):
        """Initialize multi-camera boundary alarm system with named cameras
        
        Args:
            camera_sources: List of dictionaries with 'name' and 'url' keys
                          e.g., [{'name': 'Front Door', 'url': 'rtsp://...'}, 
                                 {'name': 'Back Yard', 'url': 0}]
        """
        # Initialize pygame for sound
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        pygame.mixer.init()
        pygame.init()

        self.camera_sources = camera_sources
        self.cameras = {}
        self.camera_states = {}
        self.latest_frames = {}
        
        # Main display settings
        self.main_display_camera = None
        self.main_window_name = "Multi-Camera Security System - Grid View"
        
        # Alert window settings
        self.alert_window_size = (1280, 720)
        self.active_alert_windows = {}
        self.alert_window_positions = {}
        self.alert_frames = {}
        self.alert_window_z_order = {}
        self.windows_to_recreate = set()  # Track windows that need recreation for focus
        
        # Initialize person detection
        self.init_person_detection()
        
        # Initialize cameras with optimized OpenCV
        self.init_cameras()
        
        if not self.cameras:
            raise Exception("No cameras could be opened!")
        
        # Set initial main display camera
        self.main_display_camera = list(self.cameras.keys())[0]
        
        # Shared alarm sound
        self.alarm_sound = self.create_alarm_sound()
        self.global_alarm_active = False
        self.alarm_thread = None
        
        # Threading locks
        self.frame_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.alert_lock = threading.Lock()
        self.windows_to_recreate = set()  # Initialize here too for safety

        self.print_controls()
    
    def calculate_optimal_grid_layout(self, num_cameras):
        """Calculate optimal grid layout based on number of cameras"""
        if num_cameras <= 0:
            return 1, 1, (600, 400)
        elif num_cameras == 1:
            return 1, 1, (800, 600)
        elif num_cameras == 2:
            return 2, 1, (640, 480)
        elif num_cameras == 3:
            return 3, 1, (426, 320)
        elif num_cameras == 4:
            return 2, 2, (640, 480)
        elif num_cameras <= 6:
            return 3, 2, (426, 320)
        elif num_cameras <= 8:
            return 4, 2, (320, 240)
        elif num_cameras == 9:
            return 3, 3, (426, 320)
        elif num_cameras <= 12:
            return 4, 3, (320, 240)
        elif num_cameras <= 16:
            return 4, 4, (320, 240)
        else:
            cols = int(math.ceil(math.sqrt(num_cameras)))
            rows = int(math.ceil(num_cameras / cols))
            size = max(200, min(400, 1200 // cols))
            return cols, rows, (size, int(size * 0.75))
    
    def init_cameras(self):
        """Initialize cameras with optimized OpenCV"""
        for idx, camera_config in enumerate(self.camera_sources):
            # Extract camera info from dictionary
            if isinstance(camera_config, dict):
                camera_name = camera_config.get('name', f'Camera {idx}')
                source = camera_config.get('url', 0)
                camera_id = f"cam_{idx}"
            else:
                # Backward compatibility for old format
                camera_name = f'Camera {idx}'
                source = camera_config
                camera_id = f"cam_{idx}"
            
            print(f"\n🎥 Initializing '{camera_name}' (ID: {camera_id})")
            
            try:
                cap = None
                
                # Initialize based on source type
                if isinstance(source, str) and source.startswith('rtsp'):
                    print(f"Setting up RTSP stream for '{camera_name}'...")
                    
                    # Set environment variables for OpenCV RTSP optimization
                    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|reorder_queue_size;0'
                    
                    # Create capture with optimized FFMPEG settings
                    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                    
                    if cap.isOpened():
                        # Configure RTSP for high quality and performance
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffering
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        # Try to set high resolution
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                        
                        # Test frame read
                        for attempt in range(10):
                            ret, test_frame = cap.read()
                            if ret and test_frame is not None:
                                self.cameras[camera_id] = cap
                                print(f"✓ RTSP '{camera_name}' initialized at {test_frame.shape[1]}x{test_frame.shape[0]}")
                                cap = None  # Don't release
                                break
                            time.sleep(0.3)
                        
                        if cap is not None:
                            print(f"✗ RTSP '{camera_name}' opened but no frames received")
                            cap.release()
                            cap = None
                    else:
                        print(f"✗ Could not open '{camera_name}' with RTSP")
                        continue
                        
                else:
                    # Local camera with OpenCV
                    print(f"Setting up local camera '{camera_name}'...")
                    cap = cv2.VideoCapture(source)
                    
                    if cap.isOpened():
                        # Try to set high resolution for local cameras
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            self.cameras[camera_id] = cap
                            print(f"✓ Local '{camera_name}' initialized at {test_frame.shape[1]}x{test_frame.shape[0]}")
                            cap = None  # Don't release
                        else:
                            print(f"✗ Local '{camera_name}' not producing frames")
                            cap.release()
                            continue
                    else:
                        print(f"✗ Could not open local camera '{camera_name}'")
                        continue
                
                # If we get here, camera should be initialized
                if camera_id not in self.cameras:
                    print(f"✗ Failed to initialize '{camera_name}' with OpenCV")
                    continue
                
                # Get actual frame dimensions from a working camera
                working_cap = self.cameras[camera_id]
                ret, test_frame = working_cap.read()
                if ret and test_frame is not None:
                    actual_width = test_frame.shape[1]
                    actual_height = test_frame.shape[0]
                else:
                    # Use defaults if we can't get a frame
                    actual_width = 1920
                    actual_height = 1080
                
                # Initialize camera state with name
                self.camera_states[camera_id] = {
                    'is_alarm_active': False,
                    'boundary_lines': [],
                    'current_line': [],
                    'is_drawing': False,
                    'blink_state': False,
                    'last_blink_time': 0,
                    'blink_interval': 0.3,
                    'window_name': f'{camera_name} - Thumbnail',
                    'priority_score': 0,
                    'frame_width': actual_width,
                    'frame_height': actual_height,
                    'person_detection_enabled': True,
                    'person_confidence_threshold': 0.5,
                    'detected_persons': [],
                    'last_successful_read': time.time(),
                    'connection_timeout': 30.0,
                    'retry_count': 0,
                    'max_retries': 3,
                    'alert_window_name': f'SECURITY ALERT - {camera_name}',
                    'alert_triggered_time': None,
                    'camera_name': camera_name  # Store the display name
                }
                
                self.latest_frames[camera_id] = None
                
            except Exception as e:
                print(f"Error initializing '{camera_name}': {e}")
                continue
    
    def init_person_detection(self):
        """Initialize person detection using YOLO"""
        self.detection_methods = []
        try:
            from ultralytics import YOLO
            import torch

            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            self.yolo_model = YOLO('yolov8n.pt')
            self.yolo_model.to(device)

            if device == 'cuda':
                print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
                print(f"✓ GPU Name: {torch.cuda.get_device_name(0)}")
                
            self.detection_methods.append(('yolov8', self.yolo_model, None))
            print("✓ YOLOv8 person detection loaded")
        except Exception as e:
            print(f"⚠ YOLOv8 loading failed: {e}")
        
        if not self.detection_methods:
            print("⚠ No person detection methods available.")
    
    def detect_persons_yolo(self, frame, model, _):
        """Detect persons using YOLOv8"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(rgb_frame, conf=0.5, classes=[0])[0]
            
            persons = []
            for r in results.boxes.data:
                x1, y1, x2, y2, conf, cls = r.tolist()
                if int(cls) == 0:  # Person class
                    w = x2 - x1
                    h = y2 - y1
                    
                    persons.append({
                        'bbox': (int(x1), int(y1), int(w), int(h)),
                        'confidence': float(conf)
                    })
            
            return persons
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return []
    
    def detect_persons(self, frame, camera_idx):
        """Detect persons using available methods for specific camera"""
        state = self.camera_states[camera_idx]
        
        if not state['person_detection_enabled'] or not self.detection_methods:
            return []
        
        all_persons = []
        
        for method_name, *method_data in self.detection_methods:
            try:
                if method_name == 'yolov8':
                    net, _ = method_data
                    persons = self.detect_persons_yolo(frame, net, None)
                    all_persons.extend(persons)
                    
                    if persons and method_name == 'yolov8':
                        break
                        
            except Exception as e:
                print(f"Camera {camera_idx} - Error in {method_name} detection: {e}")
                continue
        
        filtered_persons = self.filter_duplicate_detections(all_persons)
        state['detected_persons'] = filtered_persons
        return filtered_persons
    
    def filter_duplicate_detections(self, persons):
        """Filter out duplicate person detections based on overlap"""
        if len(persons) <= 1:
            return persons
        
        filtered = []
        for person in persons:
            x1, y1, w1, h1 = person['bbox']
            is_duplicate = False
            
            for existing in filtered:
                x2, y2, w2, h2 = existing['bbox']
                
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                area1 = w1 * h1
                area2 = w2 * h2
                union_area = area1 + area2 - overlap_area
                
                if union_area > 0 and overlap_area / union_area > 0.3:
                    is_duplicate = True
                    if person['confidence'] > existing['confidence']:
                        filtered.remove(existing)
                        filtered.append(person)
                    break
            
            if not is_duplicate:
                filtered.append(person)
        
        return filtered
    
    def create_alarm_sound(self):
        """Create alarm sound using pygame"""
        try:
            sample_rate = 22050
            duration = 0.5
            frequency1 = 800
            frequency2 = 600
            
            frames = int(duration * sample_rate)
            arr = np.zeros((frames, 2))
            
            for i in range(frames):
                if i < frames // 2:
                    wave = np.sin(2 * np.pi * frequency1 * i / sample_rate)
                else:
                    wave = np.sin(2 * np.pi * frequency2 * i / sample_rate)
                arr[i] = [wave * 0.3, wave * 0.3]
            
            arr = (arr * 32767).astype(np.int16)
            return pygame.sndarray.make_sound(arr)
            
        except Exception as e:
            print(f"Warning: Could not create alarm sound: {e}")
            return None
    
    def line_intersects_rectangle(self, line_start, line_end, rect_x, rect_y, rect_w, rect_h):
        """Check if a line segment intersects with a rectangle"""
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Rectangle bounds
        left = rect_x
        right = rect_x + rect_w
        top = rect_y
        bottom = rect_y + rect_h
        
        # Check if line endpoints are inside rectangle
        if (left <= x1 <= right and top <= y1 <= bottom) or (left <= x2 <= right and top <= y2 <= bottom):
            return True
        
        # Check intersection with rectangle edges
        def line_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
            """Check if two line segments intersect"""
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-10:
                return False
            
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
            
            return 0 <= t <= 1 and 0 <= u <= 1
        
        # Check intersection with each edge of rectangle
        edges = [
            (left, top, right, top),      # top edge
            (right, top, right, bottom),  # right edge
            (right, bottom, left, bottom), # bottom edge
            (left, bottom, left, top)     # left edge
        ]
        
        for edge in edges:
            if line_intersect(x1, y1, x2, y2, edge[0], edge[1], edge[2], edge[3]):
                return True
        
        return False
    
    def is_person_crossing_boundary(self, persons, camera_idx):
        """Check if any detected person's bounding box intersects boundary lines"""
        state = self.camera_states[camera_idx]
        crossing_persons = []
        
        for person in persons:
            is_crossing = False
            x, y, w, h = person['bbox']
            
            for line in state['boundary_lines']:
                if len(line) < 2:
                    continue
                    
                for i in range(len(line) - 1):
                    p1 = line[i]
                    p2 = line[i + 1]
                    
                    if self.line_intersects_rectangle(p1, p2, x, y, w, h):
                        is_crossing = True
                        print(f"Camera {camera_idx} - Person crossing detected! Confidence: {person['confidence']:.2f}")
                        break
                        
                if is_crossing:
                    crossing_persons.append(person)
                    break
        
        return crossing_persons
    
    def check_boundary_crossing(self, persons, camera_idx):
        """Check boundary crossing using only person detection"""
        state = self.camera_states[camera_idx]
        
        if not persons or not state['boundary_lines']:
            return False
        
        if state['person_detection_enabled'] and self.detection_methods:
            crossing_persons = self.is_person_crossing_boundary(persons, camera_idx)
            
            if crossing_persons:
                self.trigger_alarm(camera_idx)
                return True
        
        return False
    
    def trigger_alarm(self, camera_idx):
        """Trigger the alarm system for specific camera"""
        with self.state_lock:
            state = self.camera_states[camera_idx]
            
            if state['is_alarm_active']:
                return
                
            state['is_alarm_active'] = True
            state['priority_score'] += 1000
            state['alert_triggered_time'] = time.time()
            
            # Open alert window for this specific camera
            self.open_alert_window(camera_idx)
            self.global_alarm_active = True
            
            if self.alarm_sound and (self.alarm_thread is None or not self.alarm_thread.is_alive()):
                self.alarm_thread = threading.Thread(target=self.play_alarm, daemon=True)
                self.alarm_thread.start()
    
    def calculate_window_position(self, camera_idx):
        """Calculate position for alert window to avoid overlap"""
        window_width, window_height = self.alert_window_size
        screen_width = 1920
        screen_height = 1080
        
        # Get list of active alert windows
        active_windows = len(self.active_alert_windows)
        
        # Calculate position based on number of active windows
        positions = [
            (50, 50),  # Top-left
            (screen_width - window_width - 50, 50),  # Top-right
            (50, screen_height - window_height - 100),  # Bottom-left
            (screen_width - window_width - 50, screen_height - window_height - 100),  # Bottom-right
        ]
        
        if active_windows < len(positions):
            return positions[active_windows]
        else:
            # For more than 4 windows, stack them with offsets
            offset = (active_windows - 4) * 30
            base_x, base_y = positions[active_windows % 4]
            return base_x + offset, base_y + offset
    
    def open_alert_window(self, camera_idx):
        """Open alert window for specific camera"""
        with self.alert_lock:
            state = self.camera_states[camera_idx]
            alert_window_name = state['alert_window_name']
            
            # Don't open if already exists
            if camera_idx in self.active_alert_windows:
                print(f"Alert window for Camera {camera_idx} already exists")
                return
            
            # Add to active windows
            self.active_alert_windows[camera_idx] = alert_window_name
            
            # Calculate window position
            x, y = self.calculate_window_position(camera_idx)
            self.alert_window_positions[camera_idx] = (x, y)
            
            # Set initial z-order
            self.alert_window_z_order[camera_idx] = time.time()
            
            print(f"📺 Alert window opened for '{state['camera_name']}' at position ({x}, {y})")
    
    def bring_alert_to_front(self, camera_idx):
        """Bring specific camera's alert window to front - Using recreation flag"""
        with self.alert_lock:
            if camera_idx not in self.active_alert_windows:
                return False
                
            state = self.camera_states[camera_idx]
            alert_window_name = state['alert_window_name']
            
            try:
                # Update z-order to make this window most recent
                self.alert_window_z_order[camera_idx] = time.time()
                
                # Mark this window for recreation to bring it to front
                self.windows_to_recreate.add(camera_idx)
                
                print(f"Alert window for Camera {camera_idx} marked for front focus")
                return True
                
            except Exception as e:
                print(f"Error bringing alert window to front for Camera {camera_idx}: {e}")
                return False
    
    def close_alert_window(self, camera_idx):
        """Close alert window for specific camera"""
        with self.alert_lock:
            if camera_idx not in self.active_alert_windows:
                return
                
            alert_window_name = self.active_alert_windows[camera_idx]
            
            try:
                cv2.destroyWindow(alert_window_name)
            except cv2.error as e:
                print(f"Warning: Error closing alert window for Camera {camera_idx}: {e}")
            
            # Remove from tracking
            del self.active_alert_windows[camera_idx]
            if camera_idx in self.alert_window_positions:
                del self.alert_window_positions[camera_idx]
            if camera_idx in self.alert_frames:
                del self.alert_frames[camera_idx]
            if camera_idx in self.alert_window_z_order:
                del self.alert_window_z_order[camera_idx]
            # Also remove from recreation queue if present
            self.windows_to_recreate.discard(camera_idx)
    
    def close_all_alert_windows(self):
        """Close all alert windows"""
        with self.alert_lock:
            camera_ids_to_close = list(self.active_alert_windows.keys())
            for camera_idx in camera_ids_to_close:
                self.close_alert_window(camera_idx)
    
    def switch_main_display(self, camera_idx):
        """Switch the main display to specified camera and bring its alert to front"""
        if camera_idx in self.cameras:
            self.main_display_camera = camera_idx
            print(f"Main display switched to Camera {camera_idx}")
            
            # Force bring the alert window to front if it exists
            if camera_idx in self.active_alert_windows:
                success = self.bring_alert_to_front(camera_idx)
                if success:
                    print(f"Alert window for Camera {camera_idx} brought to front")
                else:
                    print(f"Failed to bring alert window for Camera {camera_idx} to front")
            else:
                print(f"No active alert window for Camera {camera_idx}")
    
    def update_global_alarm_state(self):
        """Update global alarm state based on individual camera states"""
        self.global_alarm_active = any(state['is_alarm_active'] for state in self.camera_states.values())
    
    def play_alarm(self):
        """Play alarm sound"""
        try:
            while self.global_alarm_active:
                if self.alarm_sound:
                    self.alarm_sound.play()
                    pygame.time.wait(800)
                    self.alarm_sound.stop()
        except Exception as e:
            print(f"Alarm sound error: {e}")
        finally:
            if self.alarm_sound:
                self.alarm_sound.stop()
    
    def reset_alarm(self, camera_idx=None):
        """Reset alarm system"""
        with self.state_lock:
            if camera_idx is None:
                # Reset all cameras
                for idx in self.cameras.keys():
                    state = self.camera_states[idx]
                    state['is_alarm_active'] = False
                    state['blink_state'] = False
                    state['priority_score'] = 0
                    state['alert_triggered_time'] = None
                    # Close individual alert window
                    self.close_alert_window(idx)
                
                print("All cameras reset - System monitoring resumed")
            else:
                # Reset specific camera
                state = self.camera_states[camera_idx]
                state['is_alarm_active'] = False
                state['blink_state'] = False
                state['priority_score'] = 0
                state['alert_triggered_time'] = None
                # Close specific alert window
                self.close_alert_window(camera_idx)
                print(f"Camera {camera_idx} - System Reset")
            
            self.update_global_alarm_state()
            
            if not self.global_alarm_active and self.alarm_sound:
                self.alarm_sound.stop()
                if self.alarm_thread and self.alarm_thread.is_alive():
                    self.global_alarm_active = False
                    self.alarm_thread.join(timeout=1.0)
                    self.alarm_thread = None
    
    def clear_boundaries(self, camera_idx):
        """Clear all drawn boundary lines for specific camera"""
        state = self.camera_states[camera_idx]
        state['boundary_lines'] = []
        state['current_line'] = []
        print(f"Camera {camera_idx} - All boundaries cleared")
    
    def toggle_person_detection(self, camera_idx):
        """Toggle person detection on/off for specific camera"""
        state = self.camera_states[camera_idx]
        
        if self.detection_methods:
            state['person_detection_enabled'] = not state['person_detection_enabled']
            status = "ENABLED" if state['person_detection_enabled'] else "DISABLED"
            print(f"Camera {camera_idx} - Person detection {status}")
        else:
            print(f"Camera {camera_idx} - Person detection not available")
    
    def apply_screen_blink_effect(self, frame, camera_idx):
        """Apply blinking red overlay effect when alarm is active"""
        state = self.camera_states[camera_idx]
        
        if not state['is_alarm_active']:
            return frame
        
        current_time = time.time()
        
        if current_time - state['last_blink_time'] > state['blink_interval']:
            state['blink_state'] = not state['blink_state']
            state['last_blink_time'] = current_time
        
        if state['blink_state']:
            red_overlay = np.zeros_like(frame)
            red_overlay[:, :] = (0, 0, 255)
            
            alpha = 0.3
            frame = cv2.addWeighted(frame, 1-alpha, red_overlay, alpha, 0)
            
            border_thickness = 10
            cv2.rectangle(frame, 
                         (0, 0), 
                         (frame.shape[1], frame.shape[0]), 
                         (0, 0, 255), 
                         border_thickness)
        
        return frame
    
    def draw_overlays(self, frame, persons, camera_idx, is_main_display=False):
        """Draw boundary lines and person detections on frame"""
        state = self.camera_states[camera_idx]
        
        # Apply screen blinking effect first if alarm is active
        frame = self.apply_screen_blink_effect(frame, camera_idx)
        
        # Draw detected persons (simple bounding boxes only)
        for person in persons:
            x, y, w, h = person['bbox']
            confidence = person['confidence']
            
            # Draw bounding box
            color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw confidence score
            label = f"Person: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw boundary lines
        for line in state['boundary_lines']:
            if len(line) > 1:
                try:
                    pts = np.array(line, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], False, (0, 255, 0), 3)
                except:
                    for i in range(len(line) - 1):
                        cv2.line(frame, line[i], line[i+1], (0, 255, 0), 3)
        
        # Draw current line being drawn
        if is_main_display and len(state['current_line']) > 1:
            try:
                pts = np.array(state['current_line'], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (0, 255, 255), 2)
            except:
                for i in range(len(state['current_line']) - 1):
                    cv2.line(frame, state['current_line'][i], state['current_line'][i+1], (0, 255, 255), 2)
        
        # Draw status information with camera name
        if state['is_alarm_active']:
            status_bg_color = (0, 0, 150) if state['blink_state'] else (0, 0, 200)
            cv2.rectangle(frame, (5, 5), (500, 35), status_bg_color, -1)
            cv2.putText(frame, f"{state['camera_name']} - PERSON ALARM!", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.rectangle(frame, (5, 5), (350, 35), (0, 100, 0), -1)
            cv2.putText(frame, f"{state['camera_name']} - MONITORING", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Additional info for main display
        if is_main_display:
            cv2.putText(frame, f"Boundaries: {len(state['boundary_lines'])}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            person_status = "ON" if state['person_detection_enabled'] else "OFF"
            color = (0, 255, 0) if state['person_detection_enabled'] else (0, 0, 255)
            cv2.putText(frame, f"Person Detection: {person_status}", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            cv2.putText(frame, f"Persons detected: {len(persons)}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            active_alarms = [idx for idx, s in self.camera_states.items() 
                            if s['is_alarm_active'] and idx != camera_idx]
            if active_alarms:
                alarm_text = f"Other alarms: {', '.join(map(str, active_alarms))}"
                cv2.putText(frame, alarm_text, (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return frame
    
    def create_grid_display_frame(self):
        """Create the main grid display frame with all cameras"""
        num_cameras = len(self.cameras)
        if num_cameras == 0:
            return np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Calculate optimal grid layout
        grid_cols, grid_rows, camera_display_size = self.calculate_optimal_grid_layout(num_cameras)
        
        display_width = grid_cols * camera_display_size[0]
        display_height = grid_rows * camera_display_size[1]
        
        grid_frame = np.zeros((display_height, display_width, 3), dtype=np.uint8)
        
        camera_list = list(self.cameras.keys())
        for i, camera_idx in enumerate(camera_list):
            if i >= grid_cols * grid_rows:
                break
                
            row = i // grid_cols
            col = i % grid_cols
            
            start_y = row * camera_display_size[1]
            end_y = start_y + camera_display_size[1]
            start_x = col * camera_display_size[0]
            end_x = start_x + camera_display_size[0]
            
            camera_frame = None
            with self.frame_lock:
                if (camera_idx in self.latest_frames and 
                    self.latest_frames[camera_idx] is not None):
                    camera_frame = cv2.resize(self.latest_frames[camera_idx], camera_display_size)
            
            if camera_frame is None:
                camera_frame = np.zeros((camera_display_size[1], camera_display_size[0], 3), dtype=np.uint8)
                cv2.putText(camera_frame, f"Camera {camera_idx}", (50, camera_display_size[1]//2 - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(camera_frame, "No Signal", (50, camera_display_size[1]//2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Add border
            border_color = (100, 100, 100)
            border_thickness = 3
            
            if camera_idx == self.main_display_camera:
                border_color = (255, 255, 0)
                border_thickness = 5
            
            if self.camera_states[camera_idx]['is_alarm_active']:
                border_color = (0, 0, 255)
                border_thickness = 8
            
            cv2.rectangle(camera_frame, (0, 0), 
                         (camera_display_size[0]-1, camera_display_size[1]-1), 
                         border_color, border_thickness)
            
            # Add camera info overlay with name
            info_bg = np.zeros((40, camera_display_size[0], 3), dtype=np.uint8)
            camera_frame[0:40, :] = cv2.addWeighted(camera_frame[0:40, :], 0.3, info_bg, 0.7, 0)
            
            # Get camera name
            camera_name = self.camera_states[camera_idx].get('camera_name', f'Camera {camera_idx}')
            
            # Truncate name if too long for display
            max_name_length = camera_display_size[0] // 12  # Approximate character width
            if len(camera_name) > max_name_length:
                display_name = camera_name[:max_name_length-3] + "..."
            else:
                display_name = camera_name
            
            cv2.putText(camera_frame, display_name, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            status_text = "ALARM" if self.camera_states[camera_idx]['is_alarm_active'] else "OK"
            status_color = (0, 0, 255) if self.camera_states[camera_idx]['is_alarm_active'] else (0, 255, 0)
            
            cv2.putText(camera_frame, status_text, (camera_display_size[0] - 80, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # Show 'P' for person detection enabled
            person_indicator = "P" if self.camera_states[camera_idx]['person_detection_enabled'] else "-"
            indicator_color = (0, 255, 0) if person_indicator == "P" else (0, 0, 255)
            cv2.putText(camera_frame, person_indicator, (10, camera_display_size[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, indicator_color, 1)
            
            grid_frame[start_y:end_y, start_x:end_x] = camera_frame
        
        # Add grid layout info
        info_text = f"Grid: {grid_cols}x{grid_rows} | Cameras: {num_cameras}/{grid_cols*grid_rows}"
        cv2.putText(grid_frame, info_text, (10, display_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add window focus indicator to grid
        focus_text = "MAIN GRID VIEW - Press '0' to focus | Press '`' for camera list"
        cv2.putText(grid_frame, focus_text, (10, display_height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return grid_frame
    
    def create_alert_display_frame(self, camera_idx):
        """Create the enlarged alert display frame for specific camera - HIGH QUALITY"""
        if camera_idx not in self.active_alert_windows:
            return None
            
        alert_frame = None
        original_resolution = None
        
        with self.frame_lock:
            if (camera_idx in self.latest_frames and 
                self.latest_frames[camera_idx] is not None):
                alert_frame = self.latest_frames[camera_idx].copy()
                original_resolution = (alert_frame.shape[1], alert_frame.shape[0])
        
        if alert_frame is None:
            alert_frame = np.zeros((self.alert_window_size[1], self.alert_window_size[0], 3), dtype=np.uint8)
            cv2.putText(alert_frame, f"Camera {camera_idx} - No Feed", 
                       (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # Only resize if original is different from alert window size
            if (alert_frame.shape[1], alert_frame.shape[0]) != self.alert_window_size:
                # Use high-quality interpolation for upscaling/downscaling
                alert_frame = cv2.resize(alert_frame, self.alert_window_size, interpolation=cv2.INTER_LANCZOS4)
        
        state = self.camera_states[camera_idx]
        if state['is_alarm_active']:
            border_thickness = 20
            current_time = time.time()
            if int(current_time * 4) % 2:
                cv2.rectangle(alert_frame, (0, 0), 
                             (self.alert_window_size[0], self.alert_window_size[1]), 
                             (0, 0, 255), border_thickness)
            
            alert_text = f"SECURITY ALERT - {state['camera_name'].upper()}"
            detection_type = "PERSON DETECTED CROSSING BOUNDARY"
            
            text_bg = np.zeros((140, self.alert_window_size[0], 3), dtype=np.uint8)
            text_bg[:, :] = (0, 0, 150)
            alert_frame[0:140, :] = cv2.addWeighted(alert_frame[0:140, :], 0.3, text_bg, 0.7, 0)
            
            cv2.putText(alert_frame, alert_text, (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(alert_frame, detection_type, (50, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
            
            # Show original resolution info
            if original_resolution:
                resolution_text = f"Source: {original_resolution[0]}x{original_resolution[1]}"
                cv2.putText(alert_frame, resolution_text, (50, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Show alert duration
            if state['alert_triggered_time']:
                duration = int(time.time() - state['alert_triggered_time'])
                duration_text = f"Alert Duration: {duration}s"
                cv2.putText(alert_frame, duration_text, (50, self.alert_window_size[1] - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(alert_frame, f"Time: {timestamp}", (50, self.alert_window_size[1] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Show other active alerts
            other_alerts = [idx for idx in self.active_alert_windows.keys() if idx != camera_idx]
            if other_alerts:
                other_alerts_text = f"Other Alerts: {', '.join(map(str, other_alerts))}"
                cv2.putText(alert_frame, other_alerts_text, (50, self.alert_window_size[1] - 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        return alert_frame
    
    def main_display_mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for grid display"""
        num_cameras = len(self.cameras)
        if num_cameras == 0:
            return
            
        # Calculate optimal grid layout
        grid_cols, grid_rows, camera_display_size = self.calculate_optimal_grid_layout(num_cameras)
        
        col = x // camera_display_size[0]
        row = y // camera_display_size[1]
        
        if col >= grid_cols or row >= grid_rows:
            return
            
        camera_index = row * grid_cols + col
        camera_list = list(self.cameras.keys())
        
        if camera_index >= len(camera_list):
            return
            
        clicked_camera = camera_list[camera_index]
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if clicked_camera != self.main_display_camera:
                self.switch_main_display(clicked_camera)
                return
        
        if clicked_camera != self.main_display_camera:
            return
            
        camera_col_idx = camera_list.index(clicked_camera)
        actual_col = camera_col_idx % grid_cols
        actual_row = camera_col_idx // grid_cols
        
        relative_x = x - (actual_col * camera_display_size[0])
        relative_y = y - (actual_row * camera_display_size[1])
        
        if relative_x < 0 or relative_x >= camera_display_size[0] or relative_y < 0 or relative_y >= camera_display_size[1]:
            return
        
        state = self.camera_states[clicked_camera]
        
        scale_x = state['frame_width'] / camera_display_size[0]
        scale_y = state['frame_height'] / camera_display_size[1]
        
        scaled_x = int(relative_x * scale_x)
        scaled_y = int(relative_y * scale_y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            state['is_drawing'] = True
            state['current_line'] = [(scaled_x, scaled_y)]
            
        elif event == cv2.EVENT_MOUSEMOVE and state['is_drawing']:
            state['current_line'].append((scaled_x, scaled_y))
            
        elif event == cv2.EVENT_LBUTTONUP:
            if state['is_drawing'] and len(state['current_line']) > 1:
                state['boundary_lines'].append(state['current_line'].copy())
                print(f"Camera {clicked_camera} - Boundary line added. Total lines: {len(state['boundary_lines'])}")
            
            state['is_drawing'] = False
            state['current_line'] = []
    
    def process_camera(self, camera_idx):
        """Process individual camera with enhanced error handling"""
        cap = self.cameras[camera_idx]
        state = self.camera_states[camera_idx]
        
        print(f"Camera {camera_idx} processing started")
        
        consecutive_failures = 0
        max_consecutive_failures = 5
        last_reconnect_attempt = 0
        reconnect_cooldown = 5

        frame_count = 0
        yolo_skip_frames = 3
        
        while True:
            try:
                current_time = time.time()
                
                # Read frame
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    consecutive_failures += 1
                    print(f"Camera {camera_idx} - Frame read failed ({consecutive_failures}/{max_consecutive_failures})")
                    
                    if (consecutive_failures >= max_consecutive_failures and 
                        current_time - last_reconnect_attempt > reconnect_cooldown):
                        print(f"Camera {camera_idx} - Attempting to reconnect...")
                        last_reconnect_attempt = current_time
                        
                        # Get original source
                        camera_source_idx = list(self.cameras.keys()).index(camera_idx)
                        source = self.camera_sources[camera_source_idx]
                        
                        # Close existing connection
                        cap.release()
                        time.sleep(2)
                        
                        # Try to reconnect with OpenCV
                        cap.release()
                        time.sleep(2)  # Wait before reconnecting
                        
                        # Get original source from camera config
                        camera_source_idx = list(self.cameras.keys()).index(camera_idx)
                        if camera_source_idx < len(self.camera_sources):
                            camera_config = self.camera_sources[camera_source_idx]
                            if isinstance(camera_config, dict):
                                source = camera_config.get('url', 0)
                            else:
                                source = camera_config
                            
                            # Recreate capture
                            new_cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                            if new_cap.isOpened():
                                new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                new_cap.set(cv2.CAP_PROP_FPS, 30)
                                cap = new_cap
                                self.cameras[camera_idx] = cap
                                consecutive_failures = 0
                                print(f"Camera {camera_idx} - Reconnection successful")
                                continue
                    
                    time.sleep(0.1)
                    continue
                
                # Reset failure counter on successful read
                consecutive_failures = 0
                state['last_successful_read'] = current_time
                
                # Process frame - only person detection, no motion detection
                frame = frame.copy()

                persons = []
                if frame_count % yolo_skip_frames == 0:
                    persons = self.detect_persons(frame, camera_idx)
                else:
                    persons = state.get('detected_persons', [])
                
                # Check boundary crossing only with person detection
                if persons and len(state['boundary_lines']) > 0:
                    self.check_boundary_crossing(persons, camera_idx)
                
                frame = self.draw_overlays(frame, persons, camera_idx, 
                                        camera_idx == self.main_display_camera)
                
                with self.frame_lock:
                    self.latest_frames[camera_idx] = frame.copy()
                
                if camera_idx in self.active_alert_windows:
                    with self.alert_lock:
                        self.alert_frames[camera_idx] = frame.copy()
                
                # time.sleep(0.03)  # ~30 FPS limit
                
            except Exception as e:
                print(f"Camera {camera_idx} error: {e}")
                time.sleep(1)
                continue
    
    def display_thread(self):
        """Main display thread for grid view and multiple alert windows"""
        cv2.namedWindow(self.main_window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.main_window_name, self.main_display_mouse_callback)
        
        if self.cameras:
            self.main_display_camera = list(self.cameras.keys())[0]
        
        try:
            while True:
                # Display main grid
                grid_frame = self.create_grid_display_frame()
                cv2.imshow(self.main_window_name, grid_frame)
                
                # Display all active alert windows in priority order (most recent first)
                with self.alert_lock:
                    sorted_alerts = sorted(self.active_alert_windows.keys(), 
                                         key=lambda x: self.alert_window_z_order.get(x, 0), 
                                         reverse=True)
                    
                    # Handle window recreation for focus
                    windows_to_recreate = self.windows_to_recreate.copy()
                    self.windows_to_recreate.clear()
                
                # Process windows that need recreation (outside the lock)
                for camera_idx in windows_to_recreate:
                    if camera_idx in self.active_alert_windows:
                        state = self.camera_states[camera_idx]
                        alert_window_name = state['alert_window_name']
                        current_pos = self.alert_window_positions.get(camera_idx, (50, 50))
                        
                        try:
                            # Safely destroy and recreate window
                            cv2.destroyWindow(alert_window_name)
                            time.sleep(0.05)  # Wait for destruction
                            
                            # Recreate window
                            cv2.namedWindow(alert_window_name, cv2.WINDOW_AUTOSIZE)
                            try:
                                cv2.moveWindow(alert_window_name, current_pos[0], current_pos[1])
                            except:
                                pass
                            
                            print(f"Window recreated and brought to front for Camera {camera_idx}")
                            
                        except Exception as e:
                            print(f"Error recreating window for Camera {camera_idx}: {e}")
                
                # Display all alert windows
                for camera_idx in sorted_alerts:
                    if camera_idx in self.active_alert_windows:  # Double check it still exists
                        alert_frame = self.create_alert_display_frame(camera_idx)
                        if alert_frame is not None:
                            state = self.camera_states[camera_idx]
                            alert_window_name = state['alert_window_name']
                            
                            # Create window if it doesn't exist
                            window_exists = True
                            try:
                                cv2.getWindowProperty(alert_window_name, cv2.WND_PROP_VISIBLE)
                            except cv2.error:
                                window_exists = False
                                cv2.namedWindow(alert_window_name, cv2.WINDOW_AUTOSIZE)
                                # Set window position if available
                                if camera_idx in self.alert_window_positions:
                                    x, y = self.alert_window_positions[camera_idx]
                                    try:
                                        cv2.moveWindow(alert_window_name, x, y)
                                    except:
                                        pass
                            
                            # Always update the display
                            cv2.imshow(alert_window_name, alert_frame)
                            
                            # If this is the most recent window (highest priority), try to keep it on top
                            if camera_idx == sorted_alerts[0] and len(sorted_alerts) > 1:
                                try:
                                    # Safe topmost setting
                                    cv2.setWindowProperty(alert_window_name, cv2.WND_PROP_TOPMOST, 1)
                                    time.sleep(0.001)  # Very small delay
                                    cv2.setWindowProperty(alert_window_name, cv2.WND_PROP_TOPMOST, 0)
                                except:
                                    pass
                
                # Clean up closed alert windows and handle auto-focus shift
                windows_to_remove = []
                for camera_idx in sorted_alerts:
                    state = self.camera_states[camera_idx]
                    alert_window_name = state['alert_window_name']
                    try:
                        if cv2.getWindowProperty(alert_window_name, cv2.WND_PROP_VISIBLE) < 0:
                            windows_to_remove.append(camera_idx)
                    except cv2.error:
                        windows_to_remove.append(camera_idx)
                
                # Handle automatic focus shifting when windows are closed
                focus_shifted = False
                for camera_idx in windows_to_remove:
                    with self.alert_lock:
                        if camera_idx in self.active_alert_windows:
                            print(f"🔄 Alert window for Camera {camera_idx} was closed")
                            
                            # Check if this was the most recent (focused) window
                            current_sorted = sorted(self.active_alert_windows.keys(), 
                                                  key=lambda x: self.alert_window_z_order.get(x, 0), 
                                                  reverse=True)
                            was_focused_window = len(current_sorted) > 0 and current_sorted[0] == camera_idx
                            
                            # Remove from tracking
                            del self.active_alert_windows[camera_idx]
                            if camera_idx in self.alert_window_positions:
                                del self.alert_window_positions[camera_idx]
                            if camera_idx in self.alert_frames:
                                del self.alert_frames[camera_idx]
                            if camera_idx in self.alert_window_z_order:
                                del self.alert_window_z_order[camera_idx]
                            self.windows_to_recreate.discard(camera_idx)
                            
                            # Auto-shift focus to next alert window if the focused one was closed
                            if was_focused_window and not focus_shifted:
                                remaining_alerts = list(self.active_alert_windows.keys())
                                if remaining_alerts:
                                    # Sort by z-order to get the next most recent
                                    next_sorted = sorted(remaining_alerts, 
                                                       key=lambda x: self.alert_window_z_order.get(x, 0), 
                                                       reverse=True)
                                    next_camera = next_sorted[0]
                                    
                                    # Bring next window to front
                                    self.alert_window_z_order[next_camera] = time.time()
                                    self.windows_to_recreate.add(next_camera)
                                    
                                    print(f"🎯 Auto-focusing on next alert: Camera {next_camera}")
                                    focus_shifted = True
                                else:
                                    # No more alert windows, focus on main grid
                                    try:
                                        cv2.setWindowProperty(self.main_window_name, cv2.WND_PROP_TOPMOST, 1)
                                        time.sleep(0.01)
                                        cv2.setWindowProperty(self.main_window_name, cv2.WND_PROP_TOPMOST, 0)
                                        print("🏠 Auto-focusing on main grid window (no more alerts)")
                                    except:
                                        pass
                                    focus_shifted = True
                
                key = cv2.waitKey(30) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.reset_alarm()  # Reset all alarms
                elif key == ord('c') and self.main_display_camera is not None:
                    self.clear_boundaries(self.main_display_camera)
                elif key == ord('p') and self.main_display_camera is not None:
                    self.toggle_person_detection(self.main_display_camera)
                elif key == ord('r'):  # Reset specific camera alarm
                    if self.main_display_camera is not None:
                        self.reset_alarm(self.main_display_camera)
                elif key == ord('0'):
                    # Focus on main grid window
                    try:
                        # Try to bring main window to front
                        if cv2.getWindowProperty(self.main_window_name, cv2.WND_PROP_VISIBLE) >= 0:
                            cv2.setWindowProperty(self.main_window_name, cv2.WND_PROP_TOPMOST, 1)
                            time.sleep(0.01)
                            cv2.setWindowProperty(self.main_window_name, cv2.WND_PROP_TOPMOST, 0)
                            print("Main grid window brought to front")
                        else:
                            print("Main window not visible")
                    except Exception as e:
                        print(f"Error bringing main window to front: {e}")
                        
                elif key >= ord('1') and key <= ord('9'):
                    camera_num = key - ord('1')
                    camera_list = list(self.cameras.keys())
                    if camera_num < len(camera_list):
                        selected_camera = camera_list[camera_num]
                        self.switch_main_display(selected_camera)
                        print(f"Switched to Camera {camera_num + 1} ({selected_camera})")
                    else:
                        print(f"Camera {camera_num + 1} does not exist. Available cameras: 1-{len(camera_list)}")
                        
                elif key == 96:  # backtick key code
                    # Show camera list and instructions for cameras beyond 9
                    camera_list = list(self.cameras.keys())
                    print(f"\n=== Camera List ({len(camera_list)} total) ===")
                    print("Press '0' for main grid window")
                    for i, cam_id in enumerate(camera_list):
                        status = "ALARM" if self.camera_states[cam_id]['is_alarm_active'] else "OK"
                        alert_status = "ALERT WINDOW OPEN" if cam_id in self.active_alert_windows else "No alert"
                        print(f"Press '{i + 1}' for Camera {cam_id} - Status: {status} - {alert_status}")
                    if len(camera_list) > 9:
                        print(f"Note: Cameras beyond 9 require manual selection or additional key mapping")
                    print("=" * 50)
                        
        except Exception as e:
            print(f"Display thread error: {e}")
        finally:
            cv2.destroyWindow(self.main_window_name)
            self.close_all_alert_windows()
    
    def print_controls(self):
        """Print control instructions"""
        print("Multi-Camera Intruder Detection System with Named Cameras")
        print("Grid View Controls:")
        print("- Press '0' to focus on MAIN GRID WINDOW")
        print("- Press '1'-'9' to focus on individual camera alert windows")
        print("- Press '`' (backtick) to show camera list and status")
        print("- Click on any camera view to select it for boundary drawing")
        print("- Click and drag on selected camera to draw boundaries")  
        print("- Press 's' to stop ALL alarms and reset entire system")
        print("- Press 'r' to reset alarm for SELECTED camera only")
        print("- Press 'c' to clear boundaries for selected camera")
        print("- Press 'p' to toggle person detection for selected camera")
        print("- Press 'q' to quit")
        print("System uses ONLY person detection - no motion detection")
        print("Alert triggers when person bounding box touches boundary line")
        print(f"Available cameras: {len(self.camera_sources)} total")
    
    def run(self):
        """Main program - run all cameras and display"""
        if not self.cameras:
            print("No cameras available!")
            return
        
        # Start camera processing threads
        camera_threads = []
        for camera_idx in self.cameras.keys():
            thread = threading.Thread(target=self.process_camera, args=(camera_idx,), daemon=True)
            thread.start()
            camera_threads.append(thread)
            time.sleep(0.5)
        
        # Start display thread
        display_thread = threading.Thread(target=self.display_thread, daemon=True)
        display_thread.start()
        
        try:
            display_thread.join()
        except KeyboardInterrupt:
            print("\nSystem interrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.global_alarm_active = False
        
        # Close all alert windows
        self.close_all_alert_windows()
        
        for camera_idx, state in self.camera_states.items():
            state['is_alarm_active'] = False
        
        if self.alarm_sound:
            self.alarm_sound.stop()
        
        for cap in self.cameras.values():
            cap.release()
        
        cv2.destroyAllWindows()
        print("Simplified intruder detection system cleaned up. Goodbye!")

def main():
    """Main function to run the simplified intruder detection system"""
    try:
        # Camera configuration - UPDATED FORMAT WITH NAMES
        camera_sources = [
    {
        "name": "ALPHA TECHNO SQUARE - 2",
        "url": "rtsp://admin:pakistan%40123@10.115.50.163:554/ch1/main/av_stream"
    },
    {
        "name": "ALPHA TECHNO SQUARE - CAFE",
        "url": "rtsp://admin:pakistan%40123@10.115.50.158:554/ch1/main/av_stream"
    },
    {
        "name": "A11 GATE VIEW",
        "url": "rtsp://admin:pakistan%40123@10.115.50.161:554/ch1/main/av_stream"
    },
    {
        "name": "SCAN ROOM - 2",
        "url": "rtsp://admin:pakistan%40123@10.115.50.165:554/ch1/main/av_stream"
    },
    {
        "name": "CAM - 5",
        "url": "rtsp://admin:pakistan%40123@10.115.50.161:554/ch1/main/av_stream"
    },
    {
        "name": "CAM - 6",
        "url": "rtsp://admin:pakistan%40123@10.115.50.159:554/ch1/main/av_stream"
    },
    {
        "name": "CAM - 7",
        "url": "rtsp://admin:pakistan%40123@10.115.50.151:554/ch1/main/av_stream"
    },
    {
        "name": "CAM - 8",
        "url": "rtsp://admin:pakistan%40123@10.115.50.152:554/ch1/main/av_stream"
    },
    {
        "name": "CAM - 9",
        "url": "rtsp://admin:pakistan%40123@10.115.50.162:554/ch1/main/av_stream"
    },
    {
        "name": "CAM - 10",
        "url": "rtsp://admin:pakistan%40123@10.115.50.153:554/ch1/main/av_stream"
    },
    {
        "name": "CAM - 11",
        "url": "rtsp://admin:pakistan%40123@10.115.50.157:554/ch1/main/av_stream"
    },
    {
        "name": "CAM - 12",
        "url": "rtsp://admin:pakistan%40123@10.115.50.156:554/ch1/main/av_stream"
    }
        ]
        
        # Initialize system with OpenCV only
        system = MultipleCameraBoundaryAlarmSystem(
            camera_sources=camera_sources
        )
        
        print("\n" + "="*80)
        print("MULTI-CAMERA INTRUDER DETECTION SYSTEM WITH NAMED CAMERAS")
        print("PERSON DETECTION ONLY - OPTIMIZED OPENCV")
        print("="*80)
        
        system.run()
        
    except Exception as e:
        print(f"Error starting simplified intruder detection system: {e}")

if __name__ == "__main__":
    main()