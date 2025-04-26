import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from math import hypot

# Initialize MediaPipe Hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam with higher resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Get screen size and setup parameters
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
FRAME_REDUCTION = 150
SMOOTHING = 0.7
MIN_MOVEMENT_THRESHOLD = 3
SPEED_SENSITIVITY = 1.5

# Previous cursor position for smoothing
prev_x, prev_y = pyautogui.position()

# Initialize variables for gesture detection
pyautogui.PAUSE = 0.01
pyautogui.FAILSAFE = True

# Gesture state tracking
last_gesture = "none"
last_action_time = time.time()
GESTURE_COOLDOWN = 0.2

# Drag and drop state tracking
is_dragging = False
drag_start_pos = None
drag_threshold = 5  # Minimum movement required to start drag
drag_cooldown = 0.5  # Time to wait before next drag operation
last_drag_time = 0

# Scroll state tracking
last_scroll_y = None
SCROLL_SENSITIVITY = 30  # Adjust this value to change scroll speed

def calculate_distance(p1, p2):
    """Calculate distance between two points"""
    return hypot(p1.x - p2.x, p1.y - p2.y)

def get_finger_state(hand_landmarks):
    """Determine which fingers are up"""
    fingers = []
    
    # Thumb detection
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    fingers.append(thumb_tip.x < thumb_ip.x)
    
    # Other fingers
    tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    pips = [mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.PINKY_PIP]
    
    for tip, pip in zip(tips, pips):
        tip_point = hand_landmarks.landmark[tip]
        pip_point = hand_landmarks.landmark[pip]
        fingers.append(tip_point.y < pip_point.y)
    
    return fingers

def smooth_movement(current_x, current_y, prev_x, prev_y, smoothing):
    """Apply smoothing to cursor movement"""
    new_x = int(prev_x + (current_x - prev_x) * (1 - smoothing))
    new_y = int(prev_y + (current_y - prev_y) * (1 - smoothing))
    return new_x, new_y

def detect_gestures(fingers, hand_landmarks):
    """Detect gestures based on finger positions"""
    global last_gesture, last_action_time, is_dragging, drag_start_pos, last_scroll_y, last_drag_time
    
    current_time = time.time()
    if current_time - last_action_time < GESTURE_COOLDOWN:
        return last_gesture

    # Count closed fingers (not up)
    closed_fingers = sum(1 for finger in fingers if not finger)
    
    # Get palm position for stability check
    palm_pos = calculate_palm_center(hand_landmarks)
    
    # Basic gesture detection
    if closed_fingers >= 4 and current_time - last_drag_time > drag_cooldown:  # Closed fist with cooldown
        if not is_dragging:
            # Start new drag operation
            is_dragging = True
            drag_start_pos = palm_pos
            last_drag_time = current_time
        gesture = "drag"
    elif fingers[1] and fingers[2] and fingers[3] and fingers[4]:  # All four fingers up
        gesture = "scroll"
    elif fingers[1] and fingers[2]:  # Index and Middle up
        gesture = "move"
    elif fingers[1] and not fingers[2]:  # Only Index up
        gesture = "left_click"
    elif not fingers[1] and fingers[2]:  # Only Middle up
        gesture = "right_click"
    elif fingers[3]:  # Ring finger up
        gesture = "double_click"
    else:
        gesture = "none"
        if is_dragging:
            # Release drag when fist opens
            pyautogui.mouseUp()
            is_dragging = False
            drag_start_pos = None
            last_drag_time = current_time
        last_scroll_y = None  # Reset scroll tracking when gesture ends

    if gesture != last_gesture:
        last_action_time = current_time
        last_gesture = gesture
    
    return gesture

# Disable pyautogui fail-safe
pyautogui.FAILSAFE = False

# Initialize MediaPipe Hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam with higher resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Higher resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Higher resolution
cap.set(cv2.CAP_PROP_FPS, 30)  # Ensure good framerate

# Get screen size
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
# Set up the frame window for hand movement
FRAME_REDUCTION = 150  # Increased frame reduction for more precise control
SMOOTHING = 0.7  # Increased smoothing for stability

# Previous cursor position for smoothing
prev_x, prev_y = pyautogui.position()
prev_palm_x, prev_palm_y = 0, 0

# Movement control parameters
MIN_MOVEMENT_THRESHOLD = 3  # Minimum pixels to move
PALM_SMOOTHING = 0.8  # Smoothing factor for palm movement
SPEED_SENSITIVITY = 1.5  # Adjust movement sensitivity

# Initialize variables for gesture detection
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01

# Gesture state tracking
click_start_time = 0
is_clicking = False
last_gesture = "none"
last_action_time = time.time()
GESTURE_COOLDOWN = 0.2  # Reduced cooldown for more responsive gestures

def calculate_distance(p1, p2):
    """Calculate distance between two points"""
    return hypot(p1.x - p2.x, p1.y - p2.y)

def calculate_palm_center(hand_landmarks):
    """Calculate the center of the palm using wrist and finger base points"""
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    mcp_points = [
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP],
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    ]
    
    palm_x = sum([point.x for point in mcp_points]) / 4
    palm_y = sum([point.y for point in mcp_points]) / 4
    
    return palm_x, palm_y

def get_finger_state(hand_landmarks):
    """Determine which fingers are up and detect closed fist"""
    fingers = []
    
    # Thumb detection
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    
    # Check if thumb is closed (closer to palm)
    thumb_closed = calculate_distance(thumb_tip, thumb_mcp) < calculate_distance(thumb_ip, thumb_mcp)
    fingers.append(not thumb_closed)
    
    # Other fingers
    tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    pips = [mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.PINKY_PIP]
    mcps = [mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
            mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.PINKY_MCP]
    
    for tip, pip, mcp in zip(tips, pips, mcps):
        tip_point = hand_landmarks.landmark[tip]
        pip_point = hand_landmarks.landmark[pip]
        mcp_point = hand_landmarks.landmark[mcp]
        
        # Check if finger is closed (tip closer to palm than PIP)
        finger_closed = calculate_distance(tip_point, mcp_point) < calculate_distance(pip_point, mcp_point)
        fingers.append(not finger_closed)
    
    return fingers

def smooth_movement(current_x, current_y, prev_x, prev_y, smoothing):
    """Apply enhanced smoothing to cursor movement"""
    # Calculate movement delta
    dx = current_x - prev_x
    dy = current_y - prev_y
    
    # Calculate movement speed
    speed = np.sqrt(dx*dx + dy*dy)
    
    # Apply speed-based sensitivity
    if speed < MIN_MOVEMENT_THRESHOLD:
        return prev_x, prev_y
    
    # Dynamic smoothing based on speed
    dynamic_smoothing = smoothing * (1 - min(1, speed/500))
    
    # Apply non-linear smoothing
    movement_scale = min(1.0, speed * SPEED_SENSITIVITY / 100)
    new_x = int(prev_x + dx * movement_scale * (1 - dynamic_smoothing))
    new_y = int(prev_y + dy * movement_scale * (1 - dynamic_smoothing))
    
    return new_x, new_y

def detect_gestures(fingers, hand_landmarks):
    """
    Detect gestures based on finger positions:
    - Index + Middle up: Move cursor
    - Index up only: Left click
    - Middle up only: Right click
    - Ring finger up: Double click
    - Pinky up: Scroll mode
    """
    global last_gesture, last_action_time
    
    current_time = time.time()
    if current_time - last_action_time < GESTURE_COOLDOWN:
        return last_gesture

    # Basic gesture detection
    if fingers[1] and fingers[2]:  # Index and Middle up
        gesture = "move"
    elif fingers[1] and not fingers[2]:  # Only Index up
        gesture = "left_click"
    elif not fingers[1] and fingers[2]:  # Only Middle up
        gesture = "right_click"
    elif fingers[3]:  # Ring finger up
        gesture = "double_click"
    elif fingers[4]:  # Pinky up
        gesture = "scroll"
    else:
        gesture = "none"

    if gesture != last_gesture:
        last_action_time = current_time
        last_gesture = gesture
    
    return gesture

# Main loop
try:
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Failed to get frame from camera")
                break
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Get finger states
                    fingers = get_finger_state(hand_landmarks)
                    gesture = detect_gestures(fingers, hand_landmarks)
                    
                    if gesture == "scroll":
                        # Get hand position for scrolling
                        palm_x, palm_y = calculate_palm_center(hand_landmarks)
                        
                        if last_scroll_y is not None:
                            # Calculate scroll direction and amount
                            scroll_diff = (palm_y - last_scroll_y) * SCROLL_SENSITIVITY
                            if abs(scroll_diff) > MIN_MOVEMENT_THRESHOLD:
                                # Negative scroll_diff means scroll up, positive means scroll down
                                pyautogui.scroll(int(-scroll_diff))
                        
                        last_scroll_y = palm_y
                    
                    elif gesture in ["move", "drag"]:
                        try:
                            # Map palm position to screen coordinates
                            palm_x, palm_y = calculate_palm_center(hand_landmarks)
                            
                            # Map hand position to screen coordinates with reduced frame
                            screen_x = np.interp(palm_x, (FRAME_REDUCTION/cap.get(3), 1-FRAME_REDUCTION/cap.get(3)), (0, SCREEN_WIDTH))
                            screen_y = np.interp(palm_y, (FRAME_REDUCTION/cap.get(4), 1-FRAME_REDUCTION/cap.get(4)), (0, SCREEN_HEIGHT))
                            
                            # Apply smoothing
                            current_x, current_y = smooth_movement(screen_x, screen_y, prev_x, prev_y, SMOOTHING)
                            
                            # Move cursor if movement is significant
                            if abs(current_x - prev_x) > MIN_MOVEMENT_THRESHOLD or abs(current_y - prev_y) > MIN_MOVEMENT_THRESHOLD:
                                pyautogui.moveTo(current_x, current_y)
                                prev_x, prev_y = current_x, current_y

                                # Handle drag operation
                                if gesture == "drag" and not is_dragging:
                                    # Only start drag if we've moved past threshold
                                    if drag_start_pos is None or \
                                       hypot(current_x - drag_start_pos[0], current_y - drag_start_pos[1]) > drag_threshold:
                                        pyautogui.mouseDown()
                                        is_dragging = True
                                        print("Started dragging")
                        except Exception as e:
                            print(f"Error in drag operation: {e}")
                    
                    elif gesture == "left_click":
                        try:
                            pyautogui.click()
                            time.sleep(0.2)
                        except Exception as e:
                            print(f"Error performing left click: {e}")
                    
                    elif gesture == "right_click":
                        try:
                            pyautogui.rightClick()
                            time.sleep(0.2)
                        except Exception as e:
                            print(f"Error performing right click: {e}")
                    
                    elif gesture == "double_click":
                        try:
                            pyautogui.doubleClick()
                            time.sleep(0.3)
                        except Exception as e:
                            print(f"Error performing double click: {e}")
                    
                    elif gesture == "scroll":
                        # Use index finger position for scrolling
                        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
                        
                        # Calculate scroll amount based on finger position
                        # Moving finger up scrolls up, moving down scrolls down
                        scroll_amount = int((index_pip.y - index_tip.y) * 300)  # Increased sensitivity
                        
                        # Apply threshold to prevent accidental scrolling
                        if abs(scroll_amount) > 5:
                            pyautogui.scroll(scroll_amount)
                            time.sleep(0.05)  # Small delay to prevent too rapid scrolling
                    
                    # Display current gesture and finger states
                    cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Display finger controls guide
                    cv2.putText(frame, "Controls:", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(frame, "Index + Middle: Move", (10, 95),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(frame, "Index only: Left Click", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(frame, "Middle only: Right Click", (10, 145),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(frame, "Ring: Double Click", (10, 170),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(frame, "Pinky: Scroll Mode", (10, 195),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow('Hand Gesture Control', frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
                break
                
        except Exception as e:
            print(f"Error in main loop: {e}")
            continue

except KeyboardInterrupt:
    print("Program interrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
