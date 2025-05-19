import cv2
import mediapipe as mp
import pyautogui
import math
import time  # Needed for right-click cooldown

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    static_image_mode=False,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.8,
)
mp_draw = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

aspect_ratio = 16 / 9
scale_factor = 0.8

if frame_width / frame_height > aspect_ratio:
    box_height = int((frame_height - 100) * scale_factor)
    box_width = int(box_height * aspect_ratio)
else:
    box_width = int((frame_width - 100) * scale_factor)
    box_height = int(box_width / aspect_ratio)

boundary_left = (frame_width - box_width) // 2
boundary_top = (frame_height - box_height) // 2
boundary_right = boundary_left + box_width
boundary_bottom = boundary_top + box_height

prev_x, prev_y = 0, 0
smoothing = 0.4
movement_threshold = 7

click_threshold = 0.1
clicking = False

# Right click tracking
right_click_threshold = 0.1  # adjust as needed
last_right_click_time = 0
right_click_cooldown = 2.0  # seconds
show_right_click_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Draw boundary rectangle
    cv2.rectangle(frame, (boundary_left, boundary_top), (boundary_right, boundary_bottom), (0, 255, 0), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_norm = hand_landmarks.landmark[8].x
            y_norm = hand_landmarks.landmark[8].y

            thumb_x = hand_landmarks.landmark[4].x
            thumb_y = hand_landmarks.landmark[4].y

            pinky_x = hand_landmarks.landmark[17].x
            pinky_y = hand_landmarks.landmark[17].y

            middle_x = hand_landmarks.landmark[12].x
            middle_y = hand_landmarks.landmark[12].y

            cx = int(x_norm * frame_width)
            cy = int(y_norm * frame_height)

            cx = max(boundary_left, min(boundary_right, cx))
            cy = max(boundary_top, min(boundary_bottom, cy))

            screen_x = (cx - boundary_left) / (boundary_right - boundary_left) * screen_width
            screen_y = (cy - boundary_top) / (boundary_bottom - boundary_top) * screen_height

            dx = screen_x - prev_x
            dy = screen_y - prev_y
            distance = math.hypot(dx, dy)

            if distance > movement_threshold:
                curr_x = prev_x + (dx * smoothing)
                curr_y = prev_y + (dy * smoothing)
                pyautogui.moveTo(curr_x, curr_y, duration=0)
                prev_x, prev_y = curr_x, curr_y

            # Click logic using thumb tip and pinky base
            thumb_pinky_dist = math.hypot(thumb_x - pinky_x, thumb_y - pinky_y)

            if thumb_pinky_dist < click_threshold:
                if not clicking:
                    pyautogui.mouseDown()
                    clicking = True
            else:
                if clicking:
                    pyautogui.mouseUp()
                    clicking = False

            # Right-click logic
            index_middle_dist = math.hypot(x_norm - middle_x, y_norm - middle_y)
            current_time = time.time()
            if index_middle_dist > right_click_threshold:
                movement_threshold = 2500 #remember to revert this back
                if current_time - last_right_click_time > right_click_cooldown:
                    pyautogui.rightClick()
                    last_right_click_time = current_time
                    show_right_click_time = current_time
                movement_threshold = 7    


            # Show right click indicator for a short duration
            if current_time - show_right_click_time < 0.4:
                cv2.putText(frame, "Right Click!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Show click status
            status_text = "Clicking" if clicking else "Not Clicking"
            color = (0, 0, 255) if clicking else (0, 255, 0)
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    else:
        if clicking:
            pyautogui.mouseUp()
            clicking = False
        cv2.putText(frame, "No Hand", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("MediaPipe Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
