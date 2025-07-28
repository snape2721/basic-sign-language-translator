import cv2
import mediapipe as mp
import pyttsx3

# Initialize MediaPipe and TTS
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
engine = pyttsx3.init()
spoken = None  # To prevent repeating the same word

def get_finger_states(landmarks):
    tips = [4, 8, 12, 16, 20]
    pip_joints = [2, 6, 10, 14, 18]
    finger_states = []

    for tip, pip in zip(tips, pip_joints):
        if landmarks[tip].y < landmarks[pip].y:
            finger_states.append(1)  # Open
        else:
            finger_states.append(0)  # Closed
    return finger_states

def detect_gesture(states, landmarks):
    thumb, index, middle, ring, pinky = states

    # hello: all fingers open
    if states == [1, 1, 1, 1, 1]:
        return "hello"
    
    # yes: all fingers closed
    if states == [0, 0, 0, 0, 0]:
        return "yes"
    
    # no: thumb, index, middle open (hand horizontal)
    if states == [1, 1, 1, 0, 0]:
        y_diff = abs(landmarks[5].y - landmarks[17].y)
        x_diff = abs(landmarks[5].x - landmarks[17].x)
        if x_diff > y_diff:  # Horizontal check
            return "no"

    # thanks: all fingers open + hand near mouth (check y position of wrist)
    if states == [1, 1, 1, 1, 1]:
        if landmarks[0].y < 0.5:  # Wrist above mid-frame (near mouth)
            return "thanks"

    # I love you: thumb, index, pinky open
    if states == [1, 1, 0, 0, 1]:
        return "I love you"

    return None

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            finger_states = get_finger_states(landmarks)
            gesture = detect_gesture(finger_states, landmarks)

            if gesture and gesture != spoken:
                spoken = gesture
                print(f"Detected: {gesture}")
                engine.say(gesture)
                engine.runAndWait()

            if gesture:
                cv2.putText(frame, f"{gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

    cv2.imshow("Real-Time Sign Language Translator", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
