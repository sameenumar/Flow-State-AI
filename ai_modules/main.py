import cv2
import time
from mp_handler import MediaPipeHandler
from rPPG_module.rPPG import rPPG_agent
# from face_module import face_detection
# from gesture_module import gesture


def main():
    # rppg framing variables
    target_fps = 25
    frame_interval = 1.0 / target_fps
    last_sent_time = 0

    cap = cv2.VideoCapture(0)
    mp_agent = MediaPipeHandler()

    # initializing model objects
    # agent_gesture = gesture_agent()
    # agent_face = face_agent()
    agent_rppg = rPPG_agent()

    # starting the thread-based objects
    agent_rppg.start()

    while True:
        ret, bgr_frame = cap.read()
        if not ret: break

        # converting the frame 
        # from bgr format -> opencv
        # to rgb format -> mediapipe
        rgb_frame = mp_agent.prepare(bgr_frame)
        
        # process_face() & process_hands() expect
        # timestamp in milliseconds as args
        current_time = time.perf_counter()
        current_time = int(current_time * 1000)

        # running face and hand detection on the frame
        # final results to use as inputs to processing modules
        face_results = mp_agent.process_face(rgb_frame, current_time)
        hand_results = mp_agent.process_hands(rgb_frame, current_time)

        # time-based sampling for rppg model
        current_time = time.perf_counter()

        if (current_time - last_sent_time) >= frame_interval:
            agent_rppg.enqueue_frame(rgb_frame)
            last_sent_time = current_time

        rPPG_result = agent_rppg.latest_result

        if face_results.face_landmarks:
            for face_landmarks in face_results.face_landmarks:
                for lm in face_landmarks:
                    x = int(lm.x * bgr_frame.shape[1])
                    y = int(lm.y * bgr_frame.shape[0])
                    cv2.circle(bgr_frame, (x, y), 1, (0, 255, 0), -1)

        if hand_results.hand_landmarks:
            for hand_landmarks in hand_results.hand_landmarks:
                for lm in hand_landmarks:
                    x = int(lm.x * bgr_frame.shape[1])
                    y = int(lm.y * bgr_frame.shape[0])
                    cv2.circle(bgr_frame, (x, y), 1, (0, 255, 0), -1)

        cv2.putText(
            bgr_frame, f"Face: {rPPG_result}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2
        )
        
        cv2.imshow('Main Stream', bgr_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()