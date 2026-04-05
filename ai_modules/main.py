import cv2
import time
from rPPG_module.rPPG import rPPG_agent
# from face_module import face_detection
# from gesture_module import gesture


def main():
    # rppg framing variables
    target_fps = 25
    frame_interval = 1.0 / target_fps
    last_sent_time = 0

    cap = cv2.VideoCapture(0)

    # initializing model objects
    # agent_gesture = gesture_agent()
    # agent_face = face_agent()
    agent_rppg = rPPG_agent()

    # starting the thread-based objects
    agent_rppg.start()

    while True:
        ret, frame = cap.read()
        if not ret: 
            break

        # time-based sampling for rppg model
        current_time = time.perf_counter()

        if (current_time - last_sent_time) >= frame_interval:
            agent_rppg.enqueue_frame(frame)
            last_sent_time = current_time

        rPPG_result = agent_rppg.latest_result

        cv2.putText(
            frame, f"Face: {rPPG_result}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2
        )
        
        cv2.imshow('Main Stream', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()