import cv2
import time
import sys
import os
import queue
import json
from mp_handler import MediaPipeThread

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from fusion.fusion import fuse

FACE_AVAILABLE = False
GESTURE_AVAILABLE = False
RPPG_AVAILABLE = False

try:
    from face_module.face_detection import FaceAgent
    FACE_AVAILABLE = True
except:
    pass

try:
    from gesture_module.gesture import gesture_agent
    GESTURE_AVAILABLE = True
except:
    pass

try:
    from rPPG_module.rppg import rPPG_agent
    RPPG_AVAILABLE = True
except:
    pass

def main():
    cap = cv2.VideoCapture(0)
    mp_q = queue.Queue(maxsize=2)
    mp_thread = MediaPipeThread(mp_q)
    mp_thread.start()

    face = FaceAgent() if FACE_AVAILABLE else None
    if face:
        face.start()
    gesture = gesture_agent() if GESTURE_AVAILABLE else None
    if gesture:
        gesture.start()
    rppg = rPPG_agent() if RPPG_AVAILABLE else None
    if rppg:
        rppg.start()

    fusion_result = None
    last = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts = int(time.time()*1000)
        mp_thread.enqueue_frame(frame, ts)

        try:
            img, face_res, hand_res = mp_q.get_nowait()
        except:
            img, face_res, hand_res = None, None, None

        if face_res and face:
            face.enqueue_frame(face_res, img)

        if face_res and hand_res and gesture:
            gesture.enqueue_frame({
                "hand_landmarks": hand_res.hand_landmarks or [],
                "face_landmarks": face_res.face_landmarks[0] if face_res.face_landmarks else None
            })

        if face_res and img is not None and rppg:
            rppg.enqueue_frame(img.copy(), face_res.face_landmarks)

        fr = getattr(face, "latest_result", None)
        gr = getattr(gesture, "latest_result", None)
        rr = getattr(rppg, "latest_result", None)

        if time.time() - last > 0.5:
            fusion_result = fuse(fr, gr, rr)
            print(json.dumps(fusion_result, indent=2))
            last = time.time()

        draw = frame.copy()
        h, w = draw.shape[:2]

        if fusion_result:
            cv2.rectangle(draw, (10, 10), (w-10, 110), (30,30,30), -1)

            cv2.putText(draw,
                f"State: {fusion_result['cognitive_state']} | Emotion: {fusion_result['emotional_state']}",
                (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            cv2.putText(draw,
                f"Stress: {fusion_result['stress_level']} | Engagement: {fusion_result['engagement']}",
                (20,65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.putText(draw,
                f"Confidence: {fusion_result['confidence']}",
                (20,95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2)

            cv2.putText(draw,
                fusion_result["message"],
                (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Flow AI", draw)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()