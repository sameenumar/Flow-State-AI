import queue
import threading

class rPPG_agent(threading.Thread):
    def __init__(self):
        super().__init__()
        self.frame_queue = queue.Queue(maxsize=1)
        self.daemon = True
        self.latest_result = None

    def enqueue_frame(self, frame):
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put(frame)

    def run(self):
        while True:
            frame = self.frame_queue.get()

            # rPPG implementation logic