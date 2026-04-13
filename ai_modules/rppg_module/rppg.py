import queue
import threading
import numpy as np
from scipy import signal as scipy_signal
from collections import deque

# ===============
# ROI EXTRACTION
# ===============
def extract_roi_signals(frame, landmarks):
    # landmarks indices -> [forehead, left, right]
    LANDMARK_INDICES = [151, 234, 454]

    # box size for ROI -> 30x30
    BOX_HALF_SIZE = 30
    
    h, w = frame.shape[:2]
    means = []
    
    for idx in LANDMARK_INDICES:
        # get landmark for current frame
        landmark = landmarks[idx]

        # landmark (0-1) -> actual position on frame
        x_pixel = int(landmark.x * w)
        y_pixel = int(landmark.y * h)
        
        # calculate corners of ROI boxes
        x1 = max(0, x_pixel - BOX_HALF_SIZE)
        y1 = max(0, y_pixel - BOX_HALF_SIZE)
        x2 = min(w, x_pixel + BOX_HALF_SIZE)
        y2 = min(h, y_pixel + BOX_HALF_SIZE)
        
        # extract ROI boxes from frame
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
        
        # Extract R, G, B for ROIs
        mean_B = np.mean(roi[:, :, 0])
        mean_G = np.mean(roi[:, :, 1])
        mean_R = np.mean(roi[:, :, 2])
        
        means.append([mean_R, mean_G, mean_B])
    
    # Average across all 3 ROIs
    means = np.array(means)
    return np.mean(means, axis=0)


# =============================
# SIGNAL PROCESSING ALGORITHMS
# =============================
def _compute_pos(C_norm):
        # projection matrix (skin optics theory)
        S = np.array([[0,  1, -1], [-2, 1,  1]], dtype=float)
        
        # Project: S @ C_norm.T
        # C_norm - (N, 3) -> C_norm.T - (3, N)
        # Result H - (2, N)
        H = S @ C_norm.T
        
        # Extract the two signals
        H1 = H[0, :]
        H2 = H[1, :]
        
        # applying standard deviation
        std_H1 = np.std(H1)
        std_H2 = np.std(H2)
        
        # better signal-to-noise ratio -> more weight
        weight = std_H1 / (std_H2 + 1e-8)
        pulse = H1 + weight * H2
        
        return pulse

def _compute_chrom(C_norm):
    # separating color channels
    R = C_norm[:, 0]
    G = C_norm[:, 1]
    B = C_norm[:, 2]
 
    # Xs -> red-green chrominance axis
    # Ys -> luminance-balanced chrominance axis
    Xs = 3 * R - 2 * G
    Ys = 0.5 * R - G + 0.5 * B
 
    # standard deviations
    std_Xs = np.std(Xs)
    std_Ys = np.std(Ys)
 
    # pulse calculations
    weight = std_Xs / (std_Ys + 1e-8)
    pulse = Xs - weight * Ys
 
    return pulse

def _normalize_signal(signal):
    std = np.std(signal)
    if std < 1e-8:
        return np.zeros_like(signal)
    return (signal - np.mean(signal)) / std


def _compute_bpm(pulse, fps):
    N = len(pulse)

    # hanning window -> reduced spectral leakage
    windowed = np.hanning(N) * pulse

    # fft calculations
    fft_result = np.fft.rfft(windowed)
    magnitude = np.abs(fft_result)
    frequencies = np.fft.rfftfreq(N, d=1.0 / fps)
 
    # bandpass filter -> blocks out-of-range frequencies for heart activity
    # 40-180 BPM -> 0.67-3.0 Hz
    valid_mask = (frequencies > 0.67) & (frequencies < 3.0)
 
    if np.sum(valid_mask) == 0:
        return 0.0, 0.0
    
    valid_magnitudes = magnitude[valid_mask]
    valid_frequencies = frequencies[valid_mask]

    # find peak frequency in range
    peak_idx = np.argmax(valid_magnitudes)
    peak_freq = valid_frequencies[peak_idx]
    peak_mag = valid_magnitudes[peak_idx]
 
    # SNR -> ratio of peak magnitude to mean of magnitudes
    # distinct, sharp peak -> high SNR
    # noisy peak -> low SNR
    mean_mag = np.mean(valid_magnitudes)
    snr = peak_mag / (mean_mag + 1e-8)
 
    bpm = peak_freq * 60.0
 
    return bpm, snr
 
 
def _compute_hrv(pulse, fps):
    if len(pulse) < 10:
        return None
 
    # minimum distance between peaks
    # setting maximum heart rate of 180 BPM
    # 180 BPM = 3 Hz -> minimum period = fps/3 samples
    # false peaks -> eliminated
    min_distance_samples = int(fps / 3.0)

    # peaks is a list of indices of pulse
    # each index represent a peak in pusle
    peaks, _ = scipy_signal.find_peaks(
        pulse,
        distance=min_distance_samples,
        # ignore tiny fluctuations that aren't real beats
        prominence=0.1
    )

    # at least 3 peaks -> 2 intervals -> meaningful std calc
    if len(peaks) < 3:
        return None
 
    # Convert peak sample indices to time in seconds
    # peaks (indices of peaks in pusle)
    # peak_times_sec -> timestamp of the peak in pusle
    peak_times_sec = peaks / fps
 
    # RR intervals: time between consecutive peaks
    rr_intervals_sec = np.diff(peak_times_sec)
 
    # standard deviation or RR-intervals -> SDNN
    sdnn_ms = float(np.std(rr_intervals_sec) * 1000.0)
 
    return sdnn_ms
 
 
def _compute_confidence(snr_pos, snr_chrom, bpm_pos, bpm_chrom):
    # SNR -> [1.0, 10.0]
    # SNR Score -> [0.0, 1.0]
    avg_snr = (snr_pos + snr_chrom) / 2.0
    snr_score = np.clip((avg_snr - 1.0) / 9.0, 0.0, 1.0)

    # contradicting outputs -> low confidence
    bpm_diff = abs(bpm_pos - bpm_chrom)
    # 0 BPM difference -> score 1.0
    # 15+ BPM difference -> score 0.0
    agreement_score = np.clip(1.0 - (bpm_diff / 15.0), 0.0, 1.0)
 
    confidence = 0.6 * snr_score + 0.4 * agreement_score
 
    return float(confidence)
 
 
def _compute_stress_index(bpm, sdnn_ms):
    if sdnn_ms is None:
        # Without HRV, estimate from BPM only (less reliable)
        bpm_stress = np.clip((bpm - 60.0) / 40.0, 0.0, 1.0)
        return float(bpm_stress)
 
    # BPM component: 60 BPM -> 0.0, 100 BPM -> 1.0
    bpm_stress = np.clip((bpm - 60.0) / 40.0, 0.0, 1.0)
 
    # HRV component: SDNN 80ms -> 0.0 (calm), SDNN 10ms -> 1.0 (stressed)
    hrv_stress = np.clip(1.0 - (sdnn_ms - 10.0) / 70.0, 0.0, 1.0)

    stress_index = 0.5 * bpm_stress + 0.5 * hrv_stress

    return float(stress_index)


class rPPG_agent(threading.Thread):
    def __init__(self, fps=30):
        super().__init__()
        self.daemon = True
        self.frame_queue = queue.Queue(maxsize=1)
        self.window_frames = fps * 5
        self.signal_window = deque(maxlen=self.window_frames)
        self.fps = fps
        self.latest_result = None
        self.running = False

        # # performance testing
        # self.frame_count = 0
        # self.start_time = None

    def enqueue_frame(self, frame, landmarks):
        if landmarks is None or len(landmarks) == 0: # no face detected
            return
        
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put((frame , landmarks))


    def run(self):
        self.running = True

        while self.running:
            try:
                bgr_frame, landmarks_list = self.frame_queue.get(timeout=0.1)
                if bgr_frame is None:
                    break

                # Extract the per-frame [R, G, B] mean across 3 ROI boxes
                roi_signal = extract_roi_signals(bgr_frame, landmarks_list[0])

                if roi_signal is not None:
                    self.signal_window.append(roi_signal)

                # Wait until the window is fully populated before computing
                if len(self.signal_window) < self.window_frames:
                    continue
 
                # Build (N, 3) signal matrix from the deque
                C = np.array(list(self.signal_window))  # shape: (125, 3)
 
                # Temporal normalization: divide each channel by its mean
                # over the window. This removes slow DC drift from lighting
                # changes and makes values fluctuate around 1.0.
                # Both POS and CHROM operate on this normalized form.
                C_mean = np.mean(C, axis=0, keepdims=True)
                C_norm = C / (C_mean + 1e-8)
 
                # --- Run both algorithms on the same normalized data ---
                pulse_pos = _compute_pos(C_norm)
                pulse_chrom = _compute_chrom(C_norm)
 
                # --- Get BPM and SNR from each algorithm independently ---
                bpm_pos, snr_pos = _compute_bpm(pulse_pos, self.fps)
                bpm_chrom, snr_chrom = _compute_bpm(pulse_chrom, self.fps)
 
                # --- Combine POS and CHROM pulse signals ---
                # Normalize both to zero-mean unit-variance before averaging.
                # Without this step, whichever algorithm has larger amplitude
                # would dominate the combination, defeating the purpose.
                pulse_pos_norm = _normalize_signal(pulse_pos)
                pulse_chrom_norm = _normalize_signal(pulse_chrom)
                pulse_combined = 0.5 * pulse_pos_norm + 0.5 * pulse_chrom_norm
 
                # --- Final BPM from the combined signal ---
                bpm_combined, snr_combined = _compute_bpm(pulse_combined, self.fps)
 
                # --- HRV from the combined pulse waveform ---
                sdnn_ms = _compute_hrv(pulse_combined, self.fps)
 
                # --- Confidence score ---
                confidence = _compute_confidence(snr_pos, snr_chrom, bpm_pos, bpm_chrom)
 
                # --- Stress index ---
                stress_index = _compute_stress_index(bpm_combined, sdnn_ms)
 
                # --- Write structured result (atomic dict assignment) ---
                self.latest_result = {
                    "bpm":            round(bpm_combined, 1),
                    "hrv_sdnn":       round(sdnn_ms, 1) if sdnn_ms is not None else None,
                    "confidence":     round(confidence, 3),
                    "stress_index":   round(stress_index, 3),
                }
 
            except queue.Empty:
                continue
 
    def stop(self):
        self.running = False
        try:
            self.frame_queue.put_nowait(None)
        except queue.Full:
            pass