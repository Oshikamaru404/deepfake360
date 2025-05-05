from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from tqdm import tqdm
import dlib
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, AveragePooling2D, Dropout, Flatten, Dense
import librosa
import soundfile as sf
import os
import tempfile

app = FastAPI()

# === Définir l'architecture du modèle Meso4 ===
def Meso4():
    x = Input(shape=(256, 256, 3))

    y = Conv2D(8, (3, 3), padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = AveragePooling2D(pool_size=(2, 2))(y)

    y = Conv2D(8, (5, 5), padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = AveragePooling2D(pool_size=(2, 2))(y)

    y = Conv2D(16, (5, 5), padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = AveragePooling2D(pool_size=(2, 2))(y)

    y = Conv2D(16, (5, 5), padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = AveragePooling2D(pool_size=(4, 4))(y)

    y = Flatten()(y)
    y = Dropout(0.5)(y)
    y = Dense(16)(y)
    y = Activation('relu')(y)
    y = Dropout(0.5)(y)
    y = Dense(1, activation='sigmoid')(y)

    return Model(inputs=x, outputs=y)

# === Charger le modèle et les poids ===
model = Meso4()
model.load_weights("weights/Meso4_DF.h5")

# Face ROI Detection
def detect_forehead_roi(frame, cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        roi_y = int(y + 0.1 * h)
        roi_h = int(0.25 * h)
        roi = frame[roi_y:roi_y+roi_h, x:x+w]
        return roi, (x, y, w, h)
    return None, None

# Filtering
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data)

def smooth_signal(signal, window_size=5):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

# Heart Rate
def estimate_heart_rate(signal, fs, min_bpm=45, max_bpm=180):
    n = len(signal)
    freqs = np.fft.fftfreq(n, d=1/fs)
    fft = np.abs(np.fft.fft(signal))
    mask = (freqs > min_bpm/60) & (freqs < max_bpm/60)
    freqs = freqs[mask]
    fft = fft[mask]
    if len(freqs) == 0:
        return 0, [], []
    peak_freq = freqs[np.argmax(fft)]
    bpm = peak_freq * 60
    return bpm, freqs*60, fft

def estimate_hrv(signal, fs):
    peaks, _ = find_peaks(signal, distance=fs*0.6)
    if len(peaks) < 2:
        return 0
    intervals = np.diff(peaks) / fs
    hrv = np.std(intervals) * 1000
    return hrv

# Voice Resonance Verifier
def extract_audio(video_path):
    y, sr = librosa.load(video_path, sr=None)
    return y, sr

def analyze_voice_resonance(y, sr):
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean

# Deepfake Score
def estimate_deepfake_score(bpm, hrv, signal_quality, ear_mean, lap_var=None, heatmap_entropy_score=None, gan_score=0, voice_score=0):
    # Critères physiologiques
    abnormal_hr = bpm < 63 or bpm > 140  # Adjusted heart rate range for speaking person
    abnormal_hrv = hrv > 300 or hrv < 5   # HRV trop haute ou trop basse
    abnormal_ear = ear_mean < 0.15 or ear_mean > 0.4  # EAR suspecte
    bad_quality = signal_quality < 100.0  # Faible qualité vidéo (à ajuster selon tests)

    # Critères visuels
    low_texture = lap_var is not None and lap_var < 80  # Faible laplacienne = image lissée, deepfake
    high_entropy = heatmap_entropy_score is not None and heatmap_entropy_score > 6  # Bruit élevé suspect

    # Pondération avancée
    score = 0.0
    if abnormal_hr:
        score += 0.45
    if abnormal_hrv:
        score += 0.20
    if abnormal_ear:
        score += 0.20
    if bad_quality:
        score += 0.10
    if low_texture:
        score += 0.15
    if high_entropy:
        score += 0.10

    # Include GAN score
    score += gan_score * 0.3

    # Include Voice score
    score += voice_score * 0.1

    # Bonus (réduction) si tout semble physiologiquement humain
    if not abnormal_hr and not abnormal_hrv and not abnormal_ear and signal_quality > 150 and lap_var > 120:
        score -= 0.10  # confiance dans la vidéo réelle

    # Clamp entre 0 et 1
    score = max(0.0, min(score, 1.0))
    return score

# Signal Extraction
def extract_signal(source, cascade, use_webcam=False):
    cap = cv2.VideoCapture(0 if use_webcam else source)
    fps = cap.get(cv2.CAP_PROP_FPS) if not use_webcam else 30
    if fps < 1 or fps > 120:
        fps = 30
        print("⚠️ Invalid FPS. Defaulting to 30.")
    else:
        print(f"[INFO] Frame rate: {fps:.2f} FPS")

    signals, face_boxes = [], []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not use_webcam else 300

    for _ in tqdm(range(total_frames), desc="Extracting signal"):
        ret, frame = cap.read()
        if not ret:
            break
        roi, box = detect_forehead_roi(frame, cascade)
        if roi is not None:
            mean_rgb = np.mean(roi, axis=(0, 1))
            signals.append(mean_rgb)
            face_boxes.append((frame, box))

    cap.release()
    return np.array(signals), fps, face_boxes

# EAR
def detect_ear(landmarks):
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    def ear(eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)
    return (ear(left_eye) + ear(right_eye)) / 2

# Heatmap
def generate_texture_heatmap(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    gray_ubyte = img_as_ubyte(rgb2gray(frame))
    entropy_map = entropy(gray_ubyte, disk(5))
    entropy_norm = cv2.normalize(entropy_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_entropy = cv2.applyColorMap(entropy_norm, cv2.COLORMAP_JET)
    entropy_score = np.mean(entropy_map)
    return laplacian_var, entropy_score, heatmap_entropy

# Plotting
def plot_results(filtered_signal, freqs, power, bpm, deepfake_score, hrv, ear_values, entropy_heatmap, lap_var):
    fig, axs = plt.subplots(5, 1, figsize=(10, 12))

    axs[0].plot(filtered_signal, color='g')
    axs[0].set_title("Filtered rPPG Signal (Green channel)")
    axs[0].set_xlabel("Frame")
    axs[0].set_ylabel("Intensity")

    if len(freqs) > 0:
        axs[1].plot(freqs, power, color='r')
        axs[1].set_title(f"FFT Spectrum - Estimated HR: {bpm:.2f} bpm")
        axs[1].set_xlabel("BPM")
        axs[1].set_ylabel("Power")
    else:
        axs[1].text(0.5, 0.5, 'No spectrum available', ha='center')

    axs[2].bar(["Deepfake Probability", "HRV (ms)", "LapVar"], [deepfake_score, hrv, lap_var], color=['blue', 'orange', 'green'])
    axs[2].set_ylim(0, max(1, hrv + 50))

    axs[3].plot(ear_values, color='b')
    axs[3].set_title("Eye Aspect Ratio (EAR)")
    axs[3].set_xlabel("Frame")
    axs[3].set_ylabel("EAR")

    axs[4].imshow(cv2.cvtColor(entropy_heatmap, cv2.COLOR_BGR2RGB))
    axs[4].set_title("Texture Heatmap (Local Entropy)")
    axs[4].axis('off')

    plt.tight_layout()
    plt.show()

# Generate Report
def generate_report(video_path, deepfake_score, bpm, hrv, ear_mean, lap_var, entropy_score, gan_score, voice_score):
    report = f"""
    Deepfake Detection Report
    -------------------------
    Video Path: {video_path}
    Deepfake Probability: {deepfake_score*100:.1f}%

    Physiological Indicators:
    -------------------------
    Heart Rate (BPM): {bpm:.2f}
    Heart Rate Variability (HRV): {hrv:.2f} ms
    Eye Aspect Ratio (EAR) Mean: {ear_mean:.2f}

    Visual Indicators:
    -----------------
    Laplacian Variance: {lap_var:.1f}
    Entropy Score: {entropy_score:.2f}

    DeepFake Detection Scores:
    --------------------------
    GAN Score: {gan_score:.2f}
    Voice Score: {voice_score:.2f}

    Conclusion:
    -----------
    """
    if deepfake_score > 0.51:
        report += "The video is probably a DeepFake."
    else:
        report += "The video is likely real."

    print(report)

    # Optionally, save the report to a file
    with open("deepfake_report.txt", "w") as f:
        f.write(report)

    return report

@app.post("/analyze/")
async def analyze_video(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, file.filename)
        with open(video_path, "wb") as f:
            f.write(await file.read())

        # Load the cascade classifier
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

        # Extract signals
        signals, fps, face_data = extract_signal(video_path, cascade)

        if len(signals) < fps * 5:
            return JSONResponse(status_code=400, content={"message": "Not enough signal. Try a longer video or better lighting."})

        green = signals[:, 1]
        green = np.nan_to_num((green - np.mean(green)) / np.std(green))
        green = smooth_signal(green)
        filtered = bandpass_filter(green, 0.7, 4.0, fps)

        bpm, freqs, power = estimate_heart_rate(filtered, fps)
        hrv = estimate_hrv(filtered, fps)
        signal_quality = np.max(power) / np.sum(power) if len(power) > 0 else 0

        ear_values = []
        for frame_img, _ in tqdm(face_data, desc="Extracting EAR signal"):
            gray_frame = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray_frame)
            if len(faces) > 0:
                landmarks = predictor(gray_frame, faces[0])
                ear = detect_ear(np.array([[p.x, p.y] for p in landmarks.parts()]))
                ear_values.append(ear)

        ear_mean = np.mean(ear_values) if ear_values else 0

        # Texture
        sample_frame = face_data[len(face_data)//2][0]
        lap_var, entropy_score, entropy_heatmap = generate_texture_heatmap(sample_frame)

        # DeepFake detection using Meso4 model
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return JSONResponse(status_code=400, content={"message": "Impossible d'ouvrir la vidéo."})

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_samples = 10  # Number of frames to test
        step = frame_count // num_samples
        predictions = []
        probably_fake = False

        for i in range(num_samples):
            frame_number = i * step
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            if not ret:
                continue

            img = cv2.resize(frame, (256, 256))
            img = img.astype('float32') / 255.
            img = np.expand_dims(img, axis=0)

            prediction = model.predict(img)[0][0]
            predictions.append(prediction)

            # Check if the prediction exceeds the threshold
            if prediction > 0.5:
                probably_fake = True
                break

        cap.release()

        gan_score = np.mean(predictions) if predictions else 0

        # Voice Resonance Verifier
        y, sr = extract_audio(video_path)
        mfccs_mean = analyze_voice_resonance(y, sr)

        # Calculate voice score based on MFCC features
        voice_score = np.mean(mfccs_mean) / 100  # Normalize the score

        deepfake_score = estimate_deepfake_score(
            bpm, hrv, signal_quality, ear_mean,
            lap_var=lap_var,
            heatmap_entropy_score=entropy_score,
            gan_score=gan_score,
            voice_score=voice_score
        )

        # Generate report
        report = generate_report(video_path, deepfake_score, bpm, hrv, ear_mean, lap_var, entropy_score, gan_score, voice_score)

        # Return the report as JSON response
        return JSONResponse(content={"report": report})

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
