import cv2
import numpy as np
import mediapipe as mp
import sys

# Parámetros optimizados
SCALE_FACTOR = 0.5
FLOW_SCALE = 8
MOTION_THRESHOLD = 1.5
DENSE_SPACING = 6  # Espaciado entre vectores en píxeles

LK_PARAMS = dict(winSize=(25, 25),  # Ventana más grande para mejor precisión
                 maxLevel=3,        # Más niveles piramidales
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

# Inicializar MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, 
                   model_complexity=1,
                   min_detection_confidence=0.5)

# Manejo de argumentos
if len(sys.argv) < 2:
    print("Uso: python optical_flow_mediapipe.py [video_file]")
    sys.exit()

cap = cv2.VideoCapture(sys.argv[1])

# Configuración inicial
ret, prev_frame = cap.read()
if not ret:
    print("Error leyendo video")
    sys.exit()

prev_frame = cv2.resize(prev_frame, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_features = None

# Ventanas
cv2.namedWindow("Video Original", cv2.WINDOW_NORMAL)
cv2.namedWindow("Flujo Óptico Filtrado", cv2.WINDOW_NORMAL)
cv2.namedWindow("Máscara Movimiento", cv2.WINDOW_NORMAL)

def create_grid_points(shape, spacing):
    h, w = shape
    x = np.arange(0, w, spacing)
    y = np.arange(0, h, spacing)
    return np.array(np.meshgrid(x, y)).T.reshape(-1, 1, 2).astype(np.float32)

while True:
    ret, frame = cap.read()
    if not ret: 
        break
    
    # Redimensionar frame
    frame = cv2.resize(frame, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    overlay = frame.copy()
    motion_mask = np.zeros_like(frame_gray, dtype=np.uint8)
    
    # Detección de pose
    results = pose.process(frame_rgb)
    
    if results.pose_landmarks:
        keypoints = [
            mp_pose.PoseLandmark.NOSE,
            mp_pose.PoseLandmark.LEFT_EYE,
            mp_pose.PoseLandmark.RIGHT_EYE,
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.LEFT_WRIST,
            mp_pose.PoseLandmark.RIGHT_WRIST,
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP,
        ]
        
        coords = []
        for kp in keypoints:
            lm = results.pose_landmarks.landmark[kp]
            x = int(lm.x * w)
            y = int(lm.y * h)
            coords.append((x, y))
        
        if coords:
            xs, ys = zip(*coords)
            padding = 60
            x1 = max(min(xs) - padding, 0)
            y1 = max(min(ys) - padding, 0)
            x2 = min(max(xs) + padding, w)
            y2 = min(max(ys) + padding, h)
            
            if x2 > x1 and y2 > y1:
                # Crear grilla densa en la ROI
                roi_shape = (y2 - y1, x2 - x1)
                grid_points = create_grid_points(roi_shape, DENSE_SPACING)
                grid_points += np.array([x1, y1], dtype=np.float32)  # Ajustar coordenadas
                
                # Filtrar puntos en máscara
                mask_roi = np.zeros_like(frame_gray)
                mask_roi[y1:y2, x1:x2] = 255
                prev_features = cv2.goodFeaturesToTrack(prev_gray, 
                                                      maxCorners=500,
                                                      qualityLevel=0.005,
                                                      minDistance=3,
                                                      blockSize=7,
                                                      mask=mask_roi)
                
                # Combinar detección automática con grilla
                if prev_features is not None:
                    prev_features = np.vstack([prev_features, grid_points])
                else:
                    prev_features = grid_points
                
                # Tracking Lucas-Kanade
                curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, frame_gray, prev_features, None, **LK_PARAMS)
                
                # Filtrar y procesar
                valid = status.ravel() == 1
                good_prev = prev_features[valid].reshape(-1, 2)
                good_curr = curr_pts[valid].reshape(-1, 2)
                
                if len(good_prev) > 0:
                    # Calcular vectores
                    vectors = good_curr - good_prev
                    magnitudes = np.linalg.norm(vectors, axis=1)
                    
                    # Filtrar por magnitud y dibujar
                    valid_idx = magnitudes > MOTION_THRESHOLD
                    for (x0, y0), (dx, dy) in zip(good_prev[valid_idx], vectors[valid_idx]):
                        x_end = int(x0 + dx * FLOW_SCALE)
                        y_end = int(y0 + dy * FLOW_SCALE)
                        
                        # Dibujar línea con degradado
                        color_intensity = min(255, int(150 + magnitudes.mean() * 50))
                        cv2.line(overlay, (int(x0), int(y0)), 
                                (x_end, y_end), (0, color_intensity, 0), 1)
                        cv2.circle(overlay, (int(x0), int(y0)), 
                                 1, (0, 255, 0), -1)
                        
                        # Actualizar máscara
                        motion_mask[int(y0), int(x0)] = 255
                    
                    # Suavizar máscara
                    motion_mask = cv2.dilate(motion_mask, np.ones((3,3), np.uint8))
                
                # Actualizar features
                prev_features = good_curr.reshape(-1, 1, 2)
                
                # Dibujar ROI
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,0,255), 2)

    # Mostrar resultados
    cv2.imshow("Video Original", frame)
    cv2.imshow("Flujo Óptico Filtrado", overlay)
    cv2.imshow("Máscara Movimiento", motion_mask)
    
    # Actualizar frame anterior
    prev_gray = frame_gray.copy()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()