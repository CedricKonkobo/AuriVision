import cv2
from ultralytics import YOLO
import numpy as np
from aurivision.estimation.detail_objects import ObstacleDetail
from aurivision.decision.blind_guidance_optimized import DecisionEngine
from aurivision.decision.action import ActionModule
from aurivision import config
import time

# Charger le modèle YOLO
model = YOLO(config.MODEL_PATH)  # yolov8n est le modèle le plus léger et rapide

# Initialisation
detector = ObstacleDetail()
decision = DecisionEngine()
action = ActionModule()

# Ouvrir la caméra (0 = caméra par défaut)
cap = cv2.VideoCapture(config.LIVE_CAMERA_INDEX)

if not cap.isOpened():
    print("Erreur: Impossible d'ouvrir la caméra")
    exit()

# Définir la résolution (optionnel)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Détection en cours... Appuyez sur 'q' pour quitter")

# Couleurs aléatoires pour chaque classe
np.random.seed(42)
colors = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)

while True:
    # Lire une frame de la caméra
    ret, frame = cap.read()
    
    if not ret:
        print("Erreur: Impossible de lire la frame")
        break
    
    # Effectuer la détection
    results = model(frame, verbose=False)

    # Traiter les détections pour obtenir des détails supplémentaires
    obstacles = detector.process_frame(results)
    
    # Decision a prendre
    instruction = decision.process_obstacles(obstacles)

    # Action (parler si nécessaire)
    if instruction:
       action.process_instruction(instruction)
    
    
    # Parcourir les détections
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            # Extraire les coordonnées du rectangle
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # Extraire la confiance et la classe
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Obtenir le nom de la classe
            class_name = model.names[cls]
            
            # Choisir une couleur pour cette classe
            color = tuple(map(int, colors[cls % len(colors)]))
            
            # Dessiner le rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Préparer le texte (nom de la classe + confiance)
            label = f'{class_name} {conf:.2f}'
            
            # Calculer la taille du texte pour le fond
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Dessiner un rectangle de fond pour le texte
            cv2.rectangle(
                frame, 
                (x1, y1 - text_height - 10), 
                (x1 + text_width, y1), 
                color, 
                -1
            )
            
            # Écrire le texte
            cv2.putText(
                frame, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1
            )
    
    # Afficher le nombre d'objets détectés
    num_detections = len(results[0].boxes) if len(results) > 0 else 0
    cv2.putText(
        frame,
        f'Objets detectes: {num_detections}',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    
    # Afficher la vidéo avec les détections
    cv2.imshow('Detection YOLO - Appuyez sur Q pour quitter', frame)
    
    # Quitter si 'q' est pressé
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
print("Détection terminée")
