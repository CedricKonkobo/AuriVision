from aurivision.preprocessing.image_traitement import ImageTraitement
from aurivision.detection.yolo_object_filter import YOLOObjectFilter, StableObject
from aurivision.decision.blind_guidance_system import BlindGuidanceSystem
from aurivision import config
import cv2
import numpy as np
from typing import List, Dict
import threading
import time
from collections import deque
from dataclasses import asdict


def capture_and_process():
    # Créer une instance du traitement
    traitement = ImageTraitement()

    # Créer une instance du détecteur YOLO
    detector = YOLOObjectFilter(
        model_path=config.MODEL_PATH,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        iou_threshold=config.IOU_THRESHOLD,
        stability_threshold=config.STABILITY_THRESHOLD
    )

    # Creer une instnce de systeme de decision
    guidance_system = BlindGuidanceSystem()

    
    # Ouvrir la caméra en haute résolution si possible
    cap = cv2.VideoCapture(config.MAIN_CAMERA_INDEX)
    
    # Augmenter la résolution de capture (optionnel, dépend de votre caméra)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    
    frames = []
    print("Capture des 3 frames en haute résolution...")
    
    for i in range(4):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            print(f"✓ Frame {i+1} capturée (résolution: {frame.shape[1]}x{frame.shape[0]})")
            
            # Petit délai entre chaque capture
            cv2.waitKey(500)
        else:
            print(f"✗ Erreur capture frame {i+1}")
    
    cap.release()
    
    if len(frames) == 4:
        print("\nTraitement des images en cours...")
        
        # Envoyer les frames au module de traitement
        fichiers, stats = traitement.traiter_frames(frames[-3:])
        
        
        # Afficher les 3 images traitées
        #afficher_resultats(fichiers, stats)

        # # Vérifier qu'on a bien 3 images traitées
        if len(fichiers) == 3:
            print("\n" + "="*60)
            print("DÉTECTION YOLO SUR LES IMAGES TRAITÉES")
            print("="*60)
            
            # Détecter les objets stables avec YOLO
            stable_objects = detector.process_frames(
                frames=fichiers,
                reference_frame_index=1  # Frame central comme référence
            )
            
            # Afficher les obstacles détectés
            #afficher_obstacles(stable_objects)

            # Decision a prendre
            instruction = guidance_system.process_obstacles([asdict(o) for o in stable_objects])
            print("=== Scénario ===")
            if instruction:
                print(f"Action: {instruction.action.value}")
                print(f"Message: {instruction.message}")
                print(f"Urgence: {instruction.urgency.name}")
    
        else:
            print("Erreur : Impossible de charger les images traitées")

    else:
        print("Erreur : Impossible de capturer les 3 frames")

def dessiner_obstacles(image: np.ndarray, obstacles: List[StableObject]):
    """Dessine sur l'image la boîte, la classe et la distance de chaque obstacle."""
    for obj in obstacles:
        x, y, w, h = [int(v) for v in obj.bbox]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{obj.object_class} {obj.estimated_distance:.1f} m"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        top = max(y, th + 10)
        cv2.rectangle(image, (x, top - th - 10), (x + tw, top), (0, 255, 0), -1)
        cv2.putText(image, label, (x, top - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def run_live():
    """
    Boucle temps réel : caméra -> amélioration d'image -> YOLO multi-frame
    -> distances -> instructions de guidage écrites dans la console.
    Fenêtre avec boîtes et distances ; 'q' pour quitter.
    """
    traitement = ImageTraitement()
    detector = YOLOObjectFilter(
        model_path=config.MODEL_PATH,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
        iou_threshold=config.IOU_THRESHOLD,
        stability_threshold=config.STABILITY_THRESHOLD
    )
    guidance_system = BlindGuidanceSystem()

    cap = cv2.VideoCapture(config.MAIN_CAMERA_INDEX)
    if not cap.isOpened() and config.MAIN_CAMERA_INDEX != config.LIVE_CAMERA_INDEX:
        print(f"Caméra {config.MAIN_CAMERA_INDEX} indisponible, "
              f"essai de la caméra {config.LIVE_CAMERA_INDEX}")
        cap = cv2.VideoCapture(config.LIVE_CAMERA_INDEX)
    if not cap.isOpened():
        print("Erreur: impossible d'ouvrir la caméra "
              "(voir scripts/camera_index_check.py pour trouver l'index)")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    frames = deque(maxlen=3)
    print("Guidage en cours... Appuyez sur 'q' dans la fenêtre pour quitter")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erreur: impossible de lire la frame")
                break
            frames.append(frame)
            display = frame

            if len(frames) == 3:
                fichiers, stats = traitement.traiter_frames(list(frames))
                stable_objects = detector.process_frames(
                    frames=fichiers,
                    reference_frame_index=1
                )
                display = fichiers[1].copy()
                dessiner_obstacles(display, stable_objects)

                instruction = guidance_system.process_obstacles(
                    [asdict(o) for o in stable_objects]
                )
                if instruction:
                    print("=== Directive ===")
                    print(f"Action: {instruction.action.value}")
                    print(f"Message: {instruction.message}")
                    print(f"Urgence: {instruction.urgency.name}")

            cv2.imshow("AuriVision - q pour quitter", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def afficher_resultats(images: List[np.ndarray], stats: Dict):
    """
    Affiche les 3 images traitées et les statistiques
    
    Args:
        images: Liste de 3 images traitées (numpy arrays)
        stats: Dictionnaire des statistiques de traitement
    """
    # Afficher les images
    for i, img in enumerate(images):
        cv2.imshow(f"Image {i+1}", img)
    
    # Afficher les statistiques dans la console
    print("\n" + "="*70)
    print(" STATISTIQUES DE TRAITEMENT")
    print("="*70)
    print(f"Total d'images: {stats['total_frames']}")
    print(f"Images traitées: {stats['processed_frames']}")
    print(f"Images de bonne qualité (non traitées): {stats['skipped_frames']}")
    
    if stats['treatments_applied']:
        print(f"\nTraitements appliqués:")
        for treatment, count in stats['treatments_applied'].items():
            print(f"  • {treatment}: {count} fois")
    
    print("="*70)
    
    # Attendre que l'utilisateur ferme les fenêtres
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def afficher_obstacles(obstacles: List[StableObject]):
    """
    Affiche les caractéristiques de chaque obstacle détecté
    
    Args:
        obstacles: Liste des objets stables détectés
    """
    if not obstacles:
        print("\n" + "="*60)
        print("AUCUN OBSTACLE DÉTECTÉ")
        print("="*60)
        return
    
    print("\n" + "="*60)
    print(f"OBSTACLES DÉTECTÉS: {len(obstacles)}")
    print("="*60)
    
    for i, obj in enumerate(obstacles, 1):
        print(f"\n{'─'*60}")
        print(f"OBSTACLE #{i}")
        print(f"{'─'*60}")
        
        print(f"  ID Objet              : {obj.object_id}")
        print(f"  lasse                : {obj.object_class}")
        print(f"  Confiance Détection   : {obj.detection_confidence:.2%}")
        print(f"  Position Horizontale  : {obj.position_horizontal}")
        print(f"  Distance Estimée      : {obj.estimated_distance:.2f} m")
        print(f"  En Mouvement          : {' OUI' if obj.is_moving else ' NON'}")
        print(f"  Direction Mouvement   : {obj.movement_direction}")
        print(f"  Niveau de Risque     : {obj.intrinsic_risk_level}")
        print(f"  Passable              : {' OUI' if obj.is_passable else ' NON'}")
        print(f"  Ratio d'Occupation    : {obj.occupancy_ratio:.2%}")
        print(f"  Score Stabilité       : {obj.stability_score:.2%}")
        
        # Informations techniques supplémentaires
        print(f"\n  Détails Techniques:")
        print(f"    • Boîte Englobante     : x={obj.bbox[0]}, y={obj.bbox[1]}, "
              f"w={obj.bbox[2]}, h={obj.bbox[3]}")
        print(f"    • Centre               : ({obj.center[0]:.1f}, {obj.center[1]:.1f})")
        print(f"    • Surface              : {obj.area:.0f} px²")
        print(f"    • Vélocité             : ({obj.velocity[0]:.2f}, {obj.velocity[1]:.2f}) px/frame")
    
    print("\n" + "="*60)
    
    # Résumé statistique
    print("\n RÉSUMÉ STATISTIQUE:")
    print(f"  • Total obstacles        : {len(obstacles)}")
    print(f"  • Objets en mouvement    : {sum(1 for o in obstacles if o.is_moving)}")
    print(f"  • Objets statiques       : {sum(1 for o in obstacles if not o.is_moving)}")
    print(f"  • Objets passables       : {sum(1 for o in obstacles if o.is_passable)}")
    print(f"  • Objets non-passables   : {sum(1 for o in obstacles if not o.is_passable)}")
    
    # Répartition par niveau de risque
    risk_counts = {}
    for obj in obstacles:
        risk_counts[obj.intrinsic_risk_level] = risk_counts.get(obj.intrinsic_risk_level, 0) + 1
    
    print(f"\n  Répartition par Risque:")
    for risk_level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
        count = risk_counts.get(risk_level, 0)
        if count > 0:
            print(f"    • {risk_level:8s} : {count}")
    
    # Répartition par position
    pos_counts = {}
    for obj in obstacles:
        pos_counts[obj.position_horizontal] = pos_counts.get(obj.position_horizontal, 0) + 1
    
    print(f"\n  Répartition par Position:")
    for position in ['LEFT', 'CENTER', 'RIGHT']:
        count = pos_counts.get(position, 0)
        if count > 0:
            print(f"    • {position:7s} : {count}")
    
    # Obstacle le plus proche
    closest = min(obstacles, key=lambda o: o.estimated_distance)
    print(f"\n   Obstacle le plus proche:")
    print(f"    • {closest.object_class} à {closest.estimated_distance:.2f}m "
          f"({closest.position_horizontal})")
    
    print("\n" + "="*60 + "\n")

running = True

def check_input():
    """Thread qui attend que l'utilisateur appuie sur Entrée"""
    global running
    input()  # Attend la touche Entrée
    running = False
    print("\n🛑 Arrêt demandé...")
if __name__ == "__main__":
    #Démarrer le thread de surveillance

    #Boucle principale
    #while running:
    run_live()

    #print("✅ Boucle terminée.")
