"""
Système de Détection et Filtrage d'Objets YOLO
Traitement de 3 frames consécutives pour éliminer les faux positifs
"""

import cv2
from aurivision.config import MODEL_PATH
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from ultralytics import YOLO
import math


@dataclass
class Detection:
    """Représente une détection d'objet"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[float, float]
    area: float
    frame_index: int


@dataclass
class StableObject:
    """Représente un objet stable détecté dans plusieurs frames"""
    object_id: int
    object_class: str
    detection_confidence: float
    position_horizontal: str  # "LEFT", "CENTER", "RIGHT"
    estimated_distance: float
    is_moving: bool
    movement_direction: str  # "STATIONARY", "APPROACHING", "RECEDING", "LATERAL_LEFT", "LATERAL_RIGHT"
    intrinsic_risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    is_passable: bool
    occupancy_ratio: float
    
    # Attributs internes pour le traitement
    class_id: int = 0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    center: Tuple[float, float] = (0.0, 0.0)
    area: float = 0.0
    stability_score: float = 0.0
    velocity: Tuple[float, float] = (0.0, 0.0)


class YOLOObjectFilter:
    """
    Système de détection et filtrage d'objets YOLO
    Traite 3 frames consécutives pour une détection stable
    """
    
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.3,
        stability_threshold: float = 0.6,
        frame_width: int = 640,
        frame_height: int = 480
    ):
        """
        Initialise le système de filtrage
        
        Args:
            model_path: Chemin vers le modèle YOLO
            confidence_threshold: Seuil de confiance minimum
            iou_threshold: Seuil IoU pour la correspondance entre frames
            stability_threshold: Score minimum pour considérer un objet stable
            frame_width: Largeur des frames
            frame_height: Hauteur des frames
        """
        print(f"Chargement du modèle YOLO: {model_path}")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.stability_threshold = stability_threshold
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.object_counter = 0
        
        # Seuils de vitesse pour déterminer le mouvement
        self.movement_threshold = 3.0  # pixels par frame
        
        # Classes d'objets considérées comme passables
        self.passable_classes = {
            'traffic light', 'stop sign', 'parking meter', 'bench',
            'fire hydrant', 'bird', 'cat', 'dog', 'potted plant'
        }
        
        # Niveaux de risque intrinsèque par classe
        self.risk_levels = {
            'person': 'CRITICAL',
            'bicycle': 'HIGH',
            'car': 'HIGH',
            'motorcycle': 'HIGH',
            'bus': 'CRITICAL',
            'truck': 'CRITICAL',
            'train': 'CRITICAL',
            'traffic light': 'LOW',
            'stop sign': 'MEDIUM',
            'dog': 'MEDIUM',
            'cat': 'LOW',
            'bird': 'LOW'
        }
    
    def _horizontal_position(self, bbox: Tuple[int, int, int, int]) -> str:
        """
        Détermine la position horizontale de l'objet
        
        Args:
            bbox: Boîte englobante (x, y, w, h)
            
        Returns:
            "LEFT", "CENTER", ou "RIGHT"
        """
        x, y, w, h = bbox
        center_x = x + w / 2
        
        # Diviser l'image en 3 zones
        left_boundary = self.frame_width / 3
        right_boundary = 2 * self.frame_width / 3
        
        if center_x < left_boundary:
            return "LEFT"
        elif center_x > right_boundary:
            return "RIGHT"
        else:
            return "CENTER"
    
    def _estimate_distance(self, bbox: Tuple[int, int, int, int], class_name: str) -> float:
        """
        Estime la distance de l'objet basée sur sa taille apparente
        
        Args:
            bbox: Boîte englobante (x, y, w, h)
            class_name: Nom de la classe de l'objet
            
        Returns:
            Distance estimée en mètres
        """
        x, y, w, h = bbox
        
        # Tailles réelles moyennes (en mètres)
        real_sizes = {
            'person': 1.7,
            'car': 4.5,
            'truck': 8.0,
            'bicycle': 1.8,
            'motorcycle': 2.0,
            'bus': 12.0,
            'traffic light': 2.5,
            'stop sign': 0.9,
            'dog': 0.6,
            'cat': 0.3
        }
        
        # Paramètre de calibration de la caméra
        focal_length = 800  # pixels (à calibrer selon votre caméra)
        
        real_size = real_sizes.get(class_name, 2.0)
        
        # Distance = (Taille_réelle * Focale) / Taille_apparente
        if h > 0:
            distance = (real_size * focal_length) / h
        else:
            distance = 100.0
        
        return min(max(distance, 0.5), 100.0)  # Limiter entre 0.5m et 100m
    
    def _determine_movement_direction(self, velocity: Tuple[float, float]) -> str:
        """
        Détermine la direction du mouvement
        
        Args:
            velocity: Vélocité (dx, dy)
            
        Returns:
            Direction du mouvement
        """
        dx, dy = velocity
        speed = math.sqrt(dx**2 + dy**2)
        
        if speed < self.movement_threshold:
            return "STATIONARY"
        
        # Déterminer la direction principale
        if abs(dy) > abs(dx):
            # Mouvement vertical dominant
            if dy > 0:
                return "APPROACHING"  # Se rapproche (vers le bas de l'image)
            else:
                return "RECEDING"  # S'éloigne (vers le haut de l'image)
        else:
            # Mouvement horizontal dominant
            if dx > 0:
                return "LATERAL_RIGHT"
            else:
                return "LATERAL_LEFT"
    
    def _calculate_intrinsic_risk(self, class_name: str) -> str:
        """
        Calcule le niveau de risque intrinsèque de l'objet
        
        Args:
            class_name: Nom de la classe
            
        Returns:
            Niveau de risque: "LOW", "MEDIUM", "HIGH", "CRITICAL"
        """
        return self.risk_levels.get(class_name, 'MEDIUM')
    
    def _is_passable(self, class_name: str, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Détermine si l'objet est passable
        
        Args:
            class_name: Nom de la classe
            bbox: Boîte englobante
            
        Returns:
            True si l'objet est passable, False sinon
        """
        # Objets petits et fixes sont généralement passables
        if class_name in self.passable_classes:
            return True
        
        # Objets qui occupent peu d'espace peuvent être passables
        x, y, w, h = bbox
        occupancy = (w * h) / (self.frame_width * self.frame_height)
        
        if occupancy < 0.05:  # Moins de 5% de l'image
            return True
        
        return False
    
    def _calculate_occupancy_ratio(self, bbox: Tuple[int, int, int, int]) -> float:
        """
        Calcule le ratio d'occupation de l'objet dans l'image
        
        Args:
            bbox: Boîte englobante (x, y, w, h)
            
        Returns:
            Ratio d'occupation (0.0 à 1.0)
        """
        x, y, w, h = bbox
        object_area = w * h
        frame_area = self.frame_width * self.frame_height
        
        return object_area / frame_area if frame_area > 0 else 0.0
        
    def process_frames(
        self,
        frames: List[np.ndarray],
        reference_frame_index: int = 2
    ) -> List[StableObject]:
        """
        Traite 3 frames consécutives et retourne les objets stables
        
        Args:
            frames: Liste de 3 frames (t-2, t-1, t) ou (t, t+1, t+2)
            reference_frame_index: Index du frame de référence (0, 1, ou 2)
            
        Returns:
            Liste des objets stables détectés
        """
        if len(frames) != 3:
            raise ValueError("Exactement 3 frames sont requis")
        
        # Mettre à jour les dimensions si nécessaire
        if frames[0] is not None:
            self.frame_height, self.frame_width = frames[0].shape[:2]
        
        print("\n" + "="*60)
        print("DÉBUT DU TRAITEMENT DE DÉTECTION")
        print("="*60)
        
        # Étape 1: Détection YOLO sur chaque frame
        print("\nÉtape 1: Détection YOLO sur les 3 frames...")
        all_detections = self._detect_objects_all_frames(frames)
        
        # Étape 2: Extraction des détections du frame de référence
        print(f"\nÉtape 2: Extraction des objets du frame de référence (frame {reference_frame_index})...")
        reference_detections = all_detections[reference_frame_index]
        print(f"  → {len(reference_detections)} objets détectés dans le frame de référence")
        
        # Étape 3: Vérification de la présence dans les autres frames
        print("\nÉtape 3: Vérification de la présence dans les autres frames...")
        stable_objects = self._verify_and_filter(
            reference_detections,
            all_detections,
            reference_frame_index
        )
        
        # Étape 4: Calcul des attributs avancés
        print("\nÉtape 4: Calcul des attributs avancés...")
        self._compute_advanced_attributes(stable_objects, all_detections, reference_frame_index)
        
        print("\n" + "="*60)
        print(f"RÉSULTAT FINAL: {len(stable_objects)} objets stables détectés")
        print("="*60 + "\n")
        
        return stable_objects
    
    def _detect_objects_all_frames(
        self,
        frames: List[np.ndarray]
    ) -> List[List[Detection]]:
        """
        Lance YOLO sur chaque frame séparément
        
        Returns:
            Liste de listes de détections (une liste par frame)
        """
        all_detections = []
        
        for i, frame in enumerate(frames):
            print(f"  Frame {i}: Détection en cours...")
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            detections = self._extract_detections(results[0], i)
            print(f"  Frame {i}: {len(detections)} objets détectés")
            all_detections.append(detections)
        
        return all_detections
    
    def _extract_detections(
        self,
        result,
        frame_index: int
    ) -> List[Detection]:
        """
        Extrait les détections d'un résultat YOLO
        """
        detections = []
        
        if result.boxes is not None:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Extraction des données
                box = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = result.names[cls_id]
                
                # Calcul des attributs
                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                area = w * h
                
                detection = Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=(x1, y1, w, h),
                    center=(center_x, center_y),
                    area=area,
                    frame_index=frame_index
                )
                
                detections.append(detection)
        
        return detections
    
    def _calculate_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """
        Calcule l'Intersection over Union (IoU) entre deux bounding boxes
        """
        x1_1, y1_1, w1, h1 = bbox1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = bbox2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Union
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area
        
        iou = intersection_area / union_area if union_area > 0 else 0.0
        return iou
    
    def _find_matching_detection(
        self,
        reference: Detection,
        candidates: List[Detection]
    ) -> Optional[Detection]:
        """
        Trouve la détection correspondante dans une liste de candidats
        """
        best_match = None
        best_iou = 0.0
        
        for candidate in candidates:
            # Vérifier que c'est la même classe
            if candidate.class_id != reference.class_id:
                continue
            
            # Calculer IoU
            iou = self._calculate_iou(reference.bbox, candidate.bbox)
            
            if iou > best_iou and iou >= self.iou_threshold:
                best_iou = iou
                best_match = candidate
        
        return best_match
    
    def _verify_and_filter(
        self,
        reference_detections: List[Detection],
        all_detections: List[List[Detection]],
        reference_index: int
    ) -> List[StableObject]:
        """
        Vérifie la présence des objets de référence dans les autres frames
        et filtre les détections instables
        """
        stable_objects = []
        other_indices = [i for i in range(3) if i != reference_index]
        
        for ref_det in reference_detections:
            matches = [ref_det]  # Inclure la détection de référence
            match_found_in_all = True
            
            # Chercher des correspondances dans les autres frames
            for idx in other_indices:
                match = self._find_matching_detection(ref_det, all_detections[idx])
                
                if match is not None:
                    matches.append(match)
                else:
                    match_found_in_all = False
                    break
            
            # Si l'objet est présent dans tous les frames
            if match_found_in_all and len(matches) >= 2:
                # Calculer la confiance moyenne
                avg_confidence = np.mean([m.confidence for m in matches])
                
                # Calculer le score de stabilité
                stability_score = self._calculate_stability_score(matches)
                
                # Filtrer les objets avec faible stabilité
                if (stability_score >= self.stability_threshold and 
                    avg_confidence >= self.confidence_threshold):
                    
                    # Calculer la vélocité préliminaire
                    if len(matches) >= 2:
                        dx = matches[-1].center[0] - matches[0].center[0]
                        dy = matches[-1].center[1] - matches[0].center[1]
                        velocity = (dx / 2, dy / 2)
                    else:
                        velocity = (0.0, 0.0)
                    
                    # Générer un ID unique
                    self.object_counter += 1
                    
                    # Calculer les attributs requis
                    position_horizontal = self._horizontal_position(ref_det.bbox)
                    estimated_distance = self._estimate_distance(ref_det.bbox, ref_det.class_name)
                    
                    speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
                    is_moving = speed >= self.movement_threshold
                    
                    movement_direction = self._determine_movement_direction(velocity)
                    intrinsic_risk_level = self._calculate_intrinsic_risk(ref_det.class_name)
                    is_passable = self._is_passable(ref_det.class_name, ref_det.bbox)
                    occupancy_ratio = self._calculate_occupancy_ratio(ref_det.bbox)
                    
                    stable_obj = StableObject(
                        object_id=self.object_counter,
                        object_class=ref_det.class_name,
                        detection_confidence=avg_confidence,
                        position_horizontal=position_horizontal,
                        estimated_distance=estimated_distance,
                        is_moving=is_moving,
                        movement_direction=movement_direction,
                        intrinsic_risk_level=intrinsic_risk_level,
                        is_passable=is_passable,
                        occupancy_ratio=occupancy_ratio,
                        class_id=ref_det.class_id,
                        bbox=ref_det.bbox,
                        center=ref_det.center,
                        area=ref_det.area,
                        stability_score=stability_score,
                        velocity=velocity
                    )
                    
                    stable_objects.append(stable_obj)
                    
                    print(f"  ✓ Objet stable: {ref_det.class_name} "
                          f"(conf: {avg_confidence:.2f}, stabilité: {stability_score:.2f})")
                else:
                    print(f"  ✗ Objet rejeté: {ref_det.class_name} "
                          f"(stabilité: {stability_score:.2f} < {self.stability_threshold})")
            else:
                print(f"  ✗ Objet instable: {ref_det.class_name} "
                      f"(présent dans {len(matches)}/3 frames)")
        
        return stable_objects
    
    def _calculate_stability_score(
        self,
        matches: List[Detection]
    ) -> float:
        """
        Calcule un score de stabilité basé sur la variance des détections
        """
        if len(matches) < 2:
            return 0.0
        
        # Variance de la confiance
        confidences = [m.confidence for m in matches]
        conf_variance = np.var(confidences)
        conf_score = 1.0 - min(conf_variance, 1.0)
        
        # Variance de la position
        centers = [m.center for m in matches]
        center_x = [c[0] for c in centers]
        center_y = [c[1] for c in centers]
        
        pos_variance = np.var(center_x) + np.var(center_y)
        pos_score = 1.0 / (1.0 + pos_variance / 1000.0)  # Normalisation
        
        # Variance de la taille
        areas = [m.area for m in matches]
        area_variance = np.var(areas)
        mean_area = np.mean(areas)
        area_score = 1.0 - min(area_variance / (mean_area + 1e-6), 1.0)
        
        # Score combiné
        stability_score = (conf_score * 0.3 + pos_score * 0.4 + area_score * 0.3)
        
        return stability_score
    
    def _compute_advanced_attributes(
        self,
        stable_objects: List[StableObject],
        all_detections: List[List[Detection]],
        reference_index: int
    ) -> None:
        """
        Calcule et met à jour les attributs avancés (vélocité raffinée, etc.)
        """
        for obj in stable_objects:
            # Trouver les détections correspondantes dans tous les frames
            detections_over_time = []
            
            for frame_detections in all_detections:
                for det in frame_detections:
                    if (det.class_id == obj.class_id and 
                        self._calculate_iou(obj.bbox, det.bbox) >= self.iou_threshold):
                        detections_over_time.append(det)
                        break
            
            # Recalculer la vélocité avec plus de précision si nécessaire
            if len(detections_over_time) >= 2:
                dx = detections_over_time[-1].center[0] - detections_over_time[0].center[0]
                dy = detections_over_time[-1].center[1] - detections_over_time[0].center[1]
                obj.velocity = (dx / 2, dy / 2)
                
                # Mettre à jour is_moving et movement_direction
                speed = math.sqrt(dx**2 + dy**2) / 2
                obj.is_moving = speed >= self.movement_threshold
                obj.movement_direction = self._determine_movement_direction(obj.velocity)
            
            print(f"  {obj.object_class}: "
                  f"pos={obj.position_horizontal}, "
                  f"dist={obj.estimated_distance:.1f}m, "
                  f"moving={obj.is_moving}, "
                  f"dir={obj.movement_direction}, "
                  f"risk={obj.intrinsic_risk_level}, "
                  f"passable={obj.is_passable}, "
                  f"occupancy={obj.occupancy_ratio:.2%}")
    
    def visualize_results(
        self,
        frame: np.ndarray,
        stable_objects: List[StableObject],
        show: bool = True
    ) -> np.ndarray:
        """
        Visualise les objets stables détectés sur un frame
        """
        output = frame.copy()
        
        for obj in stable_objects:
            x, y, w, h = obj.bbox
            
            # Couleur basée sur le niveau de risque
            risk_colors = {
                'LOW': (0, 255, 0),      # Vert
                'MEDIUM': (0, 255, 255),  # Jaune
                'HIGH': (0, 165, 255),    # Orange
                'CRITICAL': (0, 0, 255)   # Rouge
            }
            color = risk_colors.get(obj.intrinsic_risk_level, (255, 255, 255))
            
            # Dessiner le rectangle
            thickness = 3 if obj.is_moving else 2
            cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
            
            # Préparer le texte
            label = f"ID:{obj.object_id} {obj.object_class} {obj.detection_confidence:.2f}"
            info1 = f"{obj.position_horizontal} {obj.estimated_distance:.1f}m {obj.intrinsic_risk_level}"
            info2 = f"{'MOVING' if obj.is_moving else 'STATIC'} {obj.movement_direction}"
            
            # Fond pour le texte
            cv2.rectangle(output, (x, y - 70), (x + 250, y), color, -1)
            
            # Texte
            cv2.putText(output, label, (x + 5, y - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(output, info1, (x + 5, y - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(output, info2, (x + 5, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Flèche de vélocité si en mouvement
            if obj.is_moving and obj.velocity != (0.0, 0.0):
                cx, cy = obj.center
                end_x = int(cx + obj.velocity[0] * 5)
                end_y = int(cy + obj.velocity[1] * 5)
                cv2.arrowedLine(output, (int(cx), int(cy)), (end_x, end_y),
                              (0, 255, 255), 2, tipLength=0.3)
        
        if show:
            cv2.imshow('Détections Stables', output)
            cv2.waitKey(1)
        
        return output




