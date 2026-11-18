"""
Module Obstacle Detail - Optimisé pour YOLO COCO (80 classes)
==============================================================

Ce module fournit les fonctionnalités pour analyser, filtrer et détailler
les obstacles détectés par YOLO pour un système de guidage des aveugles.

Fonctionnalités:
- Support complet des 80 classes COCO de YOLO
- Classification intelligente par priorité de danger
- Filtrage contextuel (obstacles statiques vs dynamiques)
- Estimation de trajectoire et collision potentielle
- Support natif Ultralytics YOLO
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union, Set
from enum import Enum, auto
import math
import time
from collections import defaultdict


class HorizontalPosition(Enum):
    """Positions horizontales relatives dans le champ de vision."""
    FAR_LEFT = "FAR_LEFT"      # Extrême gauche (évitable)
    LEFT = "LEFT"              # Gauche
    CENTER_LEFT = "CENTER_LEFT"  # Centre-gauche
    CENTER = "CENTER"          # Centre (trajectoire directe)
    CENTER_RIGHT = "CENTER_RIGHT"  # Centre-droite
    RIGHT = "RIGHT"            # Droite
    FAR_RIGHT = "FAR_RIGHT"    # Extrême droite (évitable)


class MovementDirection(Enum):
    """Directions de mouvement possibles d'un obstacle."""
    STATIONARY = "STATIONARY"
    APPROACHING = "APPROACHING"
    RECEDING = "RECEDING"
    CROSSING_LEFT = "CROSSING_LEFT"      # Traverse de droite à gauche
    CROSSING_RIGHT = "CROSSING_RIGHT"    # Traverse de gauche à droite
    UNKNOWN = "UNKNOWN"


class RiskLevel(Enum):
    """Niveaux de risque avec valeurs numériques pour comparaison."""
    NONE = 0           # Pas un obstacle (ex: toilettes)
    LOW = 1            # Risque minimal
    MEDIUM = 2         # Attention requise
    HIGH = 3           # Danger significatif
    CRITICAL = 4       # Danger immédiat


class ObstacleCategory(Enum):
    """Catégories d'obstacles pour le système de guidage."""
    PEDESTRIAN = auto()          # Piétons (humains)
    VEHICLE = auto()             # Véhicules motorisés
    CYCLIST = auto()             # Vélos/motos avec conducteur
    ANIMAL = auto()              # Animaux
    STATIC_HAZARD = auto()       # Obstacles fixes dangereux
    STATIC_PASSABLE = auto()     # Obstacles fixes contournables
    NAVIGATION_AID = auto()      # Aides à la navigation (feux, panneaux)
    BACKGROUND = auto()          # Objets non pertinents pour la mobilité


# ============================================================================
# CONFIGURATION COMPLÈTE DES 80 CLASSES COCO YOLO
# ============================================================================

# Mapping des classes COCO vers catégories et niveaux de risque
# Format: (catégorie, risque_base, passable, taille_typique_m, description_guidage)
COCO_OBSTACLE_CONFIG: Dict[str, Tuple[ObstacleCategory, RiskLevel, bool, float, str]] = {
    # === PERSONNES (0) ===
    "person": (ObstacleCategory.PEDESTRIAN, RiskLevel.CRITICAL, True, 1.7, 
               "Personne détectée"),
    
    # === ANIMAUX (1-10) ===
    "bird": (ObstacleCategory.ANIMAL, RiskLevel.LOW, True, 0.2,
             "Oiseau à proximité"),
    "cat": (ObstacleCategory.ANIMAL, RiskLevel.MEDIUM, True, 0.3,
            "Chat détecté"),
    "dog": (ObstacleCategory.ANIMAL, RiskLevel.MEDIUM, True, 0.6,
            "Chien détecté"),
    "horse": (ObstacleCategory.ANIMAL, RiskLevel.HIGH, False, 1.6,
              "Cheval, danger potentiel"),
    "sheep": (ObstacleCategory.ANIMAL, RiskLevel.LOW, True, 0.8,
              "Mouton sur le passage"),
    "cow": (ObstacleCategory.ANIMAL, RiskLevel.MEDIUM, False, 1.4,
            "Vache, contourner avec précaution"),
    "elephant": (ObstacleCategory.ANIMAL, RiskLevel.CRITICAL, False, 3.0,
                 "Éléphant, danger majeur"),
    "bear": (ObstacleCategory.ANIMAL, RiskLevel.CRITICAL, False, 2.0,
             "Ours, danger extrême"),
    "zebra": (ObstacleCategory.ANIMAL, RiskLevel.MEDIUM, True, 1.4,
              "Zèbre sur le passage"),
    "giraffe": (ObstacleCategory.ANIMAL, RiskLevel.MEDIUM, False, 4.0,
                "Girafe, attention hauteur"),
    
    # === VÉHICULES & TRANSPORTS (11-23) ===
    "bicycle": (ObstacleCategory.CYCLIST, RiskLevel.HIGH, True, 1.0,
                "Vélo en mouvement"),
    "car": (ObstacleCategory.VEHICLE, RiskLevel.CRITICAL, False, 1.5,
            "Voiture, danger de collision"),
    "motorcycle": (ObstacleCategory.CYCLIST, RiskLevel.CRITICAL, True, 1.2,
                   "Moto rapide, attention"),
    "airplane": (ObstacleCategory.BACKGROUND, RiskLevel.NONE, True, 0.0,
                 "Avion en arrière-plan"),
    "bus": (ObstacleCategory.VEHICLE, RiskLevel.CRITICAL, False, 2.5,
            "Bus, grand véhicule"),
    "train": (ObstacleCategory.VEHICLE, RiskLevel.CRITICAL, False, 3.0,
              "Train, danger extrême"),
    "truck": (ObstacleCategory.VEHICLE, RiskLevel.CRITICAL, False, 2.8,
              "Camion, lourd et dangereux"),
    "boat": (ObstacleCategory.BACKGROUND, RiskLevel.NONE, True, 0.0,
             "Bateau en arrière-plan"),
    "traffic light": (ObstacleCategory.NAVIGATION_AID, RiskLevel.MEDIUM, True, 3.0,
                      "Feu tricolore, attendre le vert"),
    "fire hydrant": (ObstacleCategory.STATIC_PASSABLE, RiskLevel.LOW, True, 0.8,
                     "Borne hydrante"),
    "stop sign": (ObstacleCategory.NAVIGATION_AID, RiskLevel.MEDIUM, True, 2.5,
                  "Panneau stop, intersection"),
    "parking meter": (ObstacleCategory.STATIC_PASSABLE, RiskLevel.LOW, True, 1.2,
                      "Parcmètre"),
    "bench": (ObstacleCategory.STATIC_PASSABLE, RiskLevel.LOW, True, 0.8,
              "Banc, possible s'asseoir"),
    
    # === OBJETS DU QUOTIDIEN (24-31) ===
    "backpack": (ObstacleCategory.STATIC_PASSABLE, RiskLevel.LOW, True, 0.5,
                 "Sac à dos au sol"),
    "umbrella": (ObstacleCategory.STATIC_PASSABLE, RiskLevel.LOW, True, 1.0,
                 "Parapluie ouvert"),
    "handbag": (ObstacleCategory.STATIC_PASSABLE, RiskLevel.LOW, True, 0.4,
                "Sac à main abandonné"),
    "tie": (ObstacleCategory.BACKGROUND, RiskLevel.NONE, True, 0.0,
            "Non pertinent"),
    "suitcase": (ObstacleCategory.STATIC_PASSABLE, RiskLevel.LOW, True, 0.6,
                 "Valise sur le passage"),
    
    # === SPORTS & LOISIRS (32-43) ===
    "frisbee": (ObstacleCategory.BACKGROUND, RiskLevel.NONE, True, 0.0,
                "Objet en mouvement"),
    "skis": (ObstacleCategory.STATIC_PASSABLE, RiskLevel.LOW, True, 1.5,
             "Skis contre un mur"),
    "snowboard": (ObstacleCategory.STATIC_PASSABLE, RiskLevel.LOW, True, 1.5,
                  "Snowboard"),
    "sports ball": (ObstacleCategory.BACKGROUND, RiskLevel.LOW, True, 0.2,
                    "Balle, risque de glissade"),
    "kite": (ObstacleCategory.BACKGROUND, RiskLevel.NONE, True, 0.0,
             "Cerf-volant en l'air"),
    "baseball bat": (ObstacleCategory.STATIC_PASSABLE, RiskLevel.LOW, True, 0.8,
                     "Batte de baseball"),
    "baseball glove": (ObstacleCategory.STATIC_PASSABLE, RiskLevel.LOW, True, 0.3,
                       "Gant de baseball"),
    "skateboard": (ObstacleCategory.STATIC_PASSABLE, RiskLevel.MEDIUM, True, 0.2,
                   "Skateboard au sol, risque de glissade"),
    "surfboard": (ObstacleCategory.STATIC_PASSABLE, RiskLevel.LOW, True, 2.0,
                  "Planche de surf"),
    "tennis racket": (ObstacleCategory.STATIC_PASSABLE, RiskLevel.LOW, True, 0.7,
                      "Raquette de tennis"),
    
    # === ALIMENTATION & CUISINE (44-61) ===
    "bottle": (ObstacleCategory.STATIC_PASSABLE, RiskLevel.LOW, True, 0.3,
               "Bouteille au sol"),
    "wine glass": (ObstacleCategory.STATIC_PASSABLE, RiskLevel.LOW, True, 0.15,
                   "Verre cassé possible"),
    "cup": (ObstacleCategory.STATIC_PASSABLE, RiskLevel.LOW, True, 0.1,
            "Tasse"),
    "fork": (ObstacleCategory.BACKGROUND, RiskLevel.NONE, True, 0.0,
             "Non pertinent"),
    "knife": (ObstacleCategory.STATIC_PASSABLE, RiskLevel.MEDIUM, True, 0.2,
              "Couteau au sol, danger"),
    "spoon": (ObstacleCategory.BACKGROUND, RiskLevel.NONE, True, 0.0,
              "Non pertinent"),
    "bowl": (ObstacleCategory.STATIC_PASSABLE, RiskLevel.LOW, True, 0.1,
             "Bol"),
    "banana": (ObstacleCategory.BACKGROUND, RiskLevel.LOW, True, 0.2,
               "Peau de banane, risque de glissade"),
    "apple": (ObstacleCategory.BACKGROUND, RiskLevel.NONE, True, 0.0,
              "Non pertinent"),
    "sandwich": (ObstacleCategory.BACKGROUND, RiskLevel.NONE, True, 0.0,
                 "Non pertinent"),
    "orange": (ObstacleCategory.BACKGROUND, RiskLevel.NONE, True, 0.0,
               "Non pertinent"),
    "broccoli": (ObstacleCategory.BACKGROUND, RiskLevel.NONE, True, 0.0,
                 "Non pertinent"),
    "carrot": (ObstacleCategory.BACKGROUND, RiskLevel.NONE, True, 0.0,
               "Non pertinent"),
    "hot dog": (ObstacleCategory.BACKGROUND, RiskLevel.NONE, True, 0.0,
                "Non pertinent"),
    "pizza": (ObstacleCategory.BACKGROUND, RiskLevel.NONE, True, 0.0,
              "Non pertinent"),
    "donut": (ObstacleCategory.BACKGROUND, RiskLevel.NONE, True, 0.0,
              "Non pertinent"),
    "cake": (ObstacleCategory.BACKGROUND, RiskLevel.NONE, True, 0.0,
             "Non pertinent"),
    
    # === MAISON & MOBILIER (62-69) ===
    "chair": (ObstacleCategory.STATIC_PASSABLE, RiskLevel.LOW, True, 0.9,
              "Chaise sur le passage"),
    "couch": (ObstacleCategory.STATIC_HAZARD, RiskLevel.MEDIUM, False, 0.8,
              "Canapé, obstacle large"),
    "potted plant": (ObstacleCategory.STATIC_PASSABLE, RiskLevel.LOW, True, 0.6,
                     "Plante en pot"),
    "bed": (ObstacleCategory.STATIC_HAZARD, RiskLevel.MEDIUM, False, 0.6,
            "Lit, obstacle intérieur"),
    "dining table": (ObstacleCategory.STATIC_HAZARD, RiskLevel.MEDIUM, False, 0.8,
                     "Table à manger"),
    "toilet": (ObstacleCategory.STATIC_PASSABLE, RiskLevel.LOW, True, 0.8,
               "Toilettes"),
    "tv": (ObstacleCategory.BACKGROUND, RiskLevel.NONE, True, 0.0,
           "Télévision en arrière-plan"),
    "laptop": (ObstacleCategory.BACKGROUND, RiskLevel.NONE, True, 0.0,
               "Ordinateur portable"),
    
    # === ÉLECTRONIQUE (70-73) ===
    "mouse": (ObstacleCategory.BACKGROUND, RiskLevel.NONE, True, 0.0,
              "Souris d'ordinateur"),
    "remote": (ObstacleCategory.BACKGROUND, RiskLevel.NONE, True, 0.0,
               "Télécommande"),
    "keyboard": (ObstacleCategory.BACKGROUND, RiskLevel.NONE, True, 0.0,
                 "Clavier"),
    "cell phone": (ObstacleCategory.BACKGROUND, RiskLevel.NONE, True, 0.0,
                   "Téléphone portable"),
    
    # === OBJETS DIVERS (74-79) ===
    "microwave": (ObstacleCategory.BACKGROUND, RiskLevel.NONE, True, 0.0,
                  "Micro-ondes"),
    "oven": (ObstacleCategory.STATIC_HAZARD, RiskLevel.MEDIUM, False, 0.9,
             "Four, chaud possible"),
    "toaster": (ObstacleCategory.BACKGROUND, RiskLevel.NONE, True, 0.0,
                "Grille-pain"),
    "sink": (ObstacleCategory.STATIC_PASSABLE, RiskLevel.LOW, True, 0.5,
             "Évier"),
    "refrigerator": (ObstacleCategory.STATIC_HAZARD, RiskLevel.LOW, False, 1.8,
                     "Réfrigérateur"),
    "book": (ObstacleCategory.BACKGROUND, RiskLevel.NONE, True, 0.0,
             "Livre"),
    "clock": (ObstacleCategory.BACKGROUND, RiskLevel.NONE, True, 0.0,
              "Horloge"),
    "vase": (ObstacleCategory.STATIC_PASSABLE, RiskLevel.LOW, True, 0.4,
             "Vase"),
    "scissors": (ObstacleCategory.STATIC_PASSABLE, RiskLevel.MEDIUM, True, 0.2,
                 "Ciseaux, pointus"),
    "teddy bear": (ObstacleCategory.STATIC_PASSABLE, RiskLevel.NONE, True, 0.4,
                   "Ours en peluche"),
    "hair drier": (ObstacleCategory.BACKGROUND, RiskLevel.NONE, True, 0.0,
                   "Sèche-cheveux"),
    "toothbrush": (ObstacleCategory.BACKGROUND, RiskLevel.NONE, True, 0.0,
                   "Brosse à dents"),
}

# Classes à ignorer complètement (pas des obstacles pour la mobilité)
IGNORED_CLASSES: Set[str] = {
    "airplane", "boat", "tie", "frisbee", "kite", "fork", "spoon", 
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", 
    "pizza", "donut", "cake", "tv", "laptop", "mouse", "remote", 
    "keyboard", "cell phone", "microwave", "toaster", "book", "clock",
    "hair drier", "toothbrush"
}

# Classes prioritaires (alerte immédiate)
PRIORITY_CLASSES: Set[str] = {
    "person", "car", "motorcycle", "bus", "truck", "train", 
    "bicycle", "dog", "bear", "elephant"
}

# Seuils de distance critiques par catégorie (mètres)
DISTANCE_THRESHOLDS = {
    ObstacleCategory.PEDESTRIAN: {"critical": 5.0, "warning": 10.0},
    ObstacleCategory.VEHICLE: {"critical": 10.0, "warning": 20.0},
    ObstacleCategory.CYCLIST: {"critical": 8.0, "warning": 15.0},
    ObstacleCategory.ANIMAL: {"critical": 4.0, "warning": 8.0},
    ObstacleCategory.STATIC_HAZARD: {"critical": 2.0, "warning": 4.0},
    ObstacleCategory.STATIC_PASSABLE: {"critical": 1.5, "warning": 3.0},
    ObstacleCategory.NAVIGATION_AID: {"critical": 3.0, "warning": 6.0},
    ObstacleCategory.BACKGROUND: {"critical": 0.0, "warning": 0.0},
}


@dataclass
class ObstacleCharacteristics:
    """
    Structure complète pour un obstacle détecté et analysé.
    """
    object_id: int
    object_class: str
    category: str
    detection_confidence: float
    position_horizontal: str
    estimated_distance: float
    is_moving: bool
    movement_direction: str
    intrinsic_risk_level: str
    risk_value: int  # Valeur numérique pour tri
    is_passable: bool
    occupancy_ratio: float
    collision_risk: float  # Probabilité de collision 0-1
    guidance_message: str  # Message pour synthèse vocale
    
    # Champs optionnels
    bbox: Optional[List[float]] = field(default=None)
    velocity: Optional[float] = field(default=None)
    timestamp: Optional[float] = field(default=None)
    track_id: Optional[int] = field(default=None)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour le système de décision."""
        return {
            "object_id": self.object_id,
            "object_class": self.object_class,
            "category": self.category,
            "detection_confidence": self.detection_confidence,
            "position_horizontal": self.position_horizontal,
            "estimated_distance": self.estimated_distance,
            "is_moving": self.is_moving,
            "movement_direction": self.movement_direction,
            "intrinsic_risk_level": self.intrinsic_risk_level,
            "risk_value": self.risk_value,
            "is_passable": self.is_passable,
            "occupancy_ratio": self.occupancy_ratio,
            "collision_risk": self.collision_risk,
            "guidance_message": self.guidance_message,
        }
    
    def to_detailed_dict(self) -> Dict[str, Any]:
        """Version complète avec tous les champs."""
        result = self.to_dict()
        if self.bbox is not None:
            result["bbox"] = self.bbox
        if self.velocity is not None:
            result["velocity"] = self.velocity
        if self.timestamp is not None:
            result["timestamp"] = self.timestamp
        if self.track_id is not None:
            result["track_id"] = self.track_id
        return result


class ObstacleDetail:
    """
    Processeur optimisé pour l'analyse d'obstacles YOLO dans un contexte 
    d'assistance aux personnes aveugles.
    """
    
    def __init__(
        self,
        max_distance: float = 30.0,          # Réduit pour focus proche
        min_confidence: float = 0.50,         # Seuil plus bas pour sécurité
        image_width: int = 640,
        image_height: int = 480,
        center_zone_ratio: float = 0.4,       # Zone centrale élargie
        distance_history_size: int = 5,
        movement_threshold_mps: float = 0.3,
        approaching_threshold: float = -0.2,
        receding_threshold: float = 0.2,
        focal_length: float = 600.0,          # Paramètre caméra
        enable_tracking: bool = True,
        collision_prediction_time: float = 2.0  # Secondes pour prédiction collision
    ):
        self.max_distance = max_distance
        self.min_confidence = min_confidence
        self.image_width = image_width
        self.image_height = image_height
        self.center_zone_ratio = center_zone_ratio
        self.focal_length = focal_length
        self.collision_prediction_time = collision_prediction_time
        
        # Suivi de mouvement
        self.distance_history_size = distance_history_size
        self.movement_threshold_mps = movement_threshold_mps
        self.approaching_threshold = approaching_threshold
        self.receding_threshold = receding_threshold
        
        # Historique pour tracking
        self._distance_history: Dict[int, List[Tuple[float, float]]] = {}
        self._position_history: Dict[int, List[Tuple[float, float, float]]] = {}  # x, y, t
        self._last_timestamp: float = 0.0
        self._id_counter: int = 0
        self._enable_tracking = enable_tracking
        
        # Cache pour éviter recalculs
        self._class_cache: Dict[str, Tuple] = {}
    
    def _generate_id(self) -> int:
        """Génère ID unique."""
        self._id_counter += 1
        return self._id_counter
    
    def _get_class_config(self, object_class: str) -> Tuple[ObstacleCategory, RiskLevel, bool, float, str]:
        """Récupère config avec cache."""
        normalized = object_class.lower().strip()
        if normalized not in self._class_cache:
            self._class_cache[normalized] = COCO_OBSTACLE_CONFIG.get(
                normalized, 
                (ObstacleCategory.STATIC_PASSABLE, RiskLevel.MEDIUM, True, 1.0, "Obstacle inconnu")
            )
        return self._class_cache[normalized]
    
    def _calculate_horizontal_position(
        self, 
        bbox_center_x: float, 
        image_width: Optional[int] = None
    ) -> str:
        """
        Détermine position horizontale avec 7 zones pour précision guidage.
        """
        width = image_width if image_width is not None else self.image_width
        if width == 0:
            return HorizontalPosition.CENTER.value
        
        # 7 zones: extrême gauche, gauche, centre-gauche, centre, centre-droite, droite, extrême droite
        zone_width = width / 7
        
        if bbox_center_x < zone_width:
            return HorizontalPosition.FAR_LEFT.value
        elif bbox_center_x < 2 * zone_width:
            return HorizontalPosition.LEFT.value
        elif bbox_center_x < 3 * zone_width:
            return HorizontalPosition.CENTER_LEFT.value
        elif bbox_center_x < 4 * zone_width:
            return HorizontalPosition.CENTER.value
        elif bbox_center_x < 5 * zone_width:
            return HorizontalPosition.CENTER_RIGHT.value
        elif bbox_center_x < 6 * zone_width:
            return HorizontalPosition.RIGHT.value
        else:
            return HorizontalPosition.FAR_RIGHT.value
    
    def _estimate_distance(
        self, 
        bbox: List[float], 
        object_class: str,
        depth_map: Optional[Any] = None
    ) -> float:
        """
        Estime distance avec méthodes multiples.
        """
        # 1. Depth map si disponible
        if depth_map is not None:
            try:
                import numpy as np
                x_min, y_min, x_max, y_max = map(int, bbox)
                roi = depth_map[y_min:y_max, x_min:x_max]
                valid_depths = roi[roi > 0]
                if len(valid_depths) > 0:
                    # Percentile 10 pour éviter outliers (bords de l'objet)
                    return float(np.percentile(valid_depths, 10))
            except Exception:
                pass
        
        # 2. Estimation par taille apparente
        _, _, _, typical_height, _ = self._get_class_config(object_class)
        bbox_height = bbox[3] - bbox[1]
        
        if bbox_height > 0 and typical_height > 0:
            distance = (self.focal_length * typical_height) / bbox_height
            return min(distance, self.max_distance * 1.5)
        
        return self.max_distance
    
    def _analyze_movement_advanced(
        self,
        object_id: int,
        current_distance: float,
        current_pos: Tuple[float, float],  # center_x, center_y
        current_timestamp: float,
    ) -> Tuple[bool, str, Optional[float], Optional[float]]:
        """
        Analyse avancée du mouvement avec prédiction de trajectoire.
        """
        if not self._enable_tracking:
            return False, MovementDirection.STATIONARY.value, None, None
        
        # Initialisation historique
        if object_id not in self._distance_history:
            self._distance_history[object_id] = []
            self._position_history[object_id] = []
        
        # Ajout données
        self._distance_history[object_id].append((current_timestamp, current_distance))
        self._position_history[object_id].append((current_pos[0], current_pos[1], current_timestamp))
        
        # Nettoyage anciennes données
        if len(self._distance_history[object_id]) > self.distance_history_size:
            self._distance_history[object_id].pop(0)
            self._position_history[object_id].pop(0)
        
        if len(self._distance_history[object_id]) < 2:
            return False, MovementDirection.UNKNOWN.value, None, None
        
        # Calcul vitesse radiale (distance)
        prev_time, prev_dist = self._distance_history[object_id][-2]
        curr_time, curr_dist = current_timestamp, current_distance
        time_delta = curr_time - prev_time
        
        if time_delta <= 0:
            return False, MovementDirection.STATIONARY.value, None, None
        
        radial_velocity = (curr_dist - prev_dist) / time_delta
        
        # Calcul vitesse latérale
        prev_x, prev_y, _ = self._position_history[object_id][-2]
        curr_x, curr_y = current_pos
        lateral_shift = curr_x - prev_x
        lateral_velocity = lateral_shift / time_delta if time_delta > 0 else 0
        
        # Détection direction
        is_moving = abs(radial_velocity) > self.movement_threshold_mps or \
                   abs(lateral_velocity) > 20  # pixels/sec
        
        if not is_moving:
            return False, MovementDirection.STATIONARY.value, abs(radial_velocity), 0.0
        
        # Classification direction
        if radial_velocity < self.approaching_threshold:
            direction = MovementDirection.APPROACHING
        elif radial_velocity > self.receding_threshold:
            direction = MovementDirection.RECEDING
        elif lateral_velocity < -30:
            direction = MovementDirection.CROSSING_LEFT
        elif lateral_velocity > 30:
            direction = MovementDirection.CROSSING_RIGHT
        else:
            direction = MovementDirection.STATIONARY
        
        return True, direction.value, abs(radial_velocity), lateral_velocity
    
    def _calculate_collision_risk(
        self,
        distance: float,
        velocity: float,
        position: str,
        category: ObstacleCategory,
        is_approaching: bool
    ) -> float:
        """
        Calcule probabilité de collision (0-1).
        """
        if not is_approaching or velocity < 0.5:
            return 0.0
        
        # Temps avant collision estimé
        time_to_collision = distance / velocity if velocity > 0 else float('inf')
        
        # Seuils par catégorie
        thresholds = DISTANCE_THRESHOLDS.get(category, {"critical": 3.0, "warning": 6.0})
        
        # Risque basé sur temps et position
        if time_to_collision < 1.0 and position in [HorizontalPosition.CENTER.value, 
                                                     HorizontalPosition.CENTER_LEFT.value,
                                                     HorizontalPosition.CENTER_RIGHT.value]:
            return 0.9
        elif time_to_collision < 2.0:
            return 0.7 if position in [HorizontalPosition.CENTER.value] else 0.4
        elif time_to_collision < thresholds["critical"] / velocity if velocity > 0 else 0:
            return 0.5
        elif distance < thresholds["warning"]:
            return 0.3
        
        return 0.0
    
    def _generate_guidance_message(
        self,
        object_class: str,
        category: ObstacleCategory,
        distance: float,
        position: str,
        risk: RiskLevel,
        is_moving: bool,
        direction: str,
        collision_risk: float
    ) -> str:
        """
        Génère message de guidage pour synthèse vocale.
        """
        # Messages prioritaires
        if collision_risk > 0.7:
            return f"DANGER IMMÉDIAT: {object_class} approche rapidement, arrêtez-vous!"
        
        if collision_risk > 0.4:
            return f"Attention: {object_class} en approche à {distance:.1f} mètres, ralentissez"
        
        # Messages par catégorie
        if category == ObstacleCategory.VEHICLE:
            if distance < 10:
                return f"Véhicule proche à {distance:.1f} mètres, {position.lower().replace('_', ' ')}"
            return f"Véhicule détecté à {distance:.1f} mètres"
        
        elif category == ObstacleCategory.PEDESTRIAN:
            if distance < 3:
                return f"Personne très proche, {position.lower().replace('_', ' ')}"
            return f"Personne à {distance:.1f} mètres, {position.lower().replace('_', ' ')}"
        
        elif category == ObstacleCategory.CYCLIST:
            return f"Cycliste à {distance:.1f} mètres, attention"
        
        elif category == ObstacleCategory.ANIMAL:
            return f"{object_class} à {distance:.1f} mètres"
        
        elif category == ObstacleCategory.NAVIGATION_AID:
            if "traffic light" in object_class:
                return "Feu tricolore détecté"
            elif "stop" in object_class:
                return "Panneau stop, intersection à venir"
            return f"Aide navigation: {object_class}"
        
        elif category == ObstacleCategory.STATIC_HAZARD:
            return f"Obstacle fixe: {object_class} à {distance:.1f} m"
        
        # Message par défaut
        if distance < 2:
            return f"{object_class} proche"
        return f"{object_class} à {distance:.1f} mètres"
    
    def _filter_detection(
        self, 
        confidence: float, 
        distance: float,
        object_class: str
    ) -> bool:
        """
        Filtre intelligent avec règles par classe.
        """
        # Ignorer classes background lointaines
        if object_class in IGNORED_CLASSES and distance > 5:
            return False
        
        # Confiance minimale
        if confidence < self.min_confidence:
            return False
        
        # Distance maximale
        if distance > self.max_distance:
            return False
        
        # Classes prioritaires: garder même avec confiance plus basse
        if object_class in PRIORITY_CLASSES and confidence > 0.4:
            return True
        
        return True
    
    def process_detection(
        self,
        detection: Dict[str, Any],
        timestamp: float,
        image_shape: Tuple[int, int] = (480, 640),
        depth_map: Optional[Any] = None
    ) -> Optional[ObstacleCharacteristics]:
        """
        Traite une détection unique.
        """
        # Extraction données
        bbox = detection.get("bbox", [0, 0, 0, 0])
        object_class = detection.get("class", "unknown")
        confidence = detection.get("confidence", 0.0)
        
        # Normalisation confiance
        if confidence > 1.0:
            confidence = confidence / 100.0
        
        # Récupération config classe
        category, base_risk, default_passable, typical_size, _ = self._get_class_config(object_class)
        
        # Estimation distance
        distance = self._estimate_distance(bbox, object_class, depth_map)
        
        # Filtre
        if not self._filter_detection(confidence, distance, object_class):
            return None
        
        # ID et tracking
        object_id = detection.get("track_id") or detection.get("id") or self._generate_id()
        
        # Position
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        position = self._calculate_horizontal_position(center_x, image_shape[1])
        
        # Occupation
        image_area = image_shape[0] * image_shape[1]
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        occupancy = min(bbox_area / image_area, 1.0)
        
        # Analyse mouvement
        is_moving, direction, velocity, lateral_vel = self._analyze_movement_advanced(
            object_id, distance, (center_x, center_y), timestamp
        )
        
        # Calcul risque ajusté
        thresholds = DISTANCE_THRESHOLDS.get(category, {"critical": 3.0, "warning": 6.0})
        
        if distance < thresholds["critical"]:
            risk = RiskLevel.CRITICAL if base_risk.value >= RiskLevel.HIGH.value else RiskLevel.HIGH
        elif distance < thresholds["warning"]:
            risk = base_risk
        else:
            # Réduire risque si loin
            risk = RiskLevel(max(base_risk.value - 1, RiskLevel.LOW.value))
        
        # Ajustement confiance
        if confidence < 0.6 and risk.value > RiskLevel.MEDIUM.value:
            risk = RiskLevel(risk.value - 1)
        
        # Passabilité
        is_passable = default_passable
        if distance < 2.0 and occupancy > 0.4:
            is_passable = False
        
        # Risque collision
        collision_risk = self._calculate_collision_risk(
            distance, velocity or 0, position, category, 
            direction == MovementDirection.APPROACHING.value
        )
        
        # Message guidage
        guidance = self._generate_guidance_message(
            object_class, category, distance, position, risk,
            is_moving, direction, collision_risk
        )
        
        return ObstacleCharacteristics(
            object_id=object_id,
            object_class=object_class,
            category=category.name,
            detection_confidence=round(confidence, 3),
            position_horizontal=position,
            estimated_distance=round(distance, 2),
            is_moving=is_moving,
            movement_direction=direction,
            intrinsic_risk_level=risk.name,  # ← CORRIGÉ : .label → .name
            risk_value=risk.value,
            is_passable=is_passable,
            occupancy_ratio=round(occupancy, 4),
            collision_risk=round(collision_risk, 2),
            guidance_message=guidance,
            bbox=bbox,
            velocity=velocity,
            timestamp=timestamp,
            track_id=detection.get("track_id")
        )
    
    def _is_ultralytics_result(self, data) -> bool:
        """Détecte résultats Ultralytics."""
        if isinstance(data, list) and len(data) > 0:
            return hasattr(data[0], 'boxes') or type(data[0]).__name__ == 'Results'
        return hasattr(data, 'boxes') or type(data).__name__ == 'Results'
    
    def _convert_ultralytics_results(self, results) -> Tuple[List[Dict], Tuple[int, int]]:
        """Convertit résultats Ultralytics."""
        if isinstance(results, list):
            result = results[0] if results else None
        else:
            result = results
        
        if result is None:
            return [], (480, 640)
        
        image_shape = getattr(result, 'orig_shape', (480, 640))
        
        if not hasattr(result, 'boxes') or result.boxes is None:
            return [], image_shape
        
        detections = []
        boxes = result.boxes
        
        try:
            import numpy as np
            
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else np.array(boxes.xyxy)
            confs = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') and boxes.conf is not None else []
            cls = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') and boxes.cls is not None else []
            
            ids = None
            if hasattr(boxes, 'id') and boxes.id is not None:
                ids = boxes.id.cpu().numpy() if hasattr(boxes.id, 'cpu') else np.array(boxes.id)
            
            names = getattr(result, 'names', {})
            
            for i in range(len(xyxy)):
                class_id = int(cls[i]) if len(cls) > i else 0
                detection = {
                    "bbox": xyxy[i].tolist(),
                    "confidence": float(confs[i]) if len(confs) > i else 0.0,
                    "class_id": class_id,
                    "class": names.get(class_id, "unknown")
                }
                if ids is not None and len(ids) > i:
                    detection["track_id"] = int(ids[i])
                detections.append(detection)
                
        except Exception as e:
            print(f"Erreur conversion: {e}")
            return [], image_shape
        
        return detections, image_shape
    
    def process_frame(
        self,
        detections: Union[List[Dict[str, Any]], Any],
        timestamp: Optional[float] = None,
        image_shape: Optional[Tuple[int, int]] = None,
        depth_map: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Traite une frame complète.
        """
        if timestamp is None:
            timestamp = time.time()
        self._last_timestamp = timestamp
        
        # Conversion Ultralytics
        if self._is_ultralytics_result(detections):
            detections, auto_shape = self._convert_ultralytics_results(detections)
            if image_shape is None:
                image_shape = auto_shape
        
        if image_shape is None:
            image_shape = (480, 640)
        
        # Nettoyage historique
        self._cleanup_history(timestamp)
        
        # Traitement détections
        obstacles = []
        for det in detections:
            if not isinstance(det, dict):
                continue
            
            obs = self.process_detection(det, timestamp, image_shape, depth_map)
            if obs is not None:
                obstacles.append(obs)
        
        # Tri: risque élevé + proche d'abord
        obstacles.sort(key=lambda x: (-x.risk_value, x.estimated_distance))
        
        # Limite pour performance (garder les 10 plus critiques)
        obstacles = obstacles[:10]
        
        return [obs.to_dict() for obs in obstacles]
    
    def _cleanup_history(self, current_timestamp: float, max_age: float = 3.0):
        """Nettoie historique ancien."""
        to_remove = []
        for obj_id, history in self._distance_history.items():
            if history and (current_timestamp - history[-1][0]) > max_age:
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self._distance_history[obj_id]
            if obj_id in self._position_history:
                del self._position_history[obj_id]
    
    def reset(self):
        """Réinitialise l'état."""
        self._distance_history.clear()
        self._position_history.clear()
        self._id_counter = 0
        self._class_cache.clear()
    
    def get_priority_obstacle(self, obstacles: List[Dict]) -> Optional[Dict]:
        """
        Retourne l'obstacle le plus critique pour alerte immédiate.
        """
        if not obstacles:
            return None
        
        # Cherche collision imminente
        for obs in obstacles:
            if obs.get("collision_risk", 0) > 0.7:
                return obs
        
        # Sinon le plus risqué et proche
        return obstacles[0]


# Fonction utilitaire rapide
def process_detections(
    detections: Union[List[Dict], Any],
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Traite détections sans instancier la classe.
    """
    processor = ObstacleDetail(**{k: v for k, v in kwargs.items() 
                                  if k not in ['timestamp', 'image_shape', 'depth_map']})
    return processor.process_frame(
        detections,
        timestamp=kwargs.get('timestamp'),
        image_shape=kwargs.get('image_shape'),
        depth_map=kwargs.get('depth_map')
    )