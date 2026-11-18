"""
Module Decision Engine - Système de guidage audio pour aveugles
==============================================================

Ce module reçoit les obstacles filtrés et génère des instructions 
audio claires et actionnables pour l'utilisateur.

Entrée: Liste d'obstacles depuis ObstacleDetail (process_frame)
Sortie: Messages audio priorisés et contextuels
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import time
from collections import deque


class InstructionPriority(Enum):
    """Niveaux de priorité des instructions."""
    STOP = 4        # Arrêt immédiat (danger de vie)
    WARNING = 3     # Avertissement urgent (ralentir/changer direction)
    CAUTION = 2     # Attention (être vigilant)
    INFO = 1        # Information (contexte, navigation)
    NONE = 0        # Aucune instruction


class Direction(Enum):
    """Directions de déplacement possibles."""
    LEFT = "gauche"
    RIGHT = "droite"
    CENTER = "devant"
    SLIGHT_LEFT = "légèrement à gauche"
    SLIGHT_RIGHT = "légèrement à droite"


@dataclass
class AudioInstruction:
    """
    Instruction audio complète pour le module de synthèse vocale.
    
    Attributs:
        text: Message à prononcer
        priority: Niveau d'urgence (pour interruption si nécessaire)
        obstacle_type: Catégorie d'obstacle concerné
        direction: Direction suggérée (optionnel)
        distance: Distance mentionnée (optionnel)
        repeat_count: Nombre de répétitions recommandées
        blocking: Si True, bloque les autres messages jusqu'à exécution
    """
    text: str
    priority: InstructionPriority
    obstacle_type: str
    direction: Optional[str] = None
    distance: Optional[float] = None
    repeat_count: int = 1
    blocking: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit pour transmission au module audio."""
        return {
            "text": self.text,
            "priority": self.priority.value,
            "priority_label": self.priority.name,
            "obstacle_type": self.obstacle_type,
            "direction": self.direction, 
            "distance": self.distance,
            "repeat_count": self.repeat_count,
            "blocking": self.blocking,
            "timestamp": time.time()
        }


class DecisionEngine:
    """
    Moteur de décision pour guidage audio.
    
    Transforme les données d'obstacles en instructions parlées
    adaptées au contexte de mobilité des personnes aveugles.
    """
    
    def __init__(
        self,
        min_time_between_messages: float = 2.0,  # Secondes entre messages
        max_message_length: int = 120,           # Caractères max
        enable_contextual_filtering: bool = True, # Évite répétitions
        language: str = "fr"                     # Langue des messages
    ):
        self.min_time_between_messages = min_time_between_messages
        self.max_message_length = max_message_length
        self.enable_contextual_filtering = enable_contextual_filtering
        self.language = language
        
        # Historique pour éviter répétitions
        self._last_message_time: float = 0.0
        self._last_obstacles: deque = deque(maxlen=5)  # Derniers obstacles annoncés
        self._current_context: Dict[str, Any] = {}
        
        # Seuils de décision
        self.collision_imminent_threshold = 0.7
        self.danger_close_threshold = 3.0  # mètres
        self.warning_threshold = 6.0       # mètres
    
    def process_obstacles(self, obstacles: List[Dict[str, Any]]) -> Optional[AudioInstruction]:
        """
        Point d'entrée principal: reçoit la liste d'obstacles et 
        retourne l'instruction audio la plus pertinente.
        
        Args:
            obstacles: Liste de dictionnaires depuis ObstacleDetail.process_frame()
                      (triée par risque décroissant)
        
        Returns:
            AudioInstruction à prononcer, ou None si silence approprié
        """
        if not obstacles:
            return self._check_clear_path()
        
        # Vérification anti-spam temporel
        current_time = time.time()
        if (current_time - self._last_message_time) < self.min_time_between_messages:
            # Vérifier si urgence critique bypass le délai
            if not self._has_critical_urgency(obstacles):
                return None
        
        # Analyse et sélection du message prioritaire
        instruction = self._select_priority_instruction(obstacles)
        
        if instruction and self._should_announce(instruction, obstacles):
            self._last_message_time = current_time
            self._update_context(instruction, obstacles)
            return instruction
        
        return None
    
    def _has_critical_urgency(self, obstacles: List[Dict]) -> bool:
        """Vérifie si obstacle critique nécessite interruption."""
        if not obstacles:
            return False
        top = obstacles[0]
        return (
            top.get("collision_risk", 0) > self.collision_imminent_threshold or
            (top.get("risk_value", 0) >= 4 and top.get("estimated_distance", 100) < 2.0)
        )
    
    def _select_priority_instruction(self, obstacles: List[Dict]) -> Optional[AudioInstruction]:
        """
        Sélectionne l'instruction la plus prioritaire selon l'analyse
        de tous les obstacles présents.
        """
        if not obstacles:
            return None
        
        # 1. Vérification collision imminente (premier obstacle)
        primary = obstacles[0]
        collision_risk = primary.get("collision_risk", 0)
        
        if collision_risk > self.collision_imminent_threshold:
            return self._generate_stop_instruction(primary)
        
        # 2. Analyse de la scène complète pour décision tactique
        scene_analysis = self._analyze_scene(obstacles)
        
        # 3. Décision basée sur la situation globale
        if scene_analysis["blocked_center"]:
            return self._generate_avoidance_instruction(obstacles, scene_analysis)
        
        if scene_analysis["approaching_danger"]:
            return self._generate_warning_instruction(primary)
        
        if scene_analysis["navigation_aid_present"]:
            return self._generate_navigation_instruction(obstacles)
        
        # 4. Information standard sur l'obstacle principal
        return self._generate_info_instruction(primary)
    
    def _analyze_scene(self, obstacles: List[Dict]) -> Dict[str, Any]:
        """
        Analyse la scène complète pour comprendre la situation.
        """
        analysis = {
            "blocked_center": False,
            "approaching_danger": False,
            "navigation_aid_present": False,
            "left_clear": True,
            "right_clear": True,
            "center_clear": True,
            "closest_distance": float('inf'),
            "moving_obstacles": []
        }
        
        for obs in obstacles:
            pos = obs.get("position_horizontal", "CENTER")
            dist = obs.get("estimated_distance", 100)
            risk = obs.get("risk_value", 0)
            moving = obs.get("is_moving", False)
            direction = obs.get("movement_direction", "STATIONARY")
            
            # Mise à jour distance la plus proche
            if dist < analysis["closest_distance"]:
                analysis["closest_distance"] = dist
            
            # Vérification blocage centre
            if pos in ["CENTER", "CENTER_LEFT", "CENTER_RIGHT"] and dist < self.warning_threshold:
                if risk >= 3:  # HIGH ou CRITICAL
                    analysis["center_clear"] = False
                    if dist < self.danger_close_threshold:
                        analysis["blocked_center"] = True
            
            # Vérification côtés
            if pos in ["LEFT", "FAR_LEFT"] and dist < 3:
                analysis["left_clear"] = False
            if pos in ["RIGHT", "FAR_RIGHT"] and dist < 3:
                analysis["right_clear"] = False
            
            # Danger en approche
            if moving and direction == "APPROACHING" and risk >= 3:
                analysis["approaching_danger"] = True
                analysis["moving_obstacles"].append(obs)
            
            # Aide navigation
            if obs.get("category") == "NAVIGATION_AID" and dist < 10:
                analysis["navigation_aid_present"] = True
        
        return analysis
    
    def _generate_stop_instruction(self, obstacle: Dict) -> AudioInstruction:
        """
        Génère instruction d'arrêt d'urgence.
        """
        obj_class = obstacle.get("object_class", "obstacle")
        distance = obstacle.get("estimated_distance", 0)
        
        # Messages spécifiques par type
        if obj_class in ["car", "truck", "bus", "motorcycle"]:
            text = f"Arrêt immédiat! Véhicule à {distance:.1f} mètres. Ne bougez pas."
        elif obj_class == "person":
            text = f"Attention! Personne très proche, {distance:.1f} mètres. Arrêtez-vous."
        elif obj_class in ["bicycle"]:
            text = f"Cycliste rapide approche, {distance:.1f} mètres. Stop."
        else:
            text = f"Danger immédiat, {obj_class} à {distance:.1f} mètres. Arrêt."
        
        return AudioInstruction(
            text=text[:self.max_message_length],
            priority=InstructionPriority.STOP,
            obstacle_type=obj_class,
            distance=distance,
            repeat_count=2,
            blocking=True
        ).to_dict()
    
    def _generate_avoidance_instruction(
        self, 
        obstacles: List[Dict], 
        analysis: Dict
    ) -> AudioInstruction:
        """
        Génère instruction pour contourner obstacle central.
        """
        # Déterminer direction de contournement
        if analysis["left_clear"] and analysis["right_clear"]:
            direction = Direction.LEFT  # Par défaut, privilégier droite (sens de circulation)
            dir_text = Direction.LEFT.value
        elif analysis["left_clear"]:
            direction = Direction.LEFT
            dir_text = Direction.LEFT.value
        elif analysis["right_clear"]:
            direction = Direction.RIGHT
            dir_text = Direction.RIGHT.value
        else:
            # Bloqué des deux côtés
            return self._generate_blocked_instruction(obstacles[0])
        
        primary = obstacles[0]
        obj_class = primary.get("object_class", "obstacle")
        distance = primary.get("estimated_distance", 0)
        
        text = f"Obstacle {obj_class} devant, à {distance:.1f} mètres. Déplacez-vous sur la {dir_text}."
        
        return AudioInstruction(
            text=text[:self.max_message_length],
            priority=InstructionPriority.WARNING,
            obstacle_type=obj_class,
            direction=dir_text,
            distance=distance,
            repeat_count=1
        ).to_dict()
    
    def _generate_blocked_instruction(self, obstacle: Dict) -> AudioInstruction:
        """
        Quand passage bloqué des deux côtés.
        """
        obj_class = obstacle.get("object_class", "obstacle")
        distance = obstacle.get("estimated_distance", 0)
        
        text = f"Passage bloqué par {obj_class} à {distance:.1f} mètres. Arrêtez-vous et évaluez."
        
        return AudioInstruction(
            text=text[:self.max_message_length],
            priority=InstructionPriority.STOP,
            obstacle_type=obj_class,
            distance=distance,
            blocking=True
        ).to_dict()
    
    def _generate_warning_instruction(self, obstacle: Dict) -> AudioInstruction:
        """
        Avertissement pour danger en approche.
        """
        obj_class = obstacle.get("object_class", "obstacle")
        distance = obstacle.get("estimated_distance", 0)
        direction = obstacle.get("position_horizontal", "CENTER")
        
        # Traduction position
        pos_fr = self._translate_position(direction)
        
        if obj_class in ["car", "truck", "bus"]:
            text = f"Véhicule approche {pos_fr}, {distance:.1f} mètres. Ralentissez."
        elif obj_class == "person":
            text = f"Personne se déplace {pos_fr}, {distance:.1f} mètres. Attention."
        else:
            text = f"{obj_class} en mouvement {pos_fr}, {distance:.1f} mètres."
        
        return AudioInstruction(
            text=text[:self.max_message_length],
            priority=InstructionPriority.WARNING,
            obstacle_type=obj_class,
            distance=distance
        ).to_dict()
    
    def _generate_navigation_instruction(self, obstacles: List[Dict]) -> AudioInstruction:
        """
        Instructions pour aides à la navigation (feux, panneaux).
        """
        # Chercher feu ou panneau
        nav_obs = None
        for obs in obstacles:
            if obs.get("category") == "NAVIGATION_AID":
                nav_obs = obs
                break
        
        if not nav_obs:
            return None
        
        obj_class = nav_obs.get("object_class", "")
        distance = nav_obs.get("estimated_distance", 0)
        
        if "traffic light" in obj_class:
            text = f"Feu tricolore à {distance:.1f} mètres. Attendez le signal sonore."
        elif "stop" in obj_class:
            text = f"Panneau stop à {distance:.1f} mètres. Intersection à venir."
        else:
            text = f"Aide navigation {obj_class} à {distance:.1f} mètres."
        
        return AudioInstruction(
            text=text[:self.max_message_length],
            priority=InstructionPriority.INFO,
            obstacle_type=obj_class,
            distance=distance
        ).to_dict()
    
    def _generate_info_instruction(self, obstacle: Dict) -> AudioInstruction:
        """
        Information standard sur obstacle.
        """
        obj_class = obstacle.get("object_class", "obstacle")
        distance = obstacle.get("estimated_distance", 0)
        position = obstacle.get("position_horizontal", "CENTER")
        risk = obstacle.get("intrinsic_risk_level", "LOW")
        
        pos_fr = self._translate_position(position)
        
        # Messages adaptés au niveau de risque
        if risk == "CRITICAL":
            text = f"Attention {obj_class} {pos_fr}, {distance:.1f} mètres."
        elif risk == "HIGH":
            text = f"{obj_class} à {distance:.1f} mètres, {pos_fr}."
        else:
            text = f"{obj_class} {pos_fr}, {distance:.1f} mètres."
        
        priority = InstructionPriority.CAUTION if risk in ["HIGH", "CRITICAL"] else InstructionPriority.INFO
        
        return AudioInstruction(
            text=text[:self.max_message_length],
            priority=priority,
            obstacle_type=obj_class,
            distance=distance
        ).to_dict()
    
    def _check_clear_path(self) -> Optional[AudioInstruction]:
        """
        Vérifie si on peut annoncer "chemin libre".
        """
        # Éviter de spammer "chemin libre"
        if not hasattr(self, '_last_was_clear'):
            self._last_was_clear = False
        
        if self._last_was_clear:
            return None
        
        # Vérifier délai depuis dernier message
        if (time.time() - self._last_message_time) < 5.0:
            return None
        
        self._last_was_clear = True
        return AudioInstruction(
            text="Chemin semble dégagé devant vous.",
            priority=InstructionPriority.INFO,
            obstacle_type="none"
        ).to_dict()
    
    def _translate_position(self, position: str) -> str:
        """Traduit position technique en français parlé."""
        translations = {
            "FAR_LEFT": "tout à gauche",
            "LEFT": "à gauche",
            "CENTER_LEFT": "légèrement à gauche",
            "CENTER": "devant vous",
            "CENTER_RIGHT": "légèrement à droite",
            "RIGHT": "à droite",
            "FAR_RIGHT": "tout à droite"
        }
        return translations.get(position, "devant")
    
    def _should_announce(self, instruction: Dict[str, Any], current_obstacles: List[Dict]) -> bool:
        """
        Détermine si le message doit être annoncé (évite répétitions).
        """
        if not self.enable_contextual_filtering:
            return True
        
        # Récupérer la priorité (entier ou enum)
        priority = instruction.get("priority", 0)
        if hasattr(priority, 'value'):
            priority = priority.value
        
        # Toujours annoncer les stops (priorité 4)
        if priority == 4 or instruction.get("priority_label") == "STOP":
            return True
        
        # Vérifier si similaire au dernier message
        obstacle_type = instruction.get("obstacle_type", "unknown")
        direction = instruction.get("direction")
        distance = instruction.get("distance", 0)
        
        current_key = f"{obstacle_type}_{direction}_{int(distance or 0)}"
        
        if current_key in self._last_obstacles:
            return False
        
        return True
    
    def _update_context(self, instruction: Dict[str, Any], obstacles: List[Dict]):
        """Met à jour le contexte pour filtrage futur."""
        self._last_obstacles.clear()
        
        # Ajouter la clé de l'instruction actuelle
        obstacle_type = instruction.get("obstacle_type", "unknown")
        direction = instruction.get("direction")
        distance = instruction.get("distance", 0)
        current_key = f"{obstacle_type}_{direction}_{int(distance or 0)}"
        self._last_obstacles.append(current_key)
        
        # Garder aussi les 2 obstacles principaux
        for obs in obstacles[:2]:
            key = f"{obs.get('object_class')}_{obs.get('position_horizontal')}_{int(obs.get('estimated_distance', 0))}"
            self._last_obstacles.append(key)
        
        self._last_was_clear = False
        def reset(self):
            """Réinitialise l'état."""
            self._last_message_time = 0.0
            self._last_obstacles.clear()
            self._current_context.clear()
            self._last_was_clear = False


# Fonction utilitaire simple
def get_audio_instruction(obstacles: List[Dict[str, Any]], **kwargs) -> Optional[Dict[str, Any]]:
    """
    Fonction rapide pour obtenir instruction sans instancier la classe.
    
    Returns:
        Dictionnaire prêt pour le module audio, ou None
    """
    engine = DecisionEngine(**kwargs)
    instruction = engine.process_obstacles(obstacles)
    return instruction.to_dict() if instruction else None