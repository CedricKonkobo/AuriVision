"""
Système de Guidage Intelligent pour Personnes Aveugles
======================================================
Version adaptée pour accepter les dictionnaires du module obstacle_detail.

Principes de conception :
- Parler seulement quand nécessaire
- Prioriser les dangers immédiats
- Éviter la surcharge d'informations
- Maintenir la cohérence temporelle
- Adapter le ton à l'urgence
"""

import time
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    """Types d'actions recommandées"""
    STOP = "stop"
    AVANCER = "avancer"
    TOURNER_GAUCHE = "tourner_gauche"
    TOURNER_DROITE = "tourner_droite"
    ATTENDRE = "attendre"
    RALENTIR = "ralentir"
    CONTINUER = "continuer"


class Urgency(Enum):
    """Niveaux d'urgence des instructions"""
    CRITIQUE = 3  # Danger immédiat
    HAUTE = 2     # Action requise rapidement
    NORMALE = 1   # Information utile
    FAIBLE = 0    # Contexte non urgent


@dataclass
class GuidanceInstruction:
    """Structure d'une instruction de guidage"""
    message: str
    action: ActionType
    urgency: Urgency
    should_speak: bool
    confidence: float
    timestamp: float


class BlindGuidanceSystem:
    """
    Système principal de guidage pour personnes aveugles.
    Accepte les dictionnaires du module obstacle_detail.
    """
    
    def __init__(self):
        # Historique des obstacles pour cohérence temporelle
        self.obstacle_history = {}
        
        # Dernière instruction donnée
        self.last_instruction_time = 0
        self.last_instruction_type = None
        
        # Temps minimum entre deux instructions (secondes)
        self.min_instruction_interval = 2.0
        
        # Temps minimum entre répétitions de la même instruction
        self.repetition_cooldown = 5.0
        
        # Seuils de distance (en mètres)
        self.DISTANCE_TRES_PROCHE = 1.0
        self.DISTANCE_PROCHE = 2.0
        self.DISTANCE_MOYENNE = 4.0
        
    def  process_obstacles(self, obstacles: List[Dict[str, Any]],
                          current_time: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Traite la liste des obstacles (dictionnaires) et retourne l'instruction la plus pertinente.
        
        Args:
            obstacles: Liste de dictionnaires d'obstacles du module obstacle_detail
            current_time: Timestamp actuel (utilise time.time() si None)
            
        Returns:
            GuidanceInstruction ou None si aucune instruction nécessaire
        """
        if current_time is None:
            current_time = time.time()
            
        # Mise à jour de l'historique
        self._update_history(obstacles, current_time)
        
        # Filtrer et enrichir les obstacles
        processed_obstacles = []
        for obs in obstacles:
            if self._should_process_obstacle(obs):
                enriched = self._enrich_obstacle(obs, current_time)
                processed_obstacles.append(enriched)
        
        if not processed_obstacles:
            return self._generate_clear_path_instruction(current_time)
        
        # Calculer les scores de priorité
        scored_obstacles = self._score_obstacles(processed_obstacles)
        
        # Trier par priorité décroissante
        scored_obstacles.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # Sélectionner l'obstacle le plus prioritaire
        top_obstacle = scored_obstacles[0]
        
        # Générer l'instruction appropriée
        instruction = self._generate_instruction(top_obstacle, 
                                                 processed_obstacles, 
                                                 current_time)
        
        # Vérifier si on doit parler (éviter spam)
        if instruction and self._should_speak(instruction, current_time):
            self.last_instruction_time = current_time
            self.last_instruction_type = instruction.action
            return instruction
        
        return None
    
    def _should_process_obstacle(self, obstacle: Dict[str, Any]) -> bool:
        """Détermine si un obstacle doit être traité"""
        # Vérifier confiance minimale
        if obstacle.get('detection_confidence', 0) < 0.4:
            return False
        
        # Ignorer les obstacles très éloignés
        if obstacle.get('estimated_distance', 999) > 10.0:
            return False
        
        # Ignorer les obstacles passables non dangereux à faible risque
        if obstacle.get('is_passable') and obstacle.get('intrinsic_risk_level') == 'LOW':
            return False
        
        return True
    
    def _enrich_obstacle(self, obstacle: Dict[str, Any], current_time: float):
        """Enrichit les données d'un obstacle avec l'historique"""
        obs_id = obstacle.get('object_id', 0)
        
        enriched = dict(obstacle)  # Copie du dictionnaire
        
        # Ajouter les informations d'historique
        if obs_id in self.obstacle_history:
            prev = self.obstacle_history[obs_id]
            enriched['previous_distance'] = prev.get('estimated_distance')
            enriched['time_visible'] = current_time - prev.get('first_seen', current_time)
            enriched['is_new'] = False
            
            # Calculer la tendance de distance
            if enriched['previous_distance'] is not None:
                distance_change = enriched['previous_distance'] - obstacle.get('estimated_distance', 0)
                if distance_change > 0.2:  # Se rapproche
                    enriched['distance_trend'] = 'Se rapproche'
                elif distance_change < -0.2:  # S'éloigne
                    enriched['distance_trend'] = "S'éloigne"
                else:
                    enriched['distance_trend'] = 'Stable'
            else:
                enriched['distance_trend'] = 'Stable'
        else:
            enriched['is_new'] = True
            enriched['time_visible'] = 0
            enriched['distance_trend'] = 'Stable'
            
        return enriched
    
    def _update_history(self, obstacles: List[Dict[str, Any]], current_time: float):
        """Met à jour l'historique des obstacles"""
        current_ids = set()
        
        for obs in obstacles:
            obs_id = obs.get('object_id')
            if obs_id is None:
                continue
                
            current_ids.add(obs_id)
            
            if obs_id not in self.obstacle_history:
                self.obstacle_history[obs_id] = {
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'estimated_distance': obs.get('estimated_distance', 0)
                }
            else:
                self.obstacle_history[obs_id]['last_seen'] = current_time
                self.obstacle_history[obs_id]['estimated_distance'] = obs.get('estimated_distance', 0)
        
        # Nettoyer l'historique (obstacles disparus depuis >5s)
        to_remove = []
        for obs_id, data in self.obstacle_history.items():
            if obs_id not in current_ids:
                if current_time - data['last_seen'] > 5.0:
                    to_remove.append(obs_id)
        
        for obs_id in to_remove:
            del self.obstacle_history[obs_id]
    
    def _score_obstacles(self, obstacles: List[Dict[str, Any]]):
        """
        Calcule un score de priorité pour chaque obstacle.
        Score élevé = haute priorité
        """
        scored = []
        
        for obs in obstacles:
            score = 0.0
            
            # 1. Distance (plus proche = plus prioritaire)
            distance = obs.get('estimated_distance', 999)
            if distance < self.DISTANCE_TRES_PROCHE:
                score += 100
            elif distance < self.DISTANCE_PROCHE:
                score += 50
            elif distance < self.DISTANCE_MOYENNE:
                score += 20
            else:
                score += 5
            
            # 2. Position (centre = plus prioritaire)
            position_h = obs.get('position_horizontal', 'CENTER')
            if position_h == 'CENTER':
                score += 30
            elif position_h in ['LEFT', 'RIGHT']:
                score += 15
            
            # 3. Mouvement (vers l'utilisateur = critique)
            if obs.get('is_moving'):
                movement_dir = obs.get('movement_direction', 'STATIONARY')
                if movement_dir == "APPROACHING":
                    score += 80
                elif movement_dir in ["LATERAL_LEFT", "LATERAL_RIGHT"]:
                    score += 30
            
            # 4. Niveau de risque intrinsèque
            risk = obs.get('intrinsic_risk_level', 'LOW')
            risk_scores = {'CRITICAL': 70, 'HIGH': 50, 'MEDIUM': 25, 'LOW': 5}
            score += risk_scores.get(risk, 0)
            
            # 5. Catégorie d'objet (certains plus dangereux)
            obj_class = obs.get('object_class', '').lower()
            if obj_class in ['car', 'voiture', 'vehicle', 'bus', 'truck', 'train']:
                score += 40
            elif obj_class in ['person', 'personne', 'pedestrian']:
                score += 45
            elif obj_class in ['bicycle', 'vélo', 'bike', 'motorcycle']:
                score += 35
            elif obj_class in ['stairs', 'escalier', 'marche']:
                score += 35
            elif obj_class in ['wall', 'mur', 'poteau', 'pole']:
                score += 25
            
            # 6. Obstacle non passable
            if not obs.get('is_passable', True):
                score += 30
            
            # 7. Encombrement (bloque le passage)
            occupancy = obs.get('occupancy_ratio', 0)
            if occupancy > 0.6:  # Bloque >60% du champ
                score += 25
            elif occupancy > 0.3:  # Bloque >30% du champ
                score += 15
            
            # 8. Confiance de détection
            confidence = obs.get('detection_confidence', 0)
            score *= confidence
            
            # 9. Nouveauté (nouveaux obstacles = plus d'attention)
            if obs.get('is_new'):
                score *= 1.2
            
            # 10. Tendance de distance (se rapproche)
            if obs.get('distance_trend') == 'Se rapproche':
                score *= 1.5
            
            obs['priority_score'] = score
            scored.append(obs)
        
        return scored
    
    def _generate_instruction(self, obstacle: Dict[str, Any], 
                            all_obstacles: List[Dict[str, Any]],
                            current_time: float) -> GuidanceInstruction:
        """
        Génère l'instruction vocale appropriée pour l'obstacle prioritaire.
        """
        distance = obstacle.get('estimated_distance', 0)
        obj_class = obstacle.get('object_class', 'objet')
        position_h = obstacle.get('position_horizontal', 'CENTER')
        is_moving = obstacle.get('is_moving', False)
        is_passable = obstacle.get('is_passable', True)
        movement_dir = obstacle.get('movement_direction', 'STATIONARY')
        
        # Déterminer l'urgence
        urgency = self._calculate_urgency(obstacle)
        
        # Déterminer l'action recommandée
        action, message = self._decide_action(obstacle, all_obstacles)
        
        # Adapter le message selon l'urgence et le contexte
        message = self._format_message(message, obstacle, urgency)
        
        return GuidanceInstruction(
            message=message,
            action=action,
            urgency=urgency,
            should_speak=True,
            confidence=obstacle.get('detection_confidence', 0),
            timestamp=current_time
        )
    
    def _calculate_urgency(self, obstacle: Dict[str, Any]) -> Urgency:
        """Calcule le niveau d'urgence"""
        distance = obstacle.get('estimated_distance', 999)
        is_moving = obstacle.get('is_moving', False)
        movement_dir = obstacle.get('movement_direction', 'STATIONARY')
        risk = obstacle.get('intrinsic_risk_level', 'LOW')
        
        # Danger immédiat
        if distance < self.DISTANCE_TRES_PROCHE:
            if is_moving and movement_dir == "APPROACHING":
                return Urgency.CRITIQUE
            if risk in ['HIGH', 'CRITICAL']:
                return Urgency.CRITIQUE
            return Urgency.HAUTE
        
        # Danger proche
        if distance < self.DISTANCE_PROCHE:
            if is_moving and movement_dir == "APPROACHING":
                return Urgency.HAUTE
            if risk in ['HIGH', 'CRITICAL']:
                return Urgency.HAUTE
            return Urgency.NORMALE
        
        # Danger moyen
        if distance < self.DISTANCE_MOYENNE:
            return Urgency.NORMALE
        
        return Urgency.FAIBLE
    
    def _decide_action(self, obstacle: Dict[str, Any], 
                      all_obstacles: List[Dict[str, Any]]) -> Tuple[ActionType, str]:
        """
        Décide de l'action à recommander et génère le message de base.
        """
        distance = obstacle.get('estimated_distance', 0)
        position_h = obstacle.get('position_horizontal', 'CENTER')
        obj_class = obstacle.get('object_class', 'objet')
        is_moving = obstacle.get('is_moving', False)
        is_passable = obstacle.get('is_passable', True)
        movement_dir = obstacle.get('movement_direction', 'STATIONARY')
        
        # Objet mobile dangereux approchant
        if is_moving and movement_dir == "APPROACHING":
            if distance < self.DISTANCE_PROCHE:
                return ActionType.STOP, f"Attention ! {obj_class} approche"
            else:
                return ActionType.RALENTIR, f"{obj_class} en mouvement devant"
        
        # Obstacle très proche - action immédiate
        if distance < self.DISTANCE_TRES_PROCHE:
            if position_h == 'CENTER':
                if is_passable:
                    # Chercher un passage libre
                    left_clear = self._is_direction_clear('LEFT', all_obstacles)
                    right_clear = self._is_direction_clear('RIGHT', all_obstacles)
                    
                    if left_clear and right_clear:
                        return ActionType.TOURNER_GAUCHE, f"{obj_class} devant, contournez à gauche ou droite"
                    elif left_clear:
                        return ActionType.TOURNER_GAUCHE, f"{obj_class} devant, allez à gauche"
                    elif right_clear:
                        return ActionType.TOURNER_DROITE, f"{obj_class} devant, allez à droite"
                    else:
                        return ActionType.STOP, f"{obj_class} devant, passage bloqué"
                else:
                    return ActionType.STOP, f"{obj_class} devant vous"
            
            elif position_h == 'LEFT':
                return ActionType.TOURNER_DROITE, f"{obj_class} sur votre gauche"
            elif position_h == 'RIGHT':
                return ActionType.TOURNER_GAUCHE, f"{obj_class} sur votre droite"
        
        # Obstacle proche - anticipation
        elif distance < self.DISTANCE_PROCHE:
            if position_h == 'CENTER':
                return ActionType.RALENTIR, f"{obj_class} droit devant"
            elif position_h == 'LEFT':
                return ActionType.CONTINUER, f"{obj_class} à gauche"
            elif position_h == 'RIGHT':
                return ActionType.CONTINUER, f"{obj_class} à droite"
        
        # Obstacle à distance moyenne - information
        elif distance < self.DISTANCE_MOYENNE:
            if position_h == 'CENTER' and not is_passable:
                return ActionType.RALENTIR, f"{obj_class} devant"
        
        return ActionType.CONTINUER, "Continuez"
    
    def _is_direction_clear(self, direction: str, 
                           obstacles: List[Dict[str, Any]]) -> bool:
        """Vérifie si une direction est libre d'obstacles proches"""
        for obs in obstacles:
            if obs.get('position_horizontal') == direction:
                distance = obs.get('estimated_distance', 999)
                if distance < self.DISTANCE_PROCHE:
                    return False
        return True
    
    def _format_message(self, base_message: str, 
                       obstacle: Dict[str, Any], 
                       urgency: Urgency) -> str:
        """Formate le message selon l'urgence et le contexte"""
        
        # Messages critiques : courts et directs
        if urgency == Urgency.CRITIQUE:
            return base_message.upper() if len(base_message) < 30 else base_message
        
        # Messages normaux : ajouter la distance si pertinent
        distance = obstacle.get('estimated_distance', 0)
        if urgency in [Urgency.HAUTE, Urgency.NORMALE] and distance:
            if distance < 1.5:
                distance_str = "très proche"
            elif distance < 3:
                distance_str = "proche"
            else:
                distance_str = f"à {int(distance)} mètres"
            
            # Ajouter la distance seulement si le message est court
            if len(base_message) < 40:
                base_message = f"{base_message}, {distance_str}"
        
        return base_message
    
    def _generate_clear_path_instruction(self, 
                                        current_time: float) -> Optional[GuidanceInstruction]:
        """
        Génère une instruction de voie libre si approprié.
        Ne parle que si longtemps depuis la dernière instruction.
        """
        time_since_last = current_time - self.last_instruction_time
        
        # Parler seulement après 10 secondes de silence
        if time_since_last > 10.0:
            return GuidanceInstruction(
                message="Voie libre",
                action=ActionType.AVANCER,
                urgency=Urgency.FAIBLE,
                should_speak=True,
                confidence=1.0,
                timestamp=current_time
            )
        
        return None
    
    def _should_speak(self, instruction: GuidanceInstruction, 
                     current_time: float) -> bool:
        """
        Détermine si l'instruction doit être vocalisée.
        Évite le spam d'instructions.
        """
        time_since_last = current_time - self.last_instruction_time
        
        # Toujours parler pour urgence critique
        if instruction.urgency == Urgency.CRITIQUE:
            return True
        
        # Urgence haute : respecter intervalle minimum court
        if instruction.urgency == Urgency.HAUTE:
            return time_since_last >= 1.0
        
        # Urgence normale : intervalle standard
        if instruction.urgency == Urgency.NORMALE:
            # Si même action, attendre cooldown
            if instruction.action == self.last_instruction_type:
                return time_since_last >= self.repetition_cooldown
            return time_since_last >= self.min_instruction_interval
        
        # Urgence faible : intervalle long
        return time_since_last >= 5.0