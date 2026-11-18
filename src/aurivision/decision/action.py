"""
Module Action - Affichage console des instructions de guidage
=============================================================

Ce module reçoit les messages du DecisionEngine et les affiche 
de manière claire et lisible sur la console.

Fonctionnalités:
- Affichage formaté des instructions prioritaires
- Gestion du silence quand aucune action n'est nécessaire
- Historique des dernières instructions
- Indicateurs visuels de priorité (couleurs ANSI)
"""

import time
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum


class PriorityLevel(Enum):
    """Niveaux de priorité pour affichage."""
    STOP = 4
    WARNING = 3
    CAUTION = 2
    INFO = 1
    NONE = 0


@dataclass
class DisplayConfig:
    """Configuration d'affichage."""
    show_timestamp: bool = True
    show_priority_label: bool = True
    use_colors: bool = True
    max_history: int = 5
    silent_mode: bool = False  # Si True, n'affiche que les warnings et plus


class ActionModule:
    """
    Module d'action qui affiche les instructions de guidage sur console.
    
    Reçoit les messages du DecisionEngine et les présente de manière
    lisible avec indicateurs de priorité.
    """
    
    # Codes couleur ANSI
    COLORS = {
        "RESET": "\033[0m",
        "RED": "\033[91m",
        "YELLOW": "\033[93m",
        "BLUE": "\033[94m",
        "GREEN": "\033[92m",
        "MAGENTA": "\033[95m",
        "BOLD": "\033[1m",
        "DIM": "\033[2m"
    }
    
    # Symboles par priorité
    SYMBOLS = {
        4: "🛑",  # STOP
        3: "⚠️",  # WARNING
        2: "⚡",  # CAUTION
        1: "ℹ️",  # INFO
        0: "  "   # NONE
    }
    
    def __init__(self, config: Optional[DisplayConfig] = None):
        self.config = config or DisplayConfig()
        self._last_instruction: Optional[Dict[str, Any]] = None
        self._instruction_history: List[Dict[str, Any]] = []
        self._silence_start_time: Optional[float] = None
        self._message_count: int = 0
        
        # Affichage d'initialisation
        self._print_header()
    
    def _print_header(self):
        """Affiche l'en-tête du système."""
        header = "=" * 60
        title = "SYSTÈME DE GUIDAGE - MODE CONSOLE"
        subtitle = "En attente d'instructions..."
        
        print(f"\n{header}")
        print(f"{title:^60}")
        print(f"{subtitle:^60}")
        print(f"{header}\n")
    
    def process_instruction(self, instruction: Optional[Dict[str, Any]]) -> bool:
        """
        Point d'entrée principal: reçoit l'instruction et décide d'afficher ou non.
        
        Args:
            instruction: Dictionnaire depuis DecisionEngine.process_obstacles()
                        ou None si pas d'instruction
        
        Returns:
            True si un message a été affiché, False sinon
        """
        # Cas 1: Aucune instruction -> silence
        if instruction is None:
            self._handle_silence()
            return False
        
        # Cas 2: Mode silencieux et priorité basse
        if self.config.silent_mode and instruction.get("priority", 0) < 2:
            return False
        
        # Cas 3: Message identique au précédent (éviter répétition)
        if self._is_duplicate(instruction):
            return False
        
        # Affichage de l'instruction
        self._display_instruction(instruction)
        self._update_history(instruction)
        return True
    
    def _is_duplicate(self, instruction: Dict[str, Any]) -> bool:
        """Vérifie si le message est identique au précédent."""
        if self._last_instruction is None:
            return False
        
        # Comparer texte et priorité
        same_text = instruction.get("text") == self._last_instruction.get("text")
        same_priority = instruction.get("priority") == self._last_instruction.get("priority")
        
        # Si même message dans les 3 dernières secondes, c'est un doublon
        if same_text and same_priority:
            last_time = self._last_instruction.get("timestamp", 0)
            if (time.time() - last_time) < 3.0:
                return True
        
        return False
    
    def _display_instruction(self, instruction: Dict[str, Any]):
        """Affiche l'instruction formatée sur console."""
        priority = instruction.get("priority", 0)
        text = instruction.get("text", "")
        priority_label = instruction.get("priority_label", "UNKNOWN")
        obstacle_type = instruction.get("obstacle_type", "unknown")
        direction = instruction.get("direction")
        distance = instruction.get("distance")
        
        # Construction de la ligne
        timestamp = ""
        if self.config.show_timestamp:
            timestamp = f"[{datetime.now().strftime('%H:%M:%S')}] "
        
        # Couleur selon priorité
        color = self._get_color(priority)
        symbol = self.SYMBOLS.get(priority, "  ")
        
        # Formatage principal
        lines = []
        lines.append("")  # Ligne vide pour séparation
        lines.append(f"{timestamp}{symbol} {color}{self.COLORS['BOLD']}{text}{self.COLORS['RESET']}")
        
        # Détails supplémentaires si disponibles
        details = []
        if self.config.show_priority_label:
            details.append(f"Priorité: {priority_label}")
        if obstacle_type and obstacle_type != "unknown":
            details.append(f"Type: {obstacle_type}")
        if direction:
            details.append(f"Direction: {direction}")
        if distance is not None:
            details.append(f"Distance: {distance:.1f}m")
        
        if details:
            detail_str = " | ".join(details)
            lines.append(f"    {self.COLORS['DIM']}{detail_str}{self.COLORS['RESET']}")
        
        # Affichage
        output = "\n".join(lines)
        print(output)
        
        # Ligne de séparation pour messages critiques
        if priority >= 3:
            print(f"    {color}{'=' * 50}{self.COLORS['RESET']}")
        
        # Mise à jour état
        self._last_instruction = instruction.copy()
        self._last_instruction["timestamp"] = time.time()
        self._message_count += 1
    
    def _get_color(self, priority: int) -> str:
        """Retourne la couleur selon la priorité."""
        if not self.config.use_colors:
            return ""
        
        colors_map = {
            4: self.COLORS["RED"],      # STOP
            3: self.COLORS["YELLOW"],   # WARNING
            2: self.COLORS["MAGENTA"],  # CAUTION
            1: self.COLORS["BLUE"],     # INFO
            0: self.COLORS["DIM"]       # NONE
        }
        return colors_map.get(priority, self.COLORS["RESET"])
    
    def _handle_silence(self):
        """Gère les périodes de silence (pas d'instruction)."""
        if self._silence_start_time is None:
            self._silence_start_time = time.time()
            # Afficher un point pour indiquer que le système écoute
            print(".", end="", flush=True)
    
    def _update_history(self, instruction: Dict[str, Any]):
        """Met à jour l'historique des instructions."""
        self._instruction_history.append(instruction)
        if len(self._instruction_history) > self.config.max_history:
            self._instruction_history.pop(0)
        
        # Réinitialiser le compteur de silence
        self._silence_start_time = None
    
    def show_status(self):
        """Affiche un résumé du statut actuel."""
        print(f"\n{'='*60}")
        print("STATUT DU SYSTÈME")
        print(f"{'='*60}")
        print(f"Messages affichés: {self._message_count}")
        print(f"Dernière instruction: {self._last_instruction['text'][:50] if self._last_instruction else 'Aucune'}")
        print(f"Historique ({len(self._instruction_history)} messages):")
        
        for i, inst in enumerate(reversed(self._instruction_history[-3:]), 1):
            text = inst.get("text", "")[:40]
            prio = inst.get("priority_label", "?")
            print(f"  {i}. [{prio}] {text}...")
        
        print(f"{'='*60}\n")
    
    def clear_screen(self):
        """Efface l'écran (simulation)."""
        print("\n" * 3)
        self._print_header()
    
    def set_silent_mode(self, enabled: bool):
        """Active/désactive le mode silencieux (affiche uniquement warnings+)."""
        self.config.silent_mode = enabled
        status = "activé" if enabled else "désactivé"
        print(f"\n[CONFIG] Mode silencieux {status}\n")


# Fonction utilitaire simple
def display_action(instruction: Optional[Dict[str, Any]], **kwargs) -> bool:
    """
    Affiche rapidement une instruction sans instancier la classe.
    
    Returns:
        True si affiché, False sinon
    """
    action = ActionModule(DisplayConfig(**kwargs))
    return action.process_instruction(instruction)


# Version minimale pour tests rapides
class SimpleAction:
    """
    Version simplifiée sans configuration, juste affiche ou non.
    """
    
    @staticmethod
    def show(instruction: Optional[Dict[str, Any]]):
        """Affiche simplement le message si présent."""
        if instruction is None:
            return
        
        priority = instruction.get("priority", 0)
        text = instruction.get("text", "")
        
        # Ignorer les messages INFO de faible priorité
        if priority < 2:
            return
        
        symbols = {4: "🛑", 3: "⚠️", 2: "!", 1: "ℹ️"}
        symbol = symbols.get(priority, "")
        
        print(f"\n{symbol} {text}")
        
        # Détails si disponibles
        if instruction.get("distance"):
            print(f"   → Distance: {instruction['distance']:.1f}m")


# Exemple d'utilisation
# if __name__ == "__main__":
#     # Simulation
#     action = ActionModule()
    
#     # Test 1: Danger critique
#     action.process_instruction({
#         "text": "Arrêt immédiat! Véhicule à 2.5 mètres. Ne bougez pas.",
#         "priority": 4,
#         "priority_label": "STOP",
#         "obstacle_type": "car",
#         "distance": 2.5,
#         "direction": None
#     })
    
#     time.sleep(1)
    
#     # Test 2: Warning
#     action.process_instruction({
#         "text": "Véhicule approche à droite, 8.5 mètres. Ralentissez.",
#         "priority": 3,
#         "priority_label": "WARNING",
#         "obstacle_type": "car",
#         "distance": 8.5,
#         "direction": "droite"
#     })
    
#     time.sleep(1)
    
#     # Test 3: Silence (None)
#     action.process_instruction(None)
#     time.sleep(0.5)
#     action.process_instruction(None)
    
#     time.sleep(1)
    
#     # Test 4: Info
#     action.process_instruction({
#         "text": "Chemin semble dégagé devant vous.",
#         "priority": 1,
#         "priority_label": "INFO",
#         "obstacle_type": "none"
#     })
    
#     # Statut final
#     action.show_status()