"""Chemins et paramètres du prototype (valeurs d'origine, extraites du code)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = str(PROJECT_ROOT / "models" / "yolov8n.pt")

# main.py (pipeline multi-frame)
MAIN_CAMERA_INDEX = 1
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.3
STABILITY_THRESHOLD = 0.6

# scripts/live_demo.py (pipeline temps réel)
LIVE_CAMERA_INDEX = 0
