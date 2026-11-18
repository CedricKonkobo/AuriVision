"""
Module de traitement intelligent d'images
Analyse et améliore la qualité des images avant envoi à YOLO
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class ImageQualityMetrics:
    """Métriques de qualité d'une image"""
    brightness: float
    contrast: float
    sharpness: float
    noise_level: float
    color_balance: float
    saturation: float
    overall_quality: float
    needs_processing: bool
    recommended_treatments: List[str]


class ImageTraitement:
    def __init__(self):
        """
        Initialise le processeur d'images intelligent
        """
        # Seuils de qualité
        self.quality_thresholds = {
            'min_brightness': 80,
            'max_brightness': 200,
            'min_contrast': 40,
            'min_sharpness': 50,
            'max_noise': 15,
            'color_balance_tolerance': 10,
            'min_saturation': 30,
            'overall_quality_threshold': 70
        }
    
    def traiter_frames(self, frames: List[np.ndarray]) -> Tuple[List[np.ndarray], Dict]:
        """
        Traite intelligemment 3 frames et retourne les images prêtes pour YOLO
        
        Args:
            frames: Liste de 3 images (numpy arrays)
            
        Returns:
            Tuple (liste des images traitées, statistiques)
        """
        if len(frames) != 3:
            raise ValueError(f"Exactement 3 frames attendues, {len(frames)} reçues")
        
        processed_images = []
        
        stats = {
            'total_frames': len(frames),
            'processed_frames': 0,
            'skipped_frames': 0,
            'treatments_applied': {}
        }
        
        for i, frame in enumerate(frames):
            # Analyser la qualité
            metrics = self._analyze_image_quality(frame)
            
            # Appliquer les traitements adaptatifs si nécessaire
            if metrics.needs_processing:
                processed = self._apply_adaptive_processing(frame, metrics)
                stats['processed_frames'] += 1
                
                for treatment in metrics.recommended_treatments:
                    stats['treatments_applied'][treatment] = \
                        stats['treatments_applied'].get(treatment, 0) + 1
            else:
                processed = frame.copy()
                stats['skipped_frames'] += 1
            
            # Redimensionner pour YOLO
            yolo_ready = self._resize_for_yolo(processed)
            
            processed_images.append(yolo_ready)
        
        return processed_images, stats
    
    def _analyze_image_quality(self, image: np.ndarray) -> ImageQualityMetrics:
        """Analyse complète de la qualité d'une image"""
        brightness = self._analyze_brightness(image)
        contrast = self._analyze_contrast(image)
        sharpness = self._analyze_sharpness(image)
        noise_level = self._analyze_noise(image)
        color_balance = self._analyze_color_balance(image)
        saturation = self._analyze_saturation(image)
        
        recommended_treatments = self._determine_treatments(
            brightness, contrast, sharpness, noise_level, color_balance, saturation
        )
        
        overall_quality = self._calculate_overall_quality(
            brightness, contrast, sharpness, noise_level, color_balance, saturation
        )
        
        needs_processing = overall_quality < self.quality_thresholds['overall_quality_threshold']
        
        return ImageQualityMetrics(
            brightness=brightness,
            contrast=contrast,
            sharpness=sharpness,
            noise_level=noise_level,
            color_balance=color_balance,
            saturation=saturation,
            overall_quality=overall_quality,
            needs_processing=needs_processing,
            recommended_treatments=recommended_treatments
        )
    
    def _analyze_brightness(self, image: np.ndarray) -> float:
        """Analyse la luminosité moyenne"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    
    def _analyze_contrast(self, image: np.ndarray) -> float:
        """Analyse le contraste"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)
        return min(contrast * 2, 100)
    
    def _analyze_sharpness(self, image: np.ndarray) -> float:
        """Analyse la netteté"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return min(laplacian_var / 10, 100)
    
    def _analyze_noise(self, image: np.ndarray) -> float:
        """Estime le niveau de bruit"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.medianBlur(gray, 5)
        noise = np.abs(gray.astype(float) - denoised.astype(float))
        noise_level = np.mean(noise)
        return min(noise_level * 5, 100)
    
    def _analyze_color_balance(self, image: np.ndarray) -> float:
        """Analyse l'équilibre des couleurs"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        a_deviation = abs(np.mean(a) - 128)
        b_deviation = abs(np.mean(b) - 128)
        
        total_deviation = (a_deviation + b_deviation) / 2
        balance_score = max(100 - total_deviation * 2, 0)
        
        return balance_score
    
    def _analyze_saturation(self, image: np.ndarray) -> float:
        """Analyse le niveau de saturation"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:, :, 1])
        return (saturation / 255) * 100
    
    def _calculate_overall_quality(self, brightness: float, contrast: float,
                                   sharpness: float, noise_level: float,
                                   color_balance: float, saturation: float) -> float:
        """Calcule un score global de qualité"""
        brightness_score = max(100 - abs(brightness - 127) / 1.27, 0)
        noise_score = 100 - noise_level
        
        weights = {
            'brightness': 0.15,
            'contrast': 0.20,
            'sharpness': 0.25,
            'noise': 0.20,
            'color_balance': 0.10,
            'saturation': 0.10
        }
        
        overall = (
            brightness_score * weights['brightness'] +
            contrast * weights['contrast'] +
            sharpness * weights['sharpness'] +
            noise_score * weights['noise'] +
            color_balance * weights['color_balance'] +
            saturation * weights['saturation']
        )
        
        return overall
    
    def _determine_treatments(self, brightness: float, contrast: float,
                             sharpness: float, noise_level: float,
                             color_balance: float, saturation: float) -> List[str]:
        """Détermine les traitements nécessaires"""
        treatments = []
        
        if brightness < self.quality_thresholds['min_brightness']:
            treatments.append("augmenter_luminosité")
        elif brightness > self.quality_thresholds['max_brightness']:
            treatments.append("réduire_luminosité")
        
        if contrast < self.quality_thresholds['min_contrast']:
            treatments.append("améliorer_contraste")
        
        if sharpness < self.quality_thresholds['min_sharpness']:
            treatments.append("améliorer_netteté")
        
        if noise_level > self.quality_thresholds['max_noise']:
            treatments.append("réduire_bruit")
        
        if color_balance < (100 - self.quality_thresholds['color_balance_tolerance']):
            treatments.append("corriger_balance_couleurs")
        
        if saturation < self.quality_thresholds['min_saturation']:
            treatments.append("augmenter_saturation")
        
        return treatments
    
    def _apply_adaptive_processing(self, image: np.ndarray,
                                   metrics: ImageQualityMetrics) -> np.ndarray:
        """Applique les traitements nécessaires"""
        if not metrics.needs_processing:
            return image.copy()
        
        processed = image.copy()
        
        for treatment in metrics.recommended_treatments:
            if treatment == "réduire_bruit":
                processed = self._reduce_noise(processed)
            elif treatment == "améliorer_netteté":
                processed = self._enhance_sharpness(processed)
            elif treatment == "améliorer_contraste":
                processed = self._enhance_contrast(processed)
            elif treatment == "augmenter_luminosité":
                processed = self._adjust_brightness(processed, increase=True)
            elif treatment == "réduire_luminosité":
                processed = self._adjust_brightness(processed, increase=False)
            elif treatment == "corriger_balance_couleurs":
                processed = self._correct_color_balance(processed)
            elif treatment == "augmenter_saturation":
                processed = self._enhance_saturation(processed)
        
        return processed
    
    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Réduit le bruit"""
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    def _enhance_sharpness(self, image: np.ndarray) -> np.ndarray:
        """Améliore la netteté"""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Améliore le contraste avec CLAHE"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        enhanced = cv2.merge([l_clahe, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    def _adjust_brightness(self, image: np.ndarray, increase: bool = True) -> np.ndarray:
        """Ajuste la luminosité"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        if increase:
            hsv[:, :, 2] = hsv[:, :, 2] * 1.2
        else:
            hsv[:, :, 2] = hsv[:, :, 2] * 0.8
        
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _correct_color_balance(self, image: np.ndarray) -> np.ndarray:
        """Corrige la balance des couleurs"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        
        avg_a = np.average(a)
        avg_b = np.average(b)
        
        a = a - ((avg_a - 128) * (l / 255.0) * 1.2)
        b = b - ((avg_b - 128) * (l / 255.0) * 1.2)
        
        lab = cv2.merge([l, a, b])
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _enhance_saturation(self, image: np.ndarray) -> np.ndarray:
        """Augmente la saturation"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * 1.3
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _resize_for_yolo(self, frame: np.ndarray, target_size: int = 640) -> np.ndarray:
        """Redimensionne l'image pour YOLO"""
        h, w = frame.shape[:2]
        
        if h > w:
            new_h = target_size
            new_w = int(w * (target_size / h))
        else:
            new_w = target_size
            new_h = int(h * (target_size / w))
        
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        canvas.fill(114)
        
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas