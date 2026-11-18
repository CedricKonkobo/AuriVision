import cv2

# Teste tous les index de 0 à 5
for i in range(6):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Caméra trouvée à l'index : {i}")
        cap.release()
    else:
        print(f"❌ Index {i} : pas de caméra")