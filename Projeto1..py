import cv2
import math

# carregar o arquivo de classificador Haar Cascade para detecção de faces
face_cascade = cv2.CascadeClassifier('haarcascades\haarcascade_frontalface_default.xml')

# abrir a câmera com o índice correto (0 para a câmera padrão, 1 para uma câmera externa)
cap = cv2.VideoCapture(0)

while True:
    # capturar o frame atual da câmera
    ret, frame = cap.read()

    # inverter a imagem horizontalmente
    frame = cv2.flip(frame, 1)

    # converter o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detectar as faces na imagem
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    # desenhar um retângulo em volta de cada face detectada
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

    # desenhar uma linha entre os retângulos de duas faces detectadas
    if len(faces) == 2:
        # calcular as coordenadas dos pontos médios dos dois retângulos
        x1, y1, w1, h1 = faces[0]
        x2, y2, w2, h2 = faces[1]
        center1 = (x1 + w1 // 2, y1 + h1 // 2)
        center2 = (x2 + w2 // 2, y2 + h2 // 2)
        # calcular a distância entre os pontos médios
        distance = int(math.sqrt((center2[0] - center1[0])**2 + (center2[1] - center1[1])**2))
        # desenhar uma linha entre os pontos médios
        cv2.line(frame, center1, center2, (0, 0, 255), 2)
        # adicionar um texto com a distância entre os pontos médios na imagem
        cv2.putText(frame, f"Distância: {distance} pixels", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # exibir o frame com os retângulos desenhados e a distância medida
    cv2.imshow('frame', frame)

    # esperar por uma tecla pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# liberar a câmera e fechar as janelas abertas
cap.release()
cv2.destroyAllWindows()








