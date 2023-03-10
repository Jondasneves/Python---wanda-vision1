import cv2

CarregaAlgoritmo = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

imagem = cv2.imread('imagem 4.jpg')

imagemcinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

faces = CarregaAlgoritmo.detectMultiScale(imagemcinza, scaleFactor=1.1, minNeighbors=1, minSize=(30,30))

print(faces)

for(x,y,l,a) in faces:
    cv2.rectangle(imagem,(x,y),(x+l,y+a),(255,0,0), 2)

cv2.imshow('Faces',imagem)
cv2.waitKey()