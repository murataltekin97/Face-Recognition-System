import os, json
import sys
import numpy as np
import cv2
from PIL import Image

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
global user
user = input("Adınızı giriniz: ")
yas = input("Yaşınızı giriniz: ")
user = user +" Yas "+yas

print("Kameraya bakın ve bekleyin...")
say =0
os.mkdir("dataset/"+ user)

while(True):
    ret, cerceve= cam.read()
    cerceve = cv2.flip(cerceve,1)
    gri = cv2.cvtColor(cerceve, cv2.COLOR_BGR2GRAY)   
    
    faces = face_detector.detectMultiScale(gri, 1.5, 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(cerceve, (x,y), (x+w,y+h), (255,0,0),2)
        say += 1
        path = "dataset/"+user+"/"
        cv2.imwrite(path + str(say)+ ".jpg",gri[y:y+h , x:x+w])
        cv2.putText(cerceve, "LUTFEN BEKLEYINIZ...", (x+5,y-5),font,1,(255,255,255),2)
        cv2.imshow("DATA",cerceve)
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif say >= 50:
        break
    
cam.release()
cv2.destroyAllWindows()
print("GORUNTUNUZ ALGILANDI... \n Birazdan tekrar kamera calısacak ve yuzunuzu tahminleyecek")

yol = "dataset"
tani = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

def getImagesandLabels(yol):
    faceSamples = []
    ids= []
    labels= []
    klasorler= os.listdir(yol)
    sozluk= {}
    
    for i, k1 in enumerate(klasorler):
        sozluk[k1] = int(i)
    f = open("ids.json","w")
    a= json.dump(sozluk,f)
    f.close()
    
    for k1 in klasorler:
        for res in os.listdir(os.path.join(yol,k1)):
            PIL_img= Image.open(os.path.join(yol,k1,res)).convert("L")
            img_numpy = np.array(PIL_img, "uint8")
            id = int(sozluk[k1])
            faces= detector.detectMultiScale(img_numpy)
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h , x:x+w])
                ids.append(id)
    return faceSamples, ids

faces, ids = getImagesandLabels(yol)
tani.train(faces,np.array(ids))
tani.write("trainer.yml")

tani = cv2.face.LBPHFaceRecognizer_create()
tani.read("trainer.yml")
cascadePath = cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
Id=0

sozluk = {}
names = []
dosya = open("ids.json","r")
sozluk = json.load(dosya)

for key, value in sozluk.items():
    names.append(key)
    
cam = cv2.VideoCapture(0)
sayac =0   
while True:
    ret, cerceve= cam.read()
    cerceve = cv2.flip(cerceve,1)
    gri = cv2.cvtColor(cerceve,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(gri, 1.5, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(cerceve,(x,y),(x+w, y+h),(0,255,0),2  )
        
        Id, oran = tani.predict(gri[y:y+h, x:x+w])
        print(Id)
        if (oran <70):
            Id = names[Id]
        else:
            Id= "bilinmiyor"
        cv2.putText(cerceve, user, (x+5,y-5),font,1,(255,255,255),2)
    cv2.imshow("KAMERA",cerceve)
    k=cv2.waitKey(50) & 0xff
    sayac += 1
    if k == 27:
        break
    if sayac==200:
        break          
cam.release()
cv2.destroyAllWindows()
print("Program Sonlandı...")
 

