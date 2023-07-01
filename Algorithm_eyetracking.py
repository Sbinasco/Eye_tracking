import numpy as np
import cv2 as cv2
import sys, os
import time
from datetime import datetime



# frame = cv2.imread('/home/cornelius/Desktop/new_scripts/Rangos_IR/Hamilton-Navarro-Jr/izq_izq.png')
frame = cv2.imread('D:/desktop/new_scripts/Rangos_IR/Hamilton-Navarro-Jr/izq_izq.png')

frame = frame[:,:,2]
cv2.imshow('normal frame',frame)


kernel_1 = np.array([[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]],dtype = np.uint8)
kernel_2 = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],dtype = np.uint8)


# ---- INICIO DEL ALGORITMO ----
tik = time.time()


# ---- NORMALIZACIÓN ----
normalizedImg = np.zeros((800, 800))
#frame = frame[70:360,0:420] # recorte salva
#frame = frame[70:300,0:450] # recorte rodrigo
#frame = frame[0:300,50:500] # crte daniella
#frame = frame[0:250,00:450] # corte francisco
#frame = frame[50:,0:500]
frame = frame[:,0:450]

normalizedImg = cv2.normalize(frame,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
cv2.imshow('normal',normalizedImg)

# ---- UMBRALIZACIÓN ----
_, img = cv2.threshold(normalizedImg, 65, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('normal',img)


# ---- DILATACIÓN ----
img = cv2.dilate(img, kernel=kernel_2, iterations=2) # Horizontal
img = cv2.dilate(img, kernel=kernel_1, iterations=12) # Vertical
cv2.imshow('dilatacion',img)

img = cv2.rectangle(img,(0,0),(frame.shape[1],frame.shape[0]),0,2) # Borde Negro

# ---- EROSIÓN ----
img = cv2.erode(img, kernel=kernel_1, iterations=18) # Vertical
img = cv2.erode(img, kernel=kernel_2, iterations=6) # Horizontal
cv2.imshow('erosion',img)


# ---- PRIMERA DETECCIÓN DE OBJETOS ---- : Objeto sin ruido
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
# tamaños de objetos   |||| regularizando indices
sizes = stats[1:, -1];      nb_components = nb_components - 1
min_size = 3000
# mascara
img2 = np.zeros((output.shape),dtype=np.uint8)
# rellenando mascara con objetos mayores a umbral 
for i in range(0, nb_components):
    if sizes[i] >= min_size:
        img2[output == i + 1] = 255

active_pixels = np.stack(np.where(img2))
parpadeo = False


try:    
    top_left = np.min(active_pixels, axis=1).astype(np.int32)
    bottom_right = np.max(active_pixels, axis=1).astype(np.int32)

    print(bottom_right[0]-top_left[0])
    print(bottom_right[1]-top_left[1])
    
    img_crop = img2[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]] # se extraen los objetos

    # ---- 
    if img_crop.shape[0] > 110:
        parpadeo = False
        #myKit.servo[1].angle = 70

    else:
        parpadeo = True
        pos = 'Parpadeo'
        #myKit.servo[1].angle = 110
        tek = time.time()       
    
    
    # ---- ¿Sólo pupila o con línea de pestañas? ----
    if img_crop.shape[1] >= 270:
        solo_pupila = False
        # cortando en el tercio inferior
        region = round((1/3)*img_crop.shape[0])
        img_recrop = img_crop[-region:,:]
        
        # =============================================================================
        #  objeto mas grande (2,4)
        # =============================================================================
        nb_components2, output2, stats2, centroids2 = cv2.connectedComponentsWithStats(img_recrop, connectivity=8)
        sizes2 = stats2[1:, -1]; nb_components2 = nb_components2 - 1
        # objeto mas grande 
        max_size = max(sizes2)
        
        # mascara crop
        img22 = np.zeros((output2.shape),dtype=np.uint8)
        
        # rellenando mascara con el objeto mas grande
        for i in range(0, nb_components2):
            if sizes2[i] == max_size:
                img22[output2 == i + 1] = 255
        
        #cv2.imshow('solo pup', img22)

    else:
        solo_pupila = True
        
        # Centroide
        mass_x, mass_y = np.where(img_crop > 0)
        cent_x = np.average(mass_x)
        cent_y = np.average(mass_y)
        print('Pupila')


    # =============================================================================
    # reubicando centroides
    # =============================================================================

    if solo_pupila:
        y = cent_y ; x = cent_x
    else:
        x = centroids2[np.argmax(sizes2, axis=0)+1][1] + (bottom_right-top_left)[0]*1/3
        y = centroids2[np.argmax(sizes2, axis=0)+1][0]

        
    y = y + top_left[1]
    x = x + top_left[0]

    print("posición horizontal",y)     

    img_aux = np.ones((img2.shape[0],img2.shape[1]))*255 ## MARIEL
    img_aux[:,:] = normalizedImg
    #img_final = cv2.line(img_aux, (int(y+top_left[0]),0), (int(y+top_left[0]),256), (255,0,255), 4)
    img_final = cv2.circle(normalizedImg, (int(y),int(x)), radius=10, color=(255, 0, 0), thickness=30)

except:
    parpadeo = True
    img_final = cv2.circle(normalizedImg, (int(y),int(x)), radius=10, color=(255, 0, 0), thickness=30)

print("parpadeo: ", parpadeo)
tok = time.time()
# now = datetime.now()
# current_time = now.strftime("%H:%M:%S")

# Display the resulting frame
cv2.imshow('FINAL', img_final)
print(tok-tik)

# When everything done, release the capture
cv2.waitKey(0)
cv2.destroyAllWindows()
