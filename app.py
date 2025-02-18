import streamlit as st
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

st.title("📸 Analyse qualité des chants")

# 📤 Upload d'image via Streamlit
uploaded_file = st.file_uploader("Choisissez une image", type=["png", "jpg", "jpeg", "bmp", "tiff"])

if uploaded_file:
    # Lire l'image avec PIL
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Afficher l'image originale
    st.image(image, caption="Image originale")   

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    #cropper l'image pour supprimer tout le noir autour
    #seuil pour supprimer le bruit tout autour
    gray2 = np.where(gray < 100 , 255, gray)

    # Threshold the image to create a binary mask
    _, thresh = cv2.threshold(gray2, 250, 255, cv2.THRESH_BINARY_INV)

    # Find the contours of the non-white regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the bounding rectangle for the largest contour
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # Crop the image
    gray = gray[y:y+h, x:x+w]


    # Appliquer un seuillage binaire
    _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)

    # Détecter les contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrer les contours pour ne garder que les lignes verticales
    vertical_lines = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / float(w)
        if aspect_ratio > 10:  # Ajustez ce seuil selon vos besoins
            vertical_lines.append((x, y, w, h))

    # Trier les lignes verticales par position x
    vertical_lines.sort(key=lambda line: line[0])

    # Isoler chaque rectangle entre les lignes verticales
    rectangles = []
    previous_x = 0

    for line in vertical_lines:
        x, y, w, h = line
        if x > previous_x:  # Vérifier que la largeur est non nulle
            # Isoler la partie de l'image entre previous_x et x 
            rectangle = gray[:, previous_x:x]
            if rectangle.size > 1000000:
                rectangles.append(rectangle)
        previous_x = x + w

    # Ajouter la dernière section de l'image
    if previous_x < gray.shape[1]:  # Vérifier que la largeur est non nulle
        rectangle = gray[:, previous_x:]
        if rectangle.size > 1000000:
            rectangles.append(rectangle)


    # Afficher les rectangles isolés
    for i, rect in enumerate(rectangles):

        # Initialiser des listes pour stocker les indices des lignes et colonnes à conserver
        lignes_a_conserver = []
        colonnes_a_conserver = []

        # Seuil pour considérer un pixel comme "sombre"
        seuil_sombre = 50  # Vous pouvez ajuster ce seuil si nécessaire

        #Garde les lignes et colonnes sans trop de proportion noire (élimine les bords noirs)
        lignes_a_conserver = [k for k in range(rect.shape[0]) if np.sum(rect[k, :] < 50) / rect.shape[1] < 0.8]
        colonnes_a_conserver = [j for j in range(rect.shape[1]) if np.sum(rect[:, j] < 50) / rect.shape[0] < 0.3]

        # Créer une nouvelle image avec les lignes et colonnes à conserver
        image_filtre = rect[np.ix_(lignes_a_conserver, colonnes_a_conserver)]

        #supprime le haut et le bas pour enlever les écritures
        # Calculer la hauteur
        hauteur, largeur = image_filtre.shape[:2]
        # Recadrer l'image pour supprimer le 1/x supérieur et inférieur
        image_filtre = image_filtre[hauteur // 10:(hauteur - hauteur // 10), :]

        # Appliquer un seuillage binaire
        _, binary_image = cv2.threshold(image_filtre, 127, 255, cv2.THRESH_BINARY)

        # Calculer le taux de noir sur total
        total_pixels = binary_image.size
        black_pixels = np.sum(binary_image == 0)

        if total_pixels > 0:  # Eviter la division par zéro
            black_ratio = round((black_pixels / total_pixels)*100,2)
            st.write(f"📏 **Contour {i+1} : Taux de noir = {black_ratio} %**")

        # Création de trois colonnes (une colonne vide au milieu pour l'espace)
        col1, col_space, col2 = st.columns([1, 0.2, 1])  

        with col1:
            #st.image(rect, caption=f"Rectangle {i+1}")
            st.image(rect, width = 30)

        with col2:
            #st.image(binary_image, caption="Image Noir & Blanc")
            st.image(binary_image, width = 30)
