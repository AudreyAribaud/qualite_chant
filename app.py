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
    st.image(image, caption="Image originale", use_container_width=True)

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Supprimer les bords noirs
    #remplace le noir par du blanc
    gray = np.where(gray < 10, 255, gray)

    # Seuillage pour créer un masque binaire
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    # Détecter les contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    # Crop l'image
    gray = gray[y:y+h, x:x+w]  

    # Détection de lignes verticales
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vertical_lines = [cv2.boundingRect(c) for c in contours if h / float(w) > 9]  # Ajuster ce seuil si besoin
    vertical_lines.sort(key=lambda line: line[0])

    # Découper en sous-images
    rectangles = []
    prev_x = 0
    for x, y, w, h in vertical_lines:
        if x > prev_x:
            rectangles.append(gray[:, prev_x:x])
        prev_x = x + w
    if prev_x < gray.shape[1]:
        rectangles.append(gray[:, prev_x:])

    # Afficher chaque segment
    for i, rect in enumerate(rectangles):
        if rect.size > 0:
            # Ajuster les seuils si besoin
            lignes_a_conserver = [k for k in range(rect.shape[0]) if np.sum(rect[k, :] < 50) / rect.shape[1] < 0.8]
            colonnes_a_conserver = [j for j in range(rect.shape[1]) if np.sum(rect[:, j] < 50) / rect.shape[0] < 0.5]

            image_filtre = rect[np.ix_(lignes_a_conserver, colonnes_a_conserver)]
            image_filtre = image_filtre[image_filtre.shape[0] // 10 : -image_filtre.shape[0] // 10, :]

            _, binary_image = cv2.threshold(image_filtre, 127, 255, cv2.THRESH_BINARY)
            black_ratio = round(np.sum(binary_image == 0) / binary_image.size * 100, 2) if binary_image.size > 0 else 0

            st.write(f"📏 **Contour {i+1} : Taux de noir = {black_ratio} %**")

            # Création de trois colonnes (une colonne vide au milieu pour l'espace)
            col1, col_space, col2 = st.columns([1, 0.2, 1])  

            with col1:
                st.image(rect, caption=f"Rectangle {i+1}", width=300)

            with col2:
                st.image(binary_image, caption="Image Noir & Blanc", width=300)
