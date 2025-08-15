import streamlit as st
import os
import io
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from fpdf import FPDF
import tempfile

st.set_page_config(layout="wide", page_title="Analyse IRM Cérébrales")
st.title(" Analyse d'IRM Cérébrales avec AutoEncoder 3D")

# --- Importation
st.sidebar.header("Choisissez la source des données :")
import_method = st.sidebar.radio("Méthode d'importation :", ["Uploader un ou plusieurs fichiers", "Charger depuis dossier local"])

loaded_images = []
image_names = []

if import_method == "Uploader un ou plusieurs fichiers":
    uploaded_files = st.sidebar.file_uploader("Uploader les fichiers NIfTI (.nii ou .nii.gz)", type=["nii", "nii.gz"], accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        try:
            img = nib.load(io.BytesIO(uploaded_file.read()))
            loaded_images.append(img.get_fdata())
            image_names.append(uploaded_file.name)
            st.success(f" Fichier {uploaded_file.name} chargé avec succès.")
        except Exception as e:
            st.error(f" Erreur chargement fichier {uploaded_file.name} : {e}")
elif import_method == "Charger depuis dossier local":
    folder_path = st.sidebar.text_input("Entrez le chemin complet du dossier contenant vos fichiers NIfTI :")
    if folder_path and os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".nii") or filename.endswith(".nii.gz"):
                path = os.path.join(folder_path, filename)
                img = nib.load(path)
                loaded_images.append(img.get_fdata())
                image_names.append(filename)
                st.success(f" Fichier {filename} chargé avec succès.")

# === Visualisation
if loaded_images:
    st.subheader(" Visualisation d'une coupe transversale")
    max_index = len(loaded_images) - 1
    if max_index > 0:
        index = st.slider("Choisir l'image à afficher :", 0, max_index, 0)
    else:
        index = 0
    slice_idx = loaded_images[index].shape[2] // 2
    fig, ax = plt.subplots()
    ax.imshow(loaded_images[index][:, :, slice_idx], cmap="gray")
    ax.set_title(f"Coupe axiale - {image_names[index]}")
    st.pyplot(fig)

    # === Statistiques
    st.subheader(" Statistiques descriptives")
    for i, img in enumerate(loaded_images):
        st.write(f"*{image_names[i]}*")
        st.write(f"Forme : {img.shape}")
        st.write(f"Min : {np.min(img):.2f}, Max : {np.max(img):.2f}, Moyenne : {np.mean(img):.2f}, Écart-type : {np.std(img):.2f}")

        # Histogramme
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.hist(img.flatten(), bins=50, color='skyblue')
        ax.set_title("Histogramme des intensités")
        st.pyplot(fig)

        # Heatmap
        mid_slice = img[:, :, img.shape[2] // 2]
        fig, ax = plt.subplots()
        sns.heatmap(mid_slice, cmap="viridis", ax=ax)
        ax.set_title("Heatmap 2D (coupe centrale)")
        st.pyplot(fig)

    # === Prétraitement des données
    st.subheader(" AutoEncoder 3D - Entraînement automatique")

    standardized_imgs = []
    target_shape = (64, 64, 64)
    for img in loaded_images:
        padded = np.zeros(target_shape)
        slices = tuple(slice(0, min(s, 64)) for s in img.shape[:3])
        padded[slices] = img[slices]
        standardized_imgs.append(padded)

    X = np.array(standardized_imgs)[..., np.newaxis]  # (n, 64, 64, 64, 1)
    X = X / np.max(X)
    if len(X) < 2:
        st.warning(" Pas assez d'images pour entraîner. Ajouter au moins 2 fichiers.")
        st.stop()

    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    def build_autoencoder(input_shape=(64, 64, 64, 1)):
        input_img = layers.Input(shape=input_shape)
        x = layers.Conv3D(16, 3, activation='relu', padding='same')(input_img)
        x = layers.MaxPooling3D(2, padding='same')(x)
        x = layers.Conv3D(8, 3, activation='relu', padding='same')(x)
        encoded = layers.MaxPooling3D(2, padding='same')(x)

        x = layers.Conv3D(8, 3, activation='relu', padding='same')(encoded)
        x = layers.UpSampling3D(2)(x)
        x = layers.Conv3D(16, 3, activation='relu', padding='same')(x)
        x = layers.UpSampling3D(2)(x)
        decoded = layers.Conv3D(1, 3, activation='sigmoid', padding='same')(x)

        autoencoder = models.Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        return autoencoder

    autoencoder = build_autoencoder()
    with st.spinner(" Entraînement de l'autoencoder en cours..."):
        history = autoencoder.fit(X_train, X_train, epochs=10, batch_size=2, shuffle=True, validation_data=(X_test, X_test))

    st.success(" Entraînement terminé")

    # ===  Courbe de perte
    st.subheader(" Courbe de perte")
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='Entraînement')
    ax.plot(history.history['val_loss'], label='Validation')
    ax.set_title("Évolution de la perte")
    ax.set_xlabel("Époque")
    ax.set_ylabel("Perte")
    ax.legend()
    st.pyplot(fig)

    # === Reconstruction
    st.subheader(" Reconstruction et détection d'anomalies")
    if X_test.shape[0] > 1:
        idx = st.slider("Index image test", 0, X_test.shape[0] - 1, 0)
    else:
        idx = 0

    original = X_test[idx, :, :, :, 0]
    reconstructed = autoencoder.predict(X_test[idx:idx+1])[0, :, :, :, 0]
    mse = np.mean((original - reconstructed) ** 2)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original[:, :, 32], cmap="gray")
    axs[0].set_title("Image originale")
    axs[1].imshow(reconstructed[:, :, 32], cmap="gray")
    axs[1].set_title("Image reconstruite")
    st.pyplot(fig)

    st.metric(label="Erreur de reconstruction (MSE)", value=f"{mse:.6f}")

    if mse > 0.01:
        st.error(" Anomalie détectée : reconstruction de mauvaise qualité")
    else:
        st.success(" Aucune anomalie détectée")

    # === Rapport PDF
    st.subheader(" Télécharger le rapport PDF")
    if st.button(" Générer le rapport PDF"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=14)
            pdf.cell(200, 10, txt="Rapport d'Analyse IRM Cérébrale", ln=True, align='C')
            pdf.set_font("Arial", size=11)

            for i, img in enumerate(loaded_images):
                pdf.cell(200, 10, txt=f"{image_names[i]} - Moyenne : {np.mean(img):.2f}, Écart-type : {np.std(img):.2f}", ln=True)

            pdf.cell(200, 10, txt=f"Erreur reconstruction test : {mse:.6f}", ln=True)
            pdf.output(tmp_pdf.name)
            st.success("✅ Rapport généré avec succès")
            st.download_button("📄 Télécharger le rapport", data=open(tmp_pdf.name, "rb"), file_name="rapport_IRM.pdf", mime="application/pdf")
