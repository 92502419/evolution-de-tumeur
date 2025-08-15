import streamlit as st
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from io import BytesIO
import tempfile
import tensorflow as tf
from tensorflow.keras import layers, models

# === Configuration Streamlit ===
st.set_page_config(page_title="Analyse IRM avec AutoEncoder", layout="centered")
st.title("🧠 Analyse d'IRM Cérébrales avec AutoEncoder 3D")

# === Fonction pour charger un fichier NIfTI uploadé ===
def load_nifti_file(uploaded_file):
    try:
        # Utilisation d’un fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        img = nib.load(tmp_path)
        return img.get_fdata()
    except Exception as e:
        st.error(f"Erreur de chargement du fichier : {e}")
        return None

# === Fonction pour afficher des informations de base ===
def show_image_stats(img_data):
    st.write("**Dimensions :**", img_data.shape)
    st.write("**Valeur min :**", np.min(img_data))
    st.write("**Valeur max :**", np.max(img_data))
    st.write("**Valeur moyenne :**", np.mean(img_data))

    # Afficher une coupe centrale
    mid_slice = img_data.shape[2] // 2
    fig, ax = plt.subplots()
    ax.imshow(img_data[:, :, mid_slice], cmap="gray")
    ax.set_title(f"Coupe axiale - slice {mid_slice}")
    st.pyplot(fig)

# === Définition d’un AutoEncoder simple 3D ===
def build_autoencoder(input_shape):
    input_img = layers.Input(shape=input_shape)

    # Encodeur
    x = layers.Conv3D(16, (3, 3, 3), activation="relu", padding="same")(input_img)
    x = layers.MaxPooling3D((2, 2, 2), padding="same")(x)
    x = layers.Conv3D(8, (3, 3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling3D((2, 2, 2), padding="same")(x)

    # Décodeur
    x = layers.Conv3D(8, (3, 3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling3D((2, 2, 2))(x)
    x = layers.Conv3D(16, (3, 3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling3D((2, 2, 2))(x)

    decoded = layers.Conv3D(1, (3, 3, 3), activation="sigmoid", padding="same")(x)

    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder

# === Upload du fichier NIfTI ===
uploaded_file = st.file_uploader("Uploader un fichier NIfTI (.nii ou .nii.gz)", type=["nii", "nii.gz"])

if uploaded_file is not None:
    img_data = load_nifti_file(uploaded_file)

    if img_data is not None:
        show_image_stats(img_data)

        # Prétraitement des données pour le modèle
        if len(img_data.shape) == 3:
            data = img_data.astype("float32")
            data = (data - np.min(data)) / (np.max(data) - np.min(data))  # Normalisation
            data = np.expand_dims(data, axis=0)  # Ajout batch
            data = np.expand_dims(data, axis=-1)  # Ajout canal

            st.subheader("📈 Entraînement de l'AutoEncoder")
            autoencoder = build_autoencoder(data.shape[1:])
            history = autoencoder.fit(data, data, epochs=10, batch_size=1, verbose=1)

            st.success("✅ Entraînement terminé")

            # Reconstruction
            reconstructed = autoencoder.predict(data)
            mse = mean_squared_error(data.flatten(), reconstructed.flatten())
            st.write(f"**Erreur quadratique moyenne (MSE)** : {mse:.5f}")

            # Affichage comparaison original vs reconstruit
            mid = data.shape[3] // 2
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(data[0, :, :, mid, 0], cmap="gray")
            axes[0].set_title("Image originale")
            axes[1].imshow(reconstructed[0, :, :, mid, 0], cmap="gray")
            axes[1].set_title("Image reconstruite")
            st.pyplot(fig)

            # Option de téléchargement du modèle
            with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_model:
                autoencoder.save(tmp_model.name)
                st.download_button("📥 Télécharger le modèle entraîné (.h5)", tmp_model.read(), file_name="autoencoder_3d_model.h5")
