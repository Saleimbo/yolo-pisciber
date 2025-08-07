import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import os

# Asegura dependencias mínimas (OpenCV para ultralytics)
# os.system("pip install opencv-python-headless --force-reinstall")

# ------------------ 1. Rutas de los modelos -------------------------------
# Ajusta las rutas si tus .pt tienen otro nombre o ubicación
MODEL_PATH_BIG = "best.pt"      # modelo para peces grandes
MODEL_PATH_SMALL = "gamba.pt"    # modelo para peces pequeños

# ------------------ 2. Cargar modelos (con caché) -------------------------
@st.cache_resource
def load_model(path: str):
    """Carga y devuelve un modelo YOLO; cacheado por ruta."""
    model = YOLO(path)
    model.fuse()  # optimiza para CPU
    return model

# ------------------ 3. UI -------------------------------------------------
st.set_page_config(page_title="Visión Pisciber", page_icon="🐟", layout="centered")
st.title("🐟 Contador de peces")

st.markdown(
    """
Selecciona el tipo de pez y después sube **una o varias** fotos de la cubeta
para obtener el número total de peces detectados en cada una.
"""
)

# --- 3.1 Selector de modelo (dos pulsadores) -----------------------------
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "big"  # valor por defecto

col1, col2 = st.columns(2)
with col1:
    if st.button("🐋 Peces grandes", use_container_width=True):
        st.session_state.model_choice = "big"
with col2:
    if st.button("🐟 Peces pequeños", use_container_width=True):
        st.session_state.model_choice = "small"

# Indicador visual del modelo seleccionado
modelo_activo = "Peces grandes" if st.session_state.model_choice == "big" else "Peces pequeños"
st.info(f"**Modelo seleccionado:** {modelo_activo}")

# --- 3.2 Subida de imágenes ------------------------------------------------
# Usamos una clave dinámica para poder "reiniciar" el widget cuando se pulse limpiar
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

uploaded_files = st.file_uploader(
    "Imagen (JPG o PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key=f"images_{st.session_state.uploader_key}",
)

# --- 3.3 Botón para limpiar imágenes --------------------------------------
if st.button("🗑️ Limpiar imágenes", use_container_width=True):
    st.session_state.uploader_key += 1  # cambia la clave => nuevo widget vacío
    st.stop()

# Opción de mostrar detecciones (global para todas las imágenes)
show_detections = st.checkbox("Mostrar detecciones")

# ------------------ 4. Inferencia ----------------------------------------
if uploaded_files:
    # Seleccionar el modelo correspondiente y cargarlo (cacheado)
    model_path = MODEL_PATH_BIG if st.session_state.model_choice == "big" else MODEL_PATH_SMALL
    model = load_model(model_path)

    conteos = []  # guardará el número de peces por imagen

    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        img = Image.open(uploaded_file).convert("RGB")
        st.subheader(f"Imagen {idx}: {uploaded_file.name}")
        st.image(img, caption="Imagen cargada", use_container_width=True)

        with st.spinner("Contando peces…"):
            results = model.predict(
                img,
                conf=0.3,  # umbral de confianza
                iou=0.5,   # umbral IoU para NMS
            )[0]
            fish_count = len(results.boxes)
            conteos.append(fish_count)

        st.success(f"**Peces detectados: {fish_count}**")

        # (opcional) mostrar puntos en las detecciones
        if show_detections and results.boxes is not None:
            annotated = img.copy()
            draw = ImageDraw.Draw(annotated)
            for box in results.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box[:4]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                r = 4  # radio del punto
                draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(0, 255, 0))
            st.image(annotated, caption="Detecciones", use_container_width=True)

        # Separador visual entre imágenes
        if idx < len(uploaded_files):
            st.divider()

    if conteos:
        media_peces = sum(conteos) / len(conteos)
        st.divider()
        st.success(f" Media de peces detectados en {len(conteos)} imágenes: {media_peces:.0f}")

