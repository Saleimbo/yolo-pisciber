import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw

# ------------------ 1. Rutas de los modelos -------------------------------
# Ajusta las rutas si tus .pt tienen otro nombre o ubicación
MODEL_PATH_BIG = "best.pt"      # modelo para peces grandes
MODEL_PATH_SMALL = "best.pt"  # modelo para peces pequeños

# ------------------ 2. Cargar modelos (con caché) -------------------------
@st.cache_resource
def load_model(path: str):
    """Carga y devuelve un modelo YOLO; cacheado por ruta."""
    model = YOLO(path)
    model.fuse()  # optimiza para CPU
    return model

# ------------------ 3. UI -------------------------------------------------
st.set_page_config(page_title="Vision Pisciber", page_icon="🐟", layout="centered")
st.title("🐟 Contador de peces")

st.markdown("""
Selecciona el tipo de pez y después sube una foto de la cubeta
para obtener el número total de peces detectados.
""")

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

# --- 3.2 Subida de imagen -------------------------------------------------
uploaded_file = st.file_uploader("Imagen (JPG o PNG)", type=["jpg", "jpeg", "png"])

# ------------------ 4. Inferencia ----------------------------------------
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Imagen cargada", use_container_width=True)

    # Seleccionar el modelo correspondiente y cargarlo (cacheado)
    model_path = MODEL_PATH_BIG if st.session_state.model_choice == "big" else MODEL_PATH_SMALL
    model = load_model(model_path)

    with st.spinner("Contando peces…"):
        results = model.predict(
            img,
            conf=0.5,  # umbral de confianza
            iou=0.4,   # umbral IoU para NMS
            save_txt=True,
            save_conf=False,
            save=True,
            exist_ok=True
        )[0]
        fish_count = len(results.boxes)

    st.success(f"**Peces detectados: {fish_count}**")

    # (opcional) mostrar bounding‑boxes
    if st.checkbox("Mostrar detecciones"):
            from PIL import ImageDraw as _ImageDraw
            annotated = img.copy()
            draw = _ImageDraw.Draw(annotated)
            for box in results.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box[:4]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                r = 4  # radio del punto
                draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(0, 255, 0))

            st.image(annotated, caption="Detecciones", use_container_width=True)


