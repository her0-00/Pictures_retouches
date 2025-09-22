import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io, uuid
from fpdf import FPDF
from datetime import datetime

st.set_page_config(page_title="Pictures Retouches", layout="wide")
st.markdown("<h1 style='text-align: center;'>Pictures Retouches V2</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Application web gratuite de retouche photo avancée : filtres artistiques, retouche visage et historisation des actions.</p>", unsafe_allow_html=True)
st.markdown("---")

# === Identifiant utilisateur & historique ===
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())[:8]
if "history" not in st.session_state:
    st.session_state.history = []

# === Fonctions de traitement ===
def to_bgr(img): return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
def to_rgb(img): return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def apply_blur(img, intensity, method):
    k = intensity * 2 + 1
    if method == "Gaussien": return cv2.GaussianBlur(img, (k, k), 0)
    elif method == "Médian": return cv2.medianBlur(img, k)
    elif method == "Bilatéral": return cv2.bilateralFilter(img, k, 75, 75)
    return img

def apply_pixelate(img, pixel_size):
    h, w = img.shape[:2]
    temp = cv2.resize(img, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

def apply_sketch(img, detail=5, contrast=1.0, style='soft', color=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (detail*2+1, detail*2+1), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    if style == 'hard': sketch = cv2.equalizeHist(sketch)
    elif style == 'comic': sketch = cv2.adaptiveThreshold(sketch, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
    sketch = cv2.convertScaleAbs(sketch, alpha=contrast, beta=0)
    if color:
        sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
        sketch = cv2.bitwise_and(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), sketch_rgb)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR) if not color else sketch

def apply_bw(img, contrast=1.0, grain=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.convertScaleAbs(gray, alpha=contrast, beta=0)
    if grain:
        noise = np.random.normal(0, 10, bw.shape).astype(np.uint8)
        bw = cv2.add(bw, noise)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

def rotate_flip(img, angle, flip_mode):
    h, w = img.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv2.warpAffine(img, rot_mat, (w, h))
    if flip_mode == "Horizontal": return cv2.flip(rotated, 1)
    elif flip_mode == "Vertical": return cv2.flip(rotated, 0)
    return rotated

def resize_image(img, width, height):
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return face_cascade.detectMultiScale(gray, 1.1, 4)

def apply_face_smoothing(img, intensity=30):
    faces = detect_faces(img)
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        blurred = cv2.bilateralFilter(face, 9, intensity, intensity)
        img[y:y+h, x:x+w] = blurred
    return img

def generate_pdf_report(params, user_id):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Rapport de retouche - Utilisateur {user_id}", ln=True)
    for key, value in params.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# === Interface ===
st.sidebar.title("⚙️Paramétrages")
uploaded_file = st.sidebar.file_uploader("📤 Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_bgr = to_bgr(image)
    result = img_bgr.copy()

    effect = st.sidebar.selectbox("🎨 Effet", ["Aucun", "Floutage", "Pixelisation", "Esquisse", "Noir & Blanc", "Rotation/Flip", "Redimensionnement", "Retouche Visage"])
    params = {"Effet": effect, "Date": datetime.now().strftime("%Y-%m-%d %H:%M"), "User": st.session_state.user_id}

    if effect == "Floutage":
        intensity = st.sidebar.slider("Intensité", 1, 20, 5)
        method = st.sidebar.selectbox("Méthode", ["Gaussien", "Médian", "Bilatéral"])
        result = apply_blur(img_bgr, intensity, method)
        params.update({"Intensité": intensity, "Méthode": method})

    elif effect == "Pixelisation":
        pixel_size = st.sidebar.slider("Taille des blocs", 5, 100, 20)
        result = apply_pixelate(img_bgr, pixel_size)
        params.update({"Pixel size": pixel_size})

    elif effect == "Esquisse":
        detail = st.sidebar.slider("Détail", 1, 10, 5)
        contrast = st.sidebar.slider("Contraste", 0.5, 2.0, 1.0)
        style = st.sidebar.selectbox("Style", ["soft", "hard", "comic"])
        color = st.sidebar.checkbox("Couleur")
        result = apply_sketch(img_bgr, detail, contrast, style, color)
        params.update({"Détail": detail, "Contraste": contrast, "Style": style, "Couleur": color})

    elif effect == "Noir & Blanc":
        contrast = st.sidebar.slider("Contraste", 0.5, 2.0, 1.0)
        grain = st.sidebar.checkbox("Grain")
        result = apply_bw(img_bgr, contrast, grain)
        params.update({"Contraste": contrast, "Grain": grain})

    elif effect == "Rotation/Flip":
        angle = st.sidebar.slider("Angle", -180, 180, 0)
        flip_mode = st.sidebar.radio("Miroir", ["Aucun", "Horizontal", "Vertical"])
        result = rotate_flip(img_bgr, angle, flip_mode)
        params.update({"Angle": angle, "Flip": flip_mode})

    elif effect == "Redimensionnement":
        width = st.sidebar.number_input("Largeur", 100, 2000, 800)
        height = st.sidebar.number_input("Hauteur", 100, 2000, 600)
        result = resize_image(img_bgr, int(width), int(height))
        params.update({"Largeur": width, "Hauteur": height})

    elif effect == "Retouche Visage":
        intensity = st.sidebar.slider("Lissage", 10, 100, 30)
        result = apply_face_smoothing(img_bgr, intensity)
        params.update({"Lissage visage": intensity})

    # Affichage
    col1, col2 = st.columns(2)
    col1.image(image, caption="Image originale", use_container_width=True)
    col2.image(to_rgb(result), caption="Image modifiée", use_container_width=True)

    # Historique
    st.session_state.history.append(params)

    # Téléchargement image
    result_pil = Image.fromarray(to_rgb(result))
    buf = io.BytesIO()
    result_pil.save(buf, format="PNG")
    st.download_button("📥 Télécharger l'image", data=buf.getvalue(), file_name="image_modifiée.png", mime="image/png")

    # Export PDF
    pdf_bytes = generate_pdf_report(params, st.session_state.user_id)
    st.download_button("🧾 Télécharger le rapport PDF", data=pdf_bytes, file_name=f"rapport_{st.session_state.user_id}.pdf", mime="application/pdf")

    # Historique affiché
    with st.expander("📚 Historique de retouches"):
        for i, entry in enumerate(st.session_state.history[::-1]):
            st.markdown(f"**{i+1}.** Effet: `{entry['Effet']}` — Date: `{entry['Date']}` — ID: `{entry['User']}`")
            for k, v in entry.items():
                if k not in ["Effet", "Date", "User"]:
                    st.write(f"- {k}: {v}")



