import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import streamlit as st

# Pengaturan perangkat
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

@st.cache_resource
def load_model():
    # Load checkpoint
    ckpt = torch.load("model.pth", map_location=DEVICE)
    classes = ckpt["classes"]

    # Inisialisasi arsitektur model (MobileNetV2)
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    
    # Load bobot/weights
    model.load_state_dict(ckpt["model"])
    model.eval().to(DEVICE)

    # Transformasi gambar
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return model, classes, tf

def skincare(label):
    tips = {
        "oily": [
            "Gunakan Cleanser jenis gel atau foaming",
            "Pilih Moisturizer oil-free / water-based",
            "Gunakan Sunscreen non-comedogenic"
        ],
        "dry": [
            "Gunakan Gentle cleanser yang menghidrasi",
            "Pilih Moisturizer dengan hyaluronic acid atau ceramide",
            "Hindari produk dengan kandungan alkohol tinggi"
        ],
        "normal": [
            "Gunakan Mild cleanser",
            "Gunakan Moisturizer ringan untuk menjaga kelembapan",
            "Gunakan Sunscreen secara rutin setiap pagi"
        ]
    }
    return tips.get(label.lower(), ["Tidak ada rekomendasi khusus."])

# --- TAMPILAN STREAMLIT ---
st.warning("⚠️ Pastikan wajah Anda terpapar cahaya yang cukup (terang) agar hasil analisis akurat.")
st.warning("⚠️ Catatan: Hasil analisis AI ini hanya bersifat referensi kecantikan dan bukan diagnosis medis resmi. Konsultasikan dengan dokter spesialis kulit (Dermatologis) untuk analisis profesional.")
st.set_page_config(page_title="Skin Types AI", page_icon="🧴")
st.title("🧴 Skin Types AI")
st.write("Gunakan kamera atau upload foto wajah untuk prediksi tipe kulit (Oily/Dry/Normal).")

# Inisialisasi variabel gambar
img = None

# Membuat tab agar tampilan rapi
tab1, tab2 = st.tabs(["📸 Ambil Foto", "📁 Upload File"])

with tab1:
    camera_file = st.camera_input("Ambil foto wajah")
    if camera_file:
        img = Image.open(camera_file).convert("RGB")

with tab2:
    upload_file = st.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"])
    if upload_file:
        img = Image.open(upload_file).convert("RGB")

# Logika Prediksi (Hanya berjalan jika ada gambar)
if img is not None:
    st.image(img, caption="Foto yang akan dianalisis", use_container_width=True)
    
    with st.spinner('Sedang menganalisis kulit Anda...'):
        model, classes, tf = load_model()
        
        # Preprocessing & Inference
        x = tf(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(x)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
            idx = int(probs.argmax())

        label = classes[idx]
        confidence = probs[idx] * 100

        # Menampilkan Hasil
        st.success(f"✅ Hasil Prediksi: **{label.upper()}** (Tingkat Keyakinan: {confidence:.2f}%)")

        st.divider()
        st.subheader("💡 Rekomendasi Skincare:")
        rekomendasi = skincare(label)
        for r in rekomendasi:
            st.write(f"- {r}")
else:
    st.info("Silakan ambil foto melalui kamera atau upload gambar untuk memulai.")