
```sh
# Load model
decoder = MiyawakiDecoder()
decoder.load_model("miyawaki_contrastive_clip.pth")

# Use for inference
fmri_input = your_fmri_data  # Shape: (1, 967)
clip_embedding = decoder.fmri_encoder(fmri_input)  # Shape: (1, 512)

# Compare dengan CLIP image embeddings
image_embeddings = clip_model.encode_image(images)  # Shape: (N, 512)
similarities = torch.matmul(clip_embedding, image_embeddings.T)  # Shape: (1, N)
```


🔍 CLIP MODEL TIDAK TERSIMPAN DALAM .PTH
❌ YANG TIDAK ADA DALAM .PTH:
🚫 CLIP Model Components:

✅ CLIP Vision Encoder - TIDAK tersimpan
✅ CLIP Text Encoder - TIDAK tersimpan
✅ CLIP Preprocessing - TIDAK tersimpan
✅ YANG ADA DALAM .PTH:
Mari saya tunjukkan dengan kode:

Read lines 295-301
runembedding.py
miyawaki4
Sekarang mari lihat bagaimana CLIP diinisialisasi:

Read lines 122-139
runembedding.py
miyawaki4
📦 JAWABAN LENGKAP: CLIP TIDAK TERSIMPAN DALAM .PTH
❌ CLIP MODEL TIDAK ADA DALAM .PTH:
📁 Contents of miyawaki_contrastive_clip.pth:

{
    'fmri_encoder_state_dict': {...},  # ✅ HANYA fMRI encoder weights
    'scaler': StandardScaler()         # ✅ HANYA data normalization
}
🚫 YANG TIDAK TERSIMPAN:

❌ CLIP Vision Encoder (ViT-B/32)
❌ CLIP Text Encoder
❌ CLIP Preprocessing parameters
❌ CLIP model weights
🔄 MENGAPA CLIP TIDAK DISIMPAN:
1. 🔒 CLIP Model di-Freeze:

# Line 127-129
for param in self.clip_model.parameters():
    param.requires_grad = False  # CLIP weights tidak berubah
2. 📥 CLIP Dimuat Ulang Setiap Kali:

# Line 125 - setiap kali load model
self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
3. 🎯 Hanya fMRI Encoder yang Ditraining:

# Line 163 - hanya fMRI encoder yang dioptimize
optimizer = optim.Adam(self.fmri_encoder.parameters(), lr=lr, weight_decay=1e-4)
🔄 PROSES LOAD MODEL:
📋 Ketika Load Model:

def load_model(self, filepath):
    checkpoint = torch.load(filepath)
    
    if self.fmri_encoder is None:
        self.initialize_models()  # ← CLIP dimuat ulang di sini!
    
    self.fmri_encoder.load_state_dict(checkpoint['fmri_encoder_state_dict'])
    self.scaler = checkpoint['scaler']
🔄 initialize_models() dipanggil yang akan:

# Download/load CLIP model dari internet/cache
self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
📊 SIZE COMPARISON:
📦 File Sizes:

💡 IMPLICATIONS:
✅ ADVANTAGES:

Smaller file size (18.5 MB vs 350+ MB)
Always latest CLIP (auto-download updates)
Consistent CLIP version across environments
Only custom weights saved (fMRI encoder)
⚠️ REQUIREMENTS:

Internet connection untuk first-time CLIP download
CLIP library harus terinstall
CLIP cache akan tersimpan di system (~350 MB)
🎯 SUMMARY:
🔑 KEY POINTS:

✅ .pth hanya berisi fMRI encoder + scaler
❌ CLIP model TIDAK tersimpan dalam .pth
🔄 CLIP dimuat ulang setiap kali load model
🔒 CLIP di-freeze, tidak berubah selama training
📦 File size kecil karena hanya custom weights
🧠 ANALOGI:
Seperti menyimpan "adapter" yang menghubungkan brain ke CLIP, tapi CLIP-nya sendiri tetap diambil dari "perpustakaan" (OpenAI) setiap kali dibutuhkan.

Model .pth ini adalah "brain-to-CLIP translator" bukan "complete brain-image system"! 🧠→🔗→🖼️