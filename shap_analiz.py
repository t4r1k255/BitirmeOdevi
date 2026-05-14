import numpy as np
import pandas as pd
import joblib
import shap
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # ekran açmadan kaydetmek için (runner ile de sorunsuz)
import matplotlib.pyplot as plt
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

from sklearn.model_selection import train_test_split

# ---- AYARLAR ----
MODEL_YOLU = "models/xgb_pipeline.joblib"
VERI_YOLU = "data/ModelEgitimSeti.xlsx"
SAYFA = "ModelEgitimSeti"           # sende neyse: "EgitimSeti" veya "ModelEgitimSeti"
ETIKET = "Outcome_6Ay"

etiket_haritasi = {"IsFeshi": 0, "MevkiDevami": 1, "Terfi": 2}

# Modelde kullanmayacağımız tanımsal kolonlar (sende aynı mantık)
MODEL_DISI = [
    "Yas","DepNo","SiraNo","CalisanNo","YasBand","Cinsiyet","MedeniHal",
    "MailAdresi","TelefonNo","Ad","Soyad",
    "CalismaTuru","CalismaSekli",
    "Departman","MeslekiRolu",
    "CalAnketKatilim"
]

# ---- 1) Modeli yükle ----
pipe = joblib.load(MODEL_YOLU)
on_isleme = pipe.named_steps["on_isleme"]
model = pipe.named_steps["model"]
ozellik_uret = pipe.named_steps["ozellik_uret"]
# ---- 2) Veriyi yükle ----
df = pd.read_excel(VERI_YOLU, sheet_name=SAYFA, na_values=["x", "X", " x ", "X "])

# etiket filtre + map
df = df[df[ETIKET].isin(etiket_haritasi.keys())].copy()
y = df[ETIKET].map(etiket_haritasi)

# feature set (train_model ile aynı mantık)
model_disi = [c for c in MODEL_DISI if c in df.columns]
X = df.drop(columns=[ETIKET], errors="ignore").copy()

# kategorik kolonları string yap (pipeline uyumu için)
for c in X.select_dtypes(include=["object"]).columns:
    X[c] = X[c].astype("string").str.strip()

# ---- 3) Train/Test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- 4) Feature names üret ----
# sklearn >=1.0 için çoğu zaman çalışır:
try:
    feature_names = on_isleme.get_feature_names_out()
    feature_names = list(feature_names)
except Exception as e:
    # fallback: feature name alamazsak en azından index bulmak zorlaşır
    feature_names = None
    print("Feature name alınamadı:", e)

# ---- 5) X_test'i dönüştür ----
X_test_fe = ozellik_uret.transform(X_test)
X_test_trans = on_isleme.transform(X_test_fe)

# Sparse -> dense (SHAP için)
try:
    X_test_dense = X_test_trans.toarray()
except Exception:
    X_test_dense = np.asarray(X_test_trans)

# ---- 6) SHAP (multi-class) ----
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_dense)  # list veya array olabilir

# ---- 7) AU index bul & Terfi class SHAP ----
if feature_names is None:
    print("feature_names yok; AU index bulunamaz.")
else:
    au_idx = None
    for i, fn in enumerate(feature_names):
        if str(fn).endswith("CalSurecIyilestirmeSkoru"):
            au_idx = i
            break

    print("AU index:", au_idx, "AU name:", feature_names[au_idx] if au_idx is not None else "YOK")

    # Terfi sınıfı shap matrisi (n,f)
    if isinstance(shap_values, list):
        shap_terfi = shap_values[2]
    else:
        arr = np.array(shap_values)
        if arr.ndim == 3:         # (n,f,cls)
            shap_terfi = arr[:, :, 2]
        else:
            shap_terfi = arr

    # ---- 8) SHAP görsellerini outputs'a yazdır ----
    if feature_names is not None:
        # BAR (mean |SHAP|)
        shap.summary_plot(
            shap_terfi,
            X_test_dense,
            feature_names=feature_names,
            plot_type="bar",
            show=False
        )
        plt.tight_layout()
        plt.savefig(OUT_DIR / "shap_terfi_bar.png", dpi=200)
        plt.close()

        # BEESWARM (dağılım)
        shap.summary_plot(
            shap_terfi,
            X_test_dense,
            feature_names=feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig(OUT_DIR / "shap_terfi_beeswarm.png", dpi=200)
        plt.close()

        print(
            f"[OK] SHAP görselleri yazdırıldı: {OUT_DIR / 'shap_terfi_bar.png'} ve {OUT_DIR / 'shap_terfi_beeswarm.png'}")
    else:
        print("[SKIP] feature_names yok, SHAP görselleri üretilemedi.")

    if au_idx is not None:
        print("Terfi için AU mean(|SHAP|):", float(np.mean(np.abs(shap_terfi[:, au_idx]))))

# İstersen ilk 25 feature importance (Terfi sınıfı) da yazdır:
if feature_names is not None:
    if isinstance(shap_values, list):
        sv = shap_values[2]
    else:
        a = np.array(shap_values)
        sv = a[:, :, 2] if a.ndim == 3 else a

    mean_abs = np.mean(np.abs(sv), axis=0)
    top = np.argsort(mean_abs)[::-1][:25]
    print("\nTerfi sınıfı için Top-25 mean(|SHAP|):")
    for idx in top:
        print(f"{feature_names[idx]} -> {mean_abs[idx]:.6f}")
