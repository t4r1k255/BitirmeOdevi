import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import recall_score, make_scorer

import joblib

# --- AYARLAR ---
MODEL_YOLU = "models/xgb_boruhatti.joblib"
VERI_YOLU = "data/ModelEgitimSeti.xlsx"
SAYFA = "ModelEgitimSeti"
ETIKET = "Outcome_6Ay"

# 1) Modeli yükle
model = joblib.load(MODEL_YOLU)

# 2) Veriyi oku (x => NaN)
df = pd.read_excel(VERI_YOLU, sheet_name=SAYFA, na_values=["x", "X", " x ", "X "])

# 3) Eğitimde kullandığın filtrelerle aynı hale getir
etiket_haritasi = {"IsFeshi": 0, "MevkiDevami": 1, "Terfi": 2}
df = df[df[ETIKET].isin(etiket_haritasi.keys())].copy()

# Destek eğitim dışı (eğitimde böyle yaptıysan)
if "Departman" in df.columns:
    df = df[~df["Departman"].astype(str).str.contains("destek", case=False, na=False)].copy()

y = df[ETIKET].map(etiket_haritasi)

# PII kolonlarını çıkar (eğitimle aynı)
cikar = [c for c in ["Ad", "Soyad", "MailAdresi", "TelefonNo", "CalisanNo", "SiraNo"] if c in df.columns]
X = df.drop(columns=[ETIKET] + cikar)

# 4) Aynı split (random_state aynı olmalı)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5) PROMOTE (Terfi=2) recall scorer
def promote_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, labels=[2], average="macro")

promote_scorer = make_scorer(promote_recall)

# 6) Permutation Importance (kolon bazlı!)
sonuc = permutation_importance(
    model,
    X_test,
    y_test,
    scoring=promote_scorer,
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

onem_df = pd.DataFrame({
    "Kolon": X_test.columns,
    "PromoteRecall_Onemi_Ortalama": sonuc.importances_mean,
    "PromoteRecall_Onemi_Std": sonuc.importances_std
}).sort_values("PromoteRecall_Onemi_Ortalama", ascending=False)

print("\nPROMOTE recall'a en etkili ilk 20 kolon:")
print(onem_df.head(20).to_string(index=False))

# İstersen Excel'e de yaz
onem_df.to_excel("outputs/kolon_onemi_promote_recall.xlsx", index=False)
print("\nKaydedildi: outputs/kolon_onemi_promote_recall.xlsx")
