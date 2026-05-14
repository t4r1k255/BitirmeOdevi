import os
import joblib
import numpy as np
import pandas as pd

MODEL_YOLU = "models/xgb_pipeline.joblib"
GIRDI_KOLONLARI_YOLU = "models/girdi_kolonlari.joblib"

VERI_YOLU_25K = "data/CalisanListesi_25000.xlsx"
SAYFA_ADI = 0
ETIKET_KOLONU = "Outcome_6Ay"

CIKTI_KLASOR = "outputs"
CIKTI_EXCEL = os.path.join(CIKTI_KLASOR, "departman_ici_siralama.xlsx")


def main():
    os.makedirs(CIKTI_KLASOR, exist_ok=True)

    print("1) Model yükleniyor...")
    pipe = joblib.load(MODEL_YOLU)

    print("2) Girdi kolon standardı yükleniyor...")
    girdi_kolonlari = joblib.load(GIRDI_KOLONLARI_YOLU)

    print("3) 25k veri okunuyor...")
    df = pd.read_excel(
        VERI_YOLU_25K,
        sheet_name=SAYFA_ADI,
        na_values=["x", "X", " x ", "X "]
    )

    # --- DESTEK TAMAMEN DIŞARI ---
    if "Departman" in df.columns:
        dept_str = df["Departman"].astype(str).str.strip()
        mask_destek = dept_str.str.contains("destek", case=False, na=False)
        removed = int(mask_destek.sum())
        df = df.loc[~mask_destek].copy()
        print(f"   Destek filtrelendi. Çıkarılan satır: {removed}")
    else:
        print("   UYARI: 'Departman' kolonu yok, Destek filtrelenemedi.")

    # Etiket kolonu varsa drop
    if ETIKET_KOLONU in df.columns:
        df = df.drop(columns=[ETIKET_KOLONU], errors="ignore")

    # Raporda görmek istediğin kolonlar
    rapor_kolonlari = ["Departman", "DepNo", "CalisanNo", "Ad", "Soyad", "MeslekiRolu"]
    rapor_kolonlari = [c for c in rapor_kolonlari if c in df.columns]

    # Model girdisini standartlaştır (eksik ekle, fazla at, sırala)
    X = df.copy()

    missing = [c for c in girdi_kolonlari if c not in X.columns]
    for c in missing:
        X[c] = np.nan

    extra = [c for c in X.columns if c not in girdi_kolonlari]
    if extra:
        X = X.drop(columns=extra, errors="ignore")

    X = X[girdi_kolonlari]

    print("4) İhtimaller hesaplanıyor (predict_proba)...")
    proba = pipe.predict_proba(X)  # 0=IsFeshi, 1=MevkiDevami, 2=Terfi

    out = df[rapor_kolonlari].copy()

    # İsimlendirme: proba yerine "ihtimal"
    out["ihtimal_isfesi"] = np.round(proba[:, 0], 2)
    out["ihtimal_mevkidevami"] = np.round(proba[:, 1], 2)
    out["ihtimal_terfi"] = np.round(proba[:, 2], 2)

    # Tahmin etiketi
    pred = np.argmax(proba, axis=1)
    out["tahmin_sinif"] = pred
    out["tahmin_etiket"] = out["tahmin_sinif"].map({0: "IsFeshi", 1: "MevkiDevami", 2: "Terfi"})

    print("5) Departman içi sıralama oluşturuluyor...")
    if "Departman" in out.columns:
        out["departman_ici_sira"] = (
            out.groupby("Departman")["ihtimal_terfi"]
               .rank(method="first", ascending=False)
               .astype(int)
        )
        out["global_sira"] = out["ihtimal_terfi"].rank(method="first", ascending=False).astype(int)
        out = out.sort_values(["Departman", "departman_ici_sira"], ascending=[True, True])
    else:
        out["departman_ici_sira"] = np.nan
        out["global_sira"] = out["ihtimal_terfi"].rank(method="first", ascending=False).astype(int)
        out = out.sort_values(["global_sira"], ascending=[True])

    print("6) Excel yazılıyor:", CIKTI_EXCEL)
    with pd.ExcelWriter(CIKTI_EXCEL, engine="openpyxl") as writer:
        out.to_excel(writer, index=False, sheet_name="DepartmanIciSiralama")

        if "Departman" in out.columns:
            top50 = out[out["departman_ici_sira"] <= 50].copy()
            top50.to_excel(writer, index=False, sheet_name="Top50_HerDepartman")

    print("Bitti ✅ -> outputs/departman_ici_siralama.xlsx")


if __name__ == "__main__":
    main()
