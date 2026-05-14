import os
import numpy as np
import pandas as pd
import joblib

os.makedirs("outputs", exist_ok=True)

model_yolu = "models/xgb_boruhatti.joblib"
ozellik_kolonlari_yolu = "models/ozellik_kolonlari.joblib"
etiket_haritasi_yolu = "models/etiket_haritasi.joblib"

veri_yolu = "data/AsilModelVeriseti.xlsx"
sayfa_adi = "Calisan Listesi"

model_boruhatti = joblib.load(model_yolu)
ozellik_kolonlari = joblib.load(ozellik_kolonlari_yolu)
etiket_haritasi = joblib.load(etiket_haritasi_yolu)
ters_etiket = {v: k for k, v in etiket_haritasi.items()}

tahmin_df = pd.read_excel(veri_yolu, sheet_name=sayfa_adi, na_values=["x", "X", " x ", "X "])

# kimlik alanlari varsa cikar
cikarilacak_kolonlar = [c for c in ["Ad", "Soyad", "MailAdresi", "TelefonNo", "CalisanNo", "SiraNo"] if c in tahmin_df.columns]
X = tahmin_df.drop(columns=cikarilacak_kolonlar, errors="ignore").copy()

# egitimdeki kolon semasini zorla
for kolon in ozellik_kolonlari:
    if kolon not in X.columns:
        X[kolon] = np.nan
X = X[ozellik_kolonlari]

olasiliklar = model_boruhatti.predict_proba(X)
sinif_tahmin = np.argmax(olasiliklar, axis=1)

tahmin_df["Tahmin_Outcome"] = [ters_etiket[i] for i in sinif_tahmin]
tahmin_df["P_IsFeshi"] = olasiliklar[:, 0]
tahmin_df["P_MevkiDevami"] = olasiliklar[:, 1]
tahmin_df["P_Terfi"] = olasiliklar[:, 2]

# Destek icin x
destek_mi = pd.Series(False, index=tahmin_df.index)
if "Departman" in tahmin_df.columns:
    destek_mi = tahmin_df["Departman"].astype(str).str.contains("destek", case=False, na=False)

for kolon in ["Tahmin_Outcome", "P_IsFeshi", "P_MevkiDevami", "P_Terfi"]:
    tahmin_df.loc[destek_mi, kolon] = "x"

# Departman + MeslekiRolu icinde P_Terfi ile siralama
tahmin_df["DepRol_Sira"] = np.nan
tahmin_df["DepRol_Yuzdelik"] = np.nan

if ("Departman" in tahmin_df.columns) and ("MeslekiRolu" in tahmin_df.columns):
    maske = ~destek_mi & (tahmin_df["P_Terfi"] != "x")
    gecici = tahmin_df.loc[maske].copy()

    gecici["DepRol_Sira"] = gecici.groupby(["Departman", "MeslekiRolu"])["P_Terfi"] \
                                  .rank(method="dense", ascending=False)
    gecici["DepRol_Yuzdelik"] = gecici.groupby(["Departman", "MeslekiRolu"])["P_Terfi"] \
                                      .rank(pct=True, ascending=True) * 100

    tahmin_df.loc[maske, "DepRol_Sira"] = gecici["DepRol_Sira"].values
    tahmin_df.loc[maske, "DepRol_Yuzdelik"] = gecici["DepRol_Yuzdelik"].values

tahmin_df.loc[destek_mi, "DepRol_Sira"] = "x"
tahmin_df.loc[destek_mi, "DepRol_Yuzdelik"] = "x"

cikis_yolu = "outputs/AsilModelVeriseti_skorlanmis.xlsx"
tahmin_df.to_excel(cikis_yolu, index=False)
print("Kaydedildi:", cikis_yolu)
