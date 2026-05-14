import os
import joblib
import numpy as np
import pandas as pd

# ---- AYARLAR ----
MODEL_YOLU = "models/xgb_pipeline.joblib"
GIRDI_KOLONLARI_YOLU = "models/girdi_kolonlari.joblib"

VERI_YOLU_25K = "data/CalisanListesi_25000.xlsx"
SAYFA_ADI = 0  # tek sheet ise 0; adıyla kullanmak istersen "Çalışan Listesi" gibi yaz
ETIKET_KOLONU = "Outcome_6Ay"

CIKTI_KLASOR = "outputs"
CIKTI_EXCEL = os.path.join(CIKTI_KLASOR, "gelisim_ihtiyaci_raporu.xlsx")

# Bu metrikler “gelişim ihtiyacı” skorunu etkiler.
# Not: CalAnketKatilim feature değil; burada da skora katmıyoruz.
METRIKLER = {
    # yüksek oldukça kötü
    "CalHataOrani": {"w": 2.0, "dir": "high_bad", "neden": "Hata oranı yüksek", "egitim": "Hata Azaltma ve Kalite"},
    "CalReworkSayisi": {"w": 1.2, "dir": "high_bad", "neden": "Rework sayısı yüksek", "egitim": "Rework Kök Neden Analizi"},
    "CalOrtCozumSuresi": {"w": 1.4, "dir": "high_bad", "neden": "Çözüm süresi uzun", "egitim": "Zaman Yönetimi ve Verimlilik"},
    "CalAuditBulgusuSayisi": {"w": 1.1, "dir": "high_bad", "neden": "Audit bulgusu fazla", "egitim": "Denetim Hazırlığı ve Kontroller"},
    "CalTekrarlayanBulgu": {"w": 1.0, "dir": "high_bad", "neden": "Tekrarlayan bulgu var", "egitim": "Sürdürülebilir Düzeltici Faaliyetler"},
    "CalKYCAML_GeriDonusOrani": {"w": 1.0, "dir": "high_bad", "neden": "KYC/AML geri dönüş oranı yüksek", "egitim": "KYC/AML Uygulama Pratikleri"},
    "CalEgitimGecikmeGunu": {"w": 0.9, "dir": "high_bad", "neden": "Eğitim gecikmesi yüksek", "egitim": "Eğitim Disiplini ve Takip"},
    # düşük oldukça kötü
    "CalSLAUyumOrani": {"w": 1.6, "dir": "low_bad", "neden": "SLA uyumu düşük", "egitim": "SLA ve Operasyon Yönetimi"},
    "CalZorunluEgitimTamamlamaOrani": {"w": 1.3, "dir": "low_bad", "neden": "Zorunlu eğitim tamamlama düşük", "egitim": "Zorunlu Eğitim Tamamlama"},
    "CalSertifikaPuani": {"w": 1.0, "dir": "low_bad", "neden": "Sertifika puanı düşük", "egitim": "Sertifikasyon / Teknik Yetkinlik"},
    # Anket skoru çok eksik geldiği için yalnızca varsa değerlendiriyoruz (yoksa 0 katkı)
    "CalIsTatmini_M1": {"w": 0.6, "dir": "low_bad", "neden": "İş tatmini düşük", "egitim": "Motivasyon ve Bağlılık Planı"},
}

KIMLIK_KOLONLARI = ["Departman", "DepNo", "CalisanNo", "Ad", "Soyad", "MeslekiRolu", "YasBand"]


def _is_destek(dept_series: pd.Series) -> pd.Series:
    return dept_series.astype(str).str.contains("destek", case=False, na=False)


def _ensure_model_input(df: pd.DataFrame, girdi_kolonlari: list[str]) -> pd.DataFrame:
    X = df.copy()
    missing = [c for c in girdi_kolonlari if c not in X.columns]
    for c in missing:
        X[c] = np.nan

    extra = [c for c in X.columns if c not in girdi_kolonlari]
    if extra:
        X = X.drop(columns=extra, errors="ignore")

    return X[girdi_kolonlari]


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _zscore_group(df: pd.DataFrame, col: str, group_cols: list[str]) -> pd.Series:
    x = df[col]
    # global fallback
    g_mu = float(np.nanmean(x)) if np.isfinite(np.nanmean(x)) else 0.0
    g_sd = float(np.nanstd(x)) if np.isfinite(np.nanstd(x)) else 1.0
    if g_sd == 0:
        g_sd = 1.0

    grp = df.groupby(group_cols, dropna=False)[col]
    mu = grp.transform("mean").fillna(g_mu)
    sd = grp.transform("std").fillna(g_sd).replace(0, g_sd)

    z = (x - mu) / sd
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return z.clip(-3.0, 3.0)


def main():
    os.makedirs(CIKTI_KLASOR, exist_ok=True)

    print("1) Model + kolon standardı yükleniyor...")
    pipe = joblib.load(MODEL_YOLU)
    girdi_kolonlari = joblib.load(GIRDI_KOLONLARI_YOLU)

    print("2) 25k veri okunuyor...")
    df = pd.read_excel(VERI_YOLU_25K, sheet_name=SAYFA_ADI, na_values=["x", "X", " x ", "X "])

    # Etiket kolonu varsa drop
    if ETIKET_KOLONU in df.columns:
        df = df.drop(columns=[ETIKET_KOLONU], errors="ignore")

    # Destek tamamen hariç
    if "Departman" in df.columns:
        df = df[~_is_destek(df["Departman"])].copy()

    # Kimlik kolonları (rapora basacağız)
    kimlik_cols = [c for c in KIMLIK_KOLONLARI if c in df.columns]
    id_df = df[kimlik_cols].copy() if kimlik_cols else pd.DataFrame(index=df.index)

    print("3) Model ihtimalleri hesaplanıyor...")
    X_model = _ensure_model_input(df, girdi_kolonlari)
    proba = pipe.predict_proba(X_model)  # 0=IsFeshi, 1=MevkiDevami, 2=Terfi
    pred = np.argmax(proba, axis=1)

    out = id_df.copy()
    out["İhtimal_IsFeshi"] = proba[:, 0]
    out["İhtimal_MevkiDevami"] = proba[:, 1]
    out["İhtimal_Terfi"] = proba[:, 2]
    out["Tahmin"] = pd.Series(pred).map({0: "IsFeshi", 1: "MevkiDevami", 2: "Terfi"}).values

    print("4) Gelişim metrikleri hazırlanıyor...")
    work = df.copy()

    # numeric dönüşümler (varsa)
    for col in METRIKLER.keys():
        if col in work.columns:
            work[col] = _to_num(work[col])

    # ReworkOrani türet (varsa)
    if "CalReworkSayisi" in work.columns and "Toplam iş adedi (6 ay)" in work.columns:
        work["ToplamIs6Ay"] = _to_num(work["Toplam iş adedi (6 ay)"])
        denom = work["ToplamIs6Ay"].replace(0, np.nan)
        work["ReworkOrani"] = (work["CalReworkSayisi"] / denom).clip(0, 1).replace([np.inf, -np.inf], np.nan)
    else:
        work["ReworkOrani"] = np.nan

    # Rework için “sayı” yerine “oran”ı skora sokmak daha okunur → burada küçük dokunuş:
    # Eğer oran hesaplandıysa, CalReworkSayisi skor katkısını oran üstünden yapacağız.
    rework_use_rate = "ReworkOrani" in work.columns and work["ReworkOrani"].notna().any()

    # Skor hesap parametreleri
    group_cols = [c for c in ["Departman", "MeslekiRolu"] if c in work.columns]
    if not group_cols:
        group_cols = ["Departman"] if "Departman" in work.columns else []

    # katkılar
    contrib_cols = []
    for col, cfg in METRIKLER.items():
        if col not in work.columns:
            continue

        base = work[col]

        # Anket skoru: çok eksik olabilir → sadece varsa değerlendir (yoksa 0 katkı)
        if col == "CalIsTatmini_M1":
            mask = base.notna()
        else:
            mask = pd.Series(True, index=work.index)

        # Rework: oran varsa onu kullan
        if col == "CalReworkSayisi" and rework_use_rate:
            base = work["ReworkOrani"]
            tmp_col = "__ReworkOrani__"
            work[tmp_col] = base
            z = _zscore_group(work.loc[mask], tmp_col, group_cols) if group_cols else (base.loc[mask] - base.loc[mask].mean()) / (base.loc[mask].std() or 1.0)
            z = z.reindex(work.index).fillna(0.0)
        else:
            z = _zscore_group(work.loc[mask], col, group_cols) if group_cols else (base.loc[mask] - base.loc[mask].mean()) / (base.loc[mask].std() or 1.0)
            z = z.reindex(work.index).fillna(0.0)

        # low_bad ise ters çevir (düşük → kötü)
        if cfg["dir"] == "low_bad":
            z = -z

        # sadece kötü tarafa penalize edelim (negatifler 0)
        comp = (cfg["w"] * z).clip(lower=0.0)

        ccol = f"katki__{col}"
        work[ccol] = comp
        contrib_cols.append(ccol)

    if not contrib_cols:
        raise RuntimeError("Skor hesaplanacak metrik bulunamadı. Kolon adlarını kontrol et.")

    raw_need = work[contrib_cols].sum(axis=1)
    # 0–100 normalize
    mx = float(raw_need.max()) if float(raw_need.max()) > 0 else 1.0
    out["GelisimSkoru"] = (100.0 * raw_need / mx).round(2)

    print("5) Nedenler + eğitim önerileri yazılıyor...")
    # neden & eğitim: en yüksek 3 katkı
    neden_list = []
    egitim_list = []

    # kolon->cfg map (rework oranı seçildiyse yazıyı “Rework oranı” diye düzelt)
    def _col_label(col_name: str) -> tuple[str, str]:
        if col_name == "CalReworkSayisi" and rework_use_rate:
            return "Rework oranı yüksek", "Rework Kök Neden Analizi"
        cfg = METRIKLER.get(col_name)
        if not cfg:
            return col_name, ""
        return cfg["neden"], cfg["egitim"]

    for idx in work.index:
        row = work.loc[idx, contrib_cols]
        top3 = row.sort_values(ascending=False).head(3)
        reasons = []
        trainings = []
        for ccol in top3.index:
            if float(top3[ccol]) <= 0:
                continue
            orig = ccol.replace("katki__", "")
            n, e = _col_label(orig)
            reasons.append(n)
            if e:
                trainings.append(e)
        # uniq
        trainings = list(dict.fromkeys(trainings))
        neden_list.append("; ".join(reasons) if reasons else "Belirgin kritik eksik yok")
        egitim_list.append("; ".join(trainings) if trainings else "Genel gelişim planı")

    out["OncelikliNedenler"] = neden_list
    out["OnerilenEgitimler"] = egitim_list

    print("6) Departman içi gelişim sıralaması...")
    if "Departman" in out.columns:
        out["DeptIcinde_GelisimRank"] = (
            out.groupby("Departman")["GelisimSkoru"]
               .rank(method="first", ascending=False)
               .astype(int)
        )
    else:
        out["DeptIcinde_GelisimRank"] = np.nan

    out["Global_GelisimRank"] = out["GelisimSkoru"].rank(method="first", ascending=False).astype(int)

    # raporu daha anlaşılır yapmak için bazı ham metrikleri de ekleyelim (varsa)
    ham_ekle = [
        "CalHataOrani",
        "ReworkOrani",
        "CalSLAUyumOrani",
        "CalOrtCozumSuresi",
        "CalEgitimGecikmeGunu",
        "CalZorunluEgitimTamamlamaOrani",
        "CalSertifikaPuani",
        "CalAuditBulgusuSayisi",
        "CalTekrarlayanBulgu",
        "CalKYCAML_GeriDonusOrani",
        "CalIsTatmini_M1",
        "CalAnketKatilim",
    ]
    for c in ham_ekle:
        if c in work.columns and c not in out.columns:
            out[c] = work[c]

    # sıralama
    sort_cols = ["Departman", "DeptIcinde_GelisimRank"] if "Departman" in out.columns else ["Global_GelisimRank"]
    out = out.sort_values(sort_cols, ascending=[True, True] if len(sort_cols) == 2 else [True])

    print("7) Excel yazılıyor:", CIKTI_EXCEL)
    with pd.ExcelWriter(CIKTI_EXCEL, engine="openpyxl") as writer:
        out.to_excel(writer, index=False, sheet_name="GelisimeIhtiyac")

        if "Departman" in out.columns:
            top50 = out[out["DeptIcinde_GelisimRank"] <= 50].copy()
            top50.to_excel(writer, index=False, sheet_name="Top50_HerDepartman")

    # Basit format: yüzde kolonları 0.00% görünsün
    try:
        from openpyxl import load_workbook

        wb = load_workbook(CIKTI_EXCEL)
        for sh in ["GelisimeIhtiyac", "Top50_HerDepartman"]:
            if sh not in wb.sheetnames:
                continue
            ws = wb[sh]
            headers = [cell.value for cell in ws[1]]

            percent_like = {
                "İhtimal_IsFeshi", "İhtimal_MevkiDevami", "İhtimal_Terfi",
                "CalHataOrani", "ReworkOrani", "CalSLAUyumOrani", "CalZorunluEgitimTamamlamaOrani",
                "CalKYCAML_GeriDonusOrani",
            }

            for col_idx, h in enumerate(headers, start=1):
                if h in percent_like:
                    for r in range(2, ws.max_row + 1):
                        ws.cell(r, col_idx).number_format = "0.00%"
                elif h in {"GelisimSkoru"}:
                    for r in range(2, ws.max_row + 1):
                        ws.cell(r, col_idx).number_format = "0.00"
                elif "Rank" in str(h):
                    for r in range(2, ws.max_row + 1):
                        ws.cell(r, col_idx).number_format = "0"

        wb.save(CIKTI_EXCEL)
    except Exception as e:
        print("Format atlandı (kritik değil):", e)

    print("Bitti ✅ ->", CIKTI_EXCEL)


if __name__ == "__main__":
    main()
