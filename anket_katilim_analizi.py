import os
import re
import numpy as np
import pandas as pd

# =========================
# AYARLAR
# =========================
VERI_YOLU = "data/CalisanListesi_25000.xlsx"   # 25k dosyan
SAYFA_ADI = 0                                  # tek sayfaysa 0; isimse "Çalışan Listesi"
CIKTI_KLASOR = "outputs"
CIKTI_DOSYA = os.path.join(CIKTI_KLASOR, "anket_katilim_analizi.xlsx")

# Ankete katılım kolonu
KATILIM_KOLONU = "CalAnketKatilim"

# “Destek” analize girsin mi?
DESTEK_DAHIL_ET = False

# Rapor kimlik kolonları
ID_KOLONLARI = ["CalisanNo", "Ad", "Soyad", "Departman", "MeslekiRolu", "DepNo"]

# Kullanacağımız metrik kolonları (varsa kullanır, yoksa pas geçer)
METRIKLER = {
    # Pozitif (düşük olması risk)
    "CalUstPuani": "pozitif",
    "CalAstPuani": "pozitif",
    "CalDenkPuani": "pozitif",
    "CalSLAUyumOrani": "pozitif",
    "CalZorunluEgitimTamamlamaOrani": "pozitif",
    "CalSertifikaPuani": "pozitif",
    "CalOtomasyonKatkisi": "pozitif",
    "CalSurecIyilestirmeSkoru": "pozitif",

    # Negatif (yüksek olması risk)
    "CalHataOrani": "negatif",
    "CalOrtCozumSuresi": "negatif",
    "CalAuditBulgusuSayisi": "negatif",
    "CalTekrarlayanBulgu": "negatif",
    "CalKYCAML_GeriDonusOrani": "negatif",
    "CalEgitimGecikmeGunu": "negatif",
    "CalGunlukIsSaat": "negatif",

    # Proje/iş hacmi (yüksekse baskı göstergesi gibi ele alacağız)
    "Toplam iş adedi (6 ay)": "hacim",
    "Toplam Proje": "hacim",
    "CalReworkSayisi": "negatif",  # oranını da türeteceğiz

    # Proje sonuçları (oran türetilecek)
    "Tamamladı": "ham",
    "Geç Tamamladı": "ham",
    "Tamamlayamadı": "ham",

    # Kariyer göstergeleri
    "CalSonTerfiSure": "negatif",  # uzun süre -> durağanlık sinyali
    "CalTerfiSay": "pozitif",
}

# Risk etiketleri + ağırlıkları
RISK_ETIKETLERI = [
    ("Yönetici ilişki/algı sorunu (üst puanı düşük)", 14, "low_manager"),
    ("Ekip/akran uyumu düşük (denk/ast puanı düşük)", 12, "low_peer"),
    ("İş yükü / zaman baskısı yüksek", 10, "overload"),
    ("Kalite baskısı (hata/rework yüksek)", 12, "quality_stress"),
    ("SLA / teslim performansı zayıf", 10, "sla_risk"),
    ("Eğitim borcu / gecikmesi var", 10, "training_debt"),
    ("Uyum/AML geri dönüşleri yüksek (sürtünme)", 8, "compliance_friction"),
    ("Denetim bulgusu baskısı", 8, "audit_pressure"),
    ("Kariyer durağanlığı (son terfi süresi yüksek)", 8, "career_stagnation"),
    ("Proje başarısızlık/gecikme oranı yüksek", 10, "project_outcome_risk"),
]

# =========================
# YARDIMCILAR
# =========================
def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    b2 = b.replace(0, np.nan)
    out = a / b2
    return out.fillna(0.0)

def q25(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.quantile(0.25)) if len(s) else np.nan

def q75(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.quantile(0.75)) if len(s) else np.nan

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def to_scalar(x):
    # Series/array gelirse ilk anlamlı değere indir
    if isinstance(x, pd.Series):
        if x.empty:
            return np.nan
        x2 = x.dropna()
        return x2.iloc[0] if len(x2) else np.nan
    if isinstance(x, np.ndarray):
        return x[0] if x.size else np.nan
    return x

# =========================
# ANA
# =========================
def main():
    os.makedirs(CIKTI_KLASOR, exist_ok=True)

    print("1) Veri okunuyor...")
    df = pd.read_excel(VERI_YOLU, sheet_name=SAYFA_ADI, na_values=["x", "X", " x ", "X "])

    # kolon adlarını temizle (gizli boşluk / çift boşluk vs.)
    df.columns = [str(c).strip() for c in df.columns]

    if df.columns.duplicated().any():
        dup = df.columns[df.columns.duplicated()].tolist()
        print("Uyarı: duplicate kolon(lar) bulundu:", dup)
        df = df.loc[:, ~df.columns.duplicated()].copy()

    # Excel bazen aynı başlığı 2 kez üretebiliyor -> duplicate kolonları düşür
    if df.columns.duplicated().any():
        dup = df.columns[df.columns.duplicated()].tolist()
        print("Uyarı: Mükerrer kolon başlıkları bulundu, ilkini tuttum:", dup)
        df = df.loc[:, ~df.columns.duplicated()].copy()

    # Katılım kolonu yoksa rapor anlamsız
    if KATILIM_KOLONU not in df.columns:
        raise ValueError(f"{KATILIM_KOLONU} kolonu bulunamadı. Dosyanda kolon adı farklıysa AYARLAR bölümünden değiştir.")

    # Destek filtresi (istenirse)
    if not DESTEK_DAHIL_ET and "Departman" in df.columns:
        m = df["Departman"].astype("string").fillna("").str.contains("destek", case=False, na=False)
        df = df[~m].copy()

    # Katılımı numeric yap
    df[KATILIM_KOLONU] = to_num(df[KATILIM_KOLONU]).fillna(0).astype(int)

    # Kimlik kolonları
    id_cols = [c for c in ID_KOLONLARI if c in df.columns]

    # Kullanılabilir metrik kolonları
    met_cols = [c for c in METRIKLER.keys() if c in df.columns]

    # Numeric coercion (metrikler için)
    for c in met_cols:
        if c not in ["Departman", "MeslekiRolu", "Ad", "Soyad"]:
            df[c] = to_num(df[c])

    # ---- Türetilen oranlar (varsa) ----
    if "Toplam Proje" in df.columns:
        top_proje = df["Toplam Proje"].fillna(0)
        df["TamamlamaOrani"] = safe_div(df.get("Tamamladı", 0).fillna(0), top_proje.clip(lower=1))
        df["GecOrani"] = safe_div(df.get("Geç Tamamladı", 0).fillna(0), top_proje.clip(lower=1))
        df["BasarisizlikOrani"] = safe_div(df.get("Tamamlayamadı", 0).fillna(0), top_proje.clip(lower=1))
        df["ProjeBasariSkoru"] = (1.0 * df["TamamlamaOrani"] - 0.6 * df["GecOrani"] - 1.2 * df["BasarisizlikOrani"]).fillna(0.0)

    if "CalReworkSayisi" in df.columns and "Toplam iş adedi (6 ay)" in df.columns:
        tot_job = df["Toplam iş adedi (6 ay)"].fillna(0).clip(lower=1)
        df["ReworkOrani"] = safe_div(df["CalReworkSayisi"].fillna(0), tot_job)

    derived_cols = [c for c in ["TamamlamaOrani", "GecOrani", "BasarisizlikOrani", "ProjeBasariSkoru", "ReworkOrani"] if c in df.columns]

    # ---- Katılan / katılmayan ayır ----
    katilan = df[df[KATILIM_KOLONU] == 1].copy()
    katilmayan = df[df[KATILIM_KOLONU] == 0].copy()

    print(f"2) Katılan: {len(katilan)} | Katılmayan: {len(katilmayan)}")

    # ---- Eşikler: role bazlı (varsa), yoksa departman bazlı, yoksa global ----
    # Öncelik: MeslekiRolu
    group_keys = []
    if "MeslekiRolu" in df.columns:
        group_keys = ["MeslekiRolu"]
    elif "Departman" in df.columns:
        group_keys = ["Departman"]

    # Sinyal hesaplayacağımız metrik listesi (türetilenlerle birlikte)
    signal_metrics = [
        "CalUstPuani", "CalAstPuani", "CalDenkPuani",
        "CalSLAUyumOrani", "CalHataOrani", "ReworkOrani",
        "CalOrtCozumSuresi", "CalZorunluEgitimTamamlamaOrani", "CalEgitimGecikmeGunu",
        "CalSertifikaPuani", "CalKYCAML_GeriDonusOrani",
        "CalAuditBulgusuSayisi", "CalTekrarlayanBulgu",
        "CalGunlukIsSaat", "Toplam iş adedi (6 ay)",
        "BasarisizlikOrani", "GecOrani",
        "CalSonTerfiSure"
    ]
    signal_metrics = [c for c in signal_metrics if c in df.columns]  # sadece mevcutlar

    # Eşik tablosu
    print("3) Eşikler hazırlanıyor...")
    if group_keys:
        grp = df.groupby(group_keys, dropna=False)
        thresholds = {}
        for m in signal_metrics:
            q25_map = grp[m].apply(q25)
            q75_map = grp[m].apply(q75)
            thresholds[m] = {"q25": q25_map, "q75": q75_map}
    else:
        thresholds = {}
        for m in signal_metrics:
            thresholds[m] = {"q25": q25(df[m]), "q75": q75(df[m])}

    def get_thr(row, metric, which):
        if not group_keys:
            return thresholds[metric][which]

        key = tuple(row[k] for k in group_keys)
        ser = thresholds[metric][which]

        try:
            val = ser.loc[key]
        except Exception:
            try:
                val = ser.loc[key[0]]
            except Exception:
                return np.nan

        # val bazen Series dönebilir -> scalar yap
        if isinstance(val, pd.Series):
            val = val.iloc[0] if len(val) else np.nan
        return float(val) if pd.notna(val) else np.nan

    # ---- Katılmayanlar için risk sinyalleri ----
    print("4) Katılmayanlar için sinyal/etiket üretiliyor...")
    out_cols = id_cols + [KATILIM_KOLONU] + signal_metrics + derived_cols
    out_cols = [c for c in out_cols if c in katilmayan.columns]

    # >>> KRİTİK: duplicate kolon isimlerini kaldır (sıra korunur)
    out_cols = list(dict.fromkeys(out_cols))

    out = katilmayan[out_cols].copy()

    # Flag alanları
    for _, _, code in RISK_ETIKETLERI:
        out[code] = 0

    # Her çalışan için sinyal
    for idx, row in out.iterrows():
        tags = []

        # 1) Yönetici/üst puanı düşük
        if "CalUstPuani" in out.columns:
            v = row.get("CalUstPuani", np.nan)
            thr = get_thr(row, "CalUstPuani", "q25")
            if pd.notna(v) and pd.notna(thr) and v <= thr:
                out.at[idx, "low_manager"] = 1

        # 2) Akran/ast düşük
        peer_flag = False
        for m in ["CalDenkPuani", "CalAstPuani"]:
            if m in out.columns:
                v = row.get(m, np.nan)
                thr = get_thr(row, m, "q25")
                if pd.notna(v) and pd.notna(thr) and v <= thr:
                    peer_flag = True
        if peer_flag:
            out.at[idx, "low_peer"] = 1

        # 3) İş yükü / baskı
        overload_flag = False
        for m in ["CalGunlukIsSaat", "Toplam iş adedi (6 ay)", "Toplam Proje"]:
            if m in out.columns:
                v = row.get(m, np.nan)
                thr = get_thr(row, m, "q75")
                if pd.notna(v) and pd.notna(thr) and v >= thr:
                    overload_flag = True
        if overload_flag:
            out.at[idx, "overload"] = 1

        # 4) Kalite baskısı (hata + rework)
        quality_flag = False
        if "CalHataOrani" in out.columns:
            v = row.get("CalHataOrani", np.nan)
            thr = get_thr(row, "CalHataOrani", "q75")
            if pd.notna(v) and pd.notna(thr) and v >= thr:
                quality_flag = True
        if "ReworkOrani" in out.columns:
            v = row.get("ReworkOrani", np.nan)
            thr = get_thr(row, "ReworkOrani", "q75")
            if pd.notna(v) and pd.notna(thr) and v >= thr:
                quality_flag = True
        if quality_flag:
            out.at[idx, "quality_stress"] = 1

        # 5) SLA zayıf
        if "CalSLAUyumOrani" in out.columns:
            v = row.get("CalSLAUyumOrani", np.nan)
            thr = get_thr(row, "CalSLAUyumOrani", "q25")
            if pd.notna(v) and pd.notna(thr) and v <= thr:
                out.at[idx, "sla_risk"] = 1

        # 6) Eğitim borcu
        training_flag = False
        if "CalZorunluEgitimTamamlamaOrani" in out.columns:
            v = row.get("CalZorunluEgitimTamamlamaOrani", np.nan)
            thr = get_thr(row, "CalZorunluEgitimTamamlamaOrani", "q25")
            if pd.notna(v) and pd.notna(thr) and v <= thr:
                training_flag = True
        if "CalEgitimGecikmeGunu" in out.columns:
            v = row.get("CalEgitimGecikmeGunu", np.nan)
            thr = get_thr(row, "CalEgitimGecikmeGunu", "q75")
            if pd.notna(v) and pd.notna(thr) and v >= thr:
                training_flag = True
        if training_flag:
            out.at[idx, "training_debt"] = 1

        # 7) Uyum sürtünmesi
        if "CalKYCAML_GeriDonusOrani" in out.columns:
            v = row.get("CalKYCAML_GeriDonusOrani", np.nan)
            thr = get_thr(row, "CalKYCAML_GeriDonusOrani", "q75")
            if pd.notna(v) and pd.notna(thr) and v >= thr:
                out.at[idx, "compliance_friction"] = 1

        # 8) Denetim baskısı
        audit_flag = False
        for m in ["CalAuditBulgusuSayisi", "CalTekrarlayanBulgu"]:
            if m in out.columns:
                v = row.get(m, np.nan)
                thr = get_thr(row, m, "q75")
                if pd.notna(v) and pd.notna(thr) and v >= thr:
                    audit_flag = True
        if audit_flag:
            out.at[idx, "audit_pressure"] = 1

        # 9) Kariyer durağanlığı
        if "CalSonTerfiSure" in out.columns:
            v = row.get("CalSonTerfiSure", np.nan)
            thr = get_thr(row, "CalSonTerfiSure", "q75")
            if pd.notna(v) and pd.notna(thr) and v >= thr:
                out.at[idx, "career_stagnation"] = 1

        # 10) Proje başarısızlık/gecikme
        proj_flag = False
        for m in ["BasarisizlikOrani", "GecOrani"]:
            if m in out.columns:
                v = row.get(m, np.nan)
                thr = get_thr(row, m, "q75")
                if pd.notna(v) and pd.notna(thr) and v >= thr:
                    proj_flag = True
        if proj_flag:
            out.at[idx, "project_outcome_risk"] = 1

    # ---- Skor + Etiket metni ----
    print("5) Risk skoru ve Türkçe etiketler yazılıyor...")
    # raw score
    raw = np.zeros(len(out), dtype=float)
    for label, w, code in RISK_ETIKETLERI:
        if code in out.columns:
            raw += out[code].astype(float).values * float(w)

    # 0-100 scale (cap)
    max_possible = sum(w for _, w, _ in RISK_ETIKETLERI)
    score = (raw / max_possible) * 100.0
    score = np.clip(score, 0, 100)

    out["GelisImkaniSkoru"] = np.round(score, 2)

    # Etiket metni + Top3 sinyal
    label_map = {code: label for label, _, code in RISK_ETIKETLERI}
    weight_map = {code: w for _, w, code in RISK_ETIKETLERI}

    def build_labels(r):
        present = []
        for _, _, code in RISK_ETIKETLERI:
            if code in r.index and int(r[code]) == 1:
                present.append(code)
        # ağırlığa göre sırala
        present_sorted = sorted(present, key=lambda c: weight_map.get(c, 0), reverse=True)
        labels_tr = [label_map[c] for c in present_sorted]
        top3 = labels_tr[:3]
        return (", ".join(labels_tr) if labels_tr else "Belirgin sinyal yok"), (", ".join(top3) if top3 else "")

    tmp = out.apply(build_labels, axis=1, result_type="expand")
    out["OlasıNedenler"] = tmp[0]
    out["Top3Sinyal"] = tmp[1]

    # Sayısal kolonları yuvarla (2 basamak)
    round_cols = []
    for c in out.columns:
        if c in ["GelisImkaniSkoru"]:
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            round_cols.append(c)
    out[round_cols] = out[round_cols].round(2)

    # ---- Departman özet ----
    print("6) Departman özet hazırlanıyor...")
    dept_summary = None
    if "Departman" in df.columns:
        # departman bazında katılmama oranı
        base = df.copy()
        base["is_non"] = (base[KATILIM_KOLONU] == 0).astype(int)
        rate = base.groupby("Departman", dropna=False)["is_non"].mean().reset_index()
        rate["AnketeKatilmamaOrani"] = (rate["is_non"] * 100).round(2)
        rate = rate.drop(columns=["is_non"])

        # katılmayanların skor ortalaması
        avg_score = out.groupby("Departman", dropna=False)["GelisImkaniSkoru"].mean().reset_index()
        avg_score["OrtalamaGelisImkaniSkoru"] = avg_score["GelisImkaniSkoru"].round(2)
        avg_score = avg_score.drop(columns=["GelisImkaniSkoru"])

        dept_summary = rate.merge(avg_score, on="Departman", how="left")

        # etiket sayımları
        for label, _, code in RISK_ETIKETLERI:
            if code in out.columns:
                cnt = out.groupby("Departman")[code].sum().reset_index()
                cnt = cnt.rename(columns={code: f"Sayim_{code}"})
                dept_summary = dept_summary.merge(cnt, on="Departman", how="left")

        dept_summary = dept_summary.fillna(0)
    # ---- DepartmanOzet kolonlarını Türkçeleştir ----
    rename_map = {
        "AnketeKatilmamaOrani": "Ankete katılmama oranı (%)",
        "OrtalamaGelisImkaniSkoru": "Ortalama gelişim ihtiyacı skoru",

        # Sinyal sayımları (adet)
        "Sayim_low_manager": "Üst değerlendirme düşük (adet)",
        "Sayim_low_peer": "Ast/akran uyumu düşük (adet)",
        "Sayim_overload": "İş yükü / zaman baskısı yüksek (adet)",
        "Sayim_quality_stress": "Kalite baskısı (hata/rework) yüksek (adet)",
        "Sayim_sla_risk": "SLA / teslim riski (adet)",
        "Sayim_training_debt": "Eğitim borcu / gecikmesi var (adet)",
        "Sayim_compliance_friction": "Uyum/AML sürtünmesi yüksek (adet)",
        "Sayim_audit_pressure": "Denetim baskısı (adet)",
        "Sayim_career_stagnation": "Kariyer durağanlığı (adet)",
        "Sayim_project_outcome_risk": "Proje gecikme/başarısızlık riski (adet)",
    }

    dept_summary = dept_summary.rename(columns={k: v for k, v in rename_map.items() if k in dept_summary.columns})

    # ---- Katılan vs Katılmayan karşılaştırma ----
    print("7) Karşılaştırma tablosu hazırlanıyor...")
    compare_metrics = [c for c in signal_metrics if c in df.columns] + derived_cols
    compare_metrics = list(dict.fromkeys(compare_metrics))  # uniq

    rows = []
    for m in compare_metrics:
        a = to_num(katilan[m]) if m in katilan.columns else pd.Series([], dtype=float)
        b = to_num(katilmayan[m]) if m in katilmayan.columns else pd.Series([], dtype=float)

        a_mean = float(a.mean(skipna=True)) if len(a) else np.nan
        b_mean = float(b.mean(skipna=True)) if len(b) else np.nan
        a_std = float(a.std(skipna=True)) if len(a) else np.nan
        # basit etki: fark / std
        effect = (b_mean - a_mean) / (a_std if a_std and a_std > 1e-9 else np.nan)

        rows.append({
            "Metrik": m,
            "Ortalama_Katilan": round(a_mean, 4) if pd.notna(a_mean) else np.nan,
            "Ortalama_Katilmayan": round(b_mean, 4) if pd.notna(b_mean) else np.nan,
            "BasitEtki_(Fark/Std)": round(effect, 4) if pd.notna(effect) else np.nan
        })

    compare_df = pd.DataFrame(rows)
    compare_df["AbsEtki"] = compare_df["BasitEtki_(Fark/Std)"].abs()
    compare_df = compare_df.sort_values("AbsEtki", ascending=False).drop(columns=["AbsEtki"])

    # Yuvarlama
    for c in ["Ortalama_Katilan", "Ortalama_Katilmayan", "Fark_(Katilmayan-Katilan)", "BasitEtki_(Fark/Std)"]:
        if c in compare_df.columns:
            compare_df[c] = pd.to_numeric(compare_df[c], errors="coerce").round(4)

    # ---- Excel yaz ----
    print("8) Excel yazılıyor:", CIKTI_DOSYA)
    with pd.ExcelWriter(CIKTI_DOSYA, engine="openpyxl") as writer:
        out.to_excel(writer, index=False, sheet_name="AnketeKatilmayanlar")
        if dept_summary is not None:
            dept_summary.to_excel(writer, index=False, sheet_name="DepartmanOzet")
        compare_df.to_excel(writer, index=False, sheet_name="Karsilastirma")

    print("Bitti ✅ ->", CIKTI_DOSYA)


if __name__ == "__main__":
    main()
