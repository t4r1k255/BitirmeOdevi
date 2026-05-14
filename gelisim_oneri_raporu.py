import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


# =========================
#  GELISIM + ONERI RAPORU (TR, OFFLINE)
#  - Skorlar 0..100
#  - Destek tamamen hariç
#  - Eğitim etiketleri + öneriler Türkçe
#  - Terfi ihtimali % olarak (0..100)
# =========================

# ---- Proje kökü (working directory ne olursa olsun doğru çalışsın) ----
BASE_DIR = Path(__file__).resolve().parent

# ---- Dosya yolları ----
MODEL_YOLU = BASE_DIR / "models" / "xgb_pipeline.joblib"
GIRDI_KOLONLARI_YOLU = BASE_DIR / "models" / "girdi_kolonlari.joblib"

VERI_YOLU_25K = BASE_DIR / "data" / "CalisanListesi_25000.xlsx"
SAYFA_ADI = 0  # tek sayfaysa 0; isimle çalışacaksan "Çalışan Listesi" gibi yaz.

CIKTI_KLASOR = BASE_DIR / "outputs"
CIKTI_EXCEL = CIKTI_KLASOR / "gelisim_oneri_raporu.xlsx"

ETIKET_KOLONU = "Outcome_6Ay"  # 25k dosyada varsa da drop edilecek


# -------------------------
# Yardımcılar
# -------------------------
def safe_cols(df: pd.DataFrame, cols):
    return [c for c in cols if c in df.columns]


def to_numeric_series(s: pd.Series) -> pd.Series:
    """
    Excel'den gelen karışık tipleri (x, %, string) güvenli numeric'e çevirir.
    - 'x' -> NaN
    - '6.7%' -> 0.067
    - 6.7 -> 0.067 varsayımı (eğer >1.5 ise yüzde olabilir)
    """
    if s is None:
        return s

    if pd.api.types.is_numeric_dtype(s):
        # yüzde gibi duran >1 değerler varsa 100'e böl (korumalı)
        out = s.copy()
        # sadece olası yüzde kolonlarında düzeltmek için: burada genel güvenli yaklaşım
        # ama şiddetli zarar vermesin diye: 0..100 aralığında beklenen kolonlarda kullanacağız
        return out

    # string'e çevir
    ss = s.astype("string").str.strip()
    ss = ss.replace({"x": pd.NA, "X": pd.NA, " x ": pd.NA, "X ": pd.NA})
    # yüzde işareti temizle
    has_pct = ss.str.contains("%", na=False)
    ss2 = ss.str.replace("%", "", regex=False)

    out = pd.to_numeric(ss2, errors="coerce")

    # yüzde işareti olanları 100'e böl
    out.loc[has_pct] = out.loc[has_pct] / 100.0

    return out


def ensure_percent_0_1(s: pd.Series) -> pd.Series:
    """
    Oran kolonlarını 0..1'e çeker:
    - 0.067 -> 0.067 (ok)
    - 6.7 -> 0.067 (yüzde gibi)
    - 67 -> 0.67 (yüzde gibi)
    """
    s = to_numeric_series(s)
    if s is None:
        return s
    out = s.copy()
    # 1.5 üstü değerler büyük ihtimal yüzde
    mask = out > 1.5
    out.loc[mask] = out.loc[mask] / 100.0
    return out


def clip_0_1(s: pd.Series) -> pd.Series:
    return s.clip(lower=0.0, upper=1.0)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline içindeki OzellikUretici'ninkine benzer şekilde
    raporlama için bazı türev kolonları üretir (varsa):
      - TamamlamaOrani, GecOrani, BasarisizlikOrani
      - ProjeBasariSkoru (0..1 ölçekli)
      - CalSirketOmru_Z, CalDepOmru_Z (global z-score)
    """
    out = df.copy()

    # Proje kolonları isimleri (sende bu şekilde)
    col_toplam = "Toplam Proje"
    col_tamam = "Tamamladı"
    col_gec = "Geç Tamamladı"
    col_tamamlayamadi = "Tamamlayamadı"

    if all(c in out.columns for c in [col_toplam, col_tamam, col_gec, col_tamamlayamadi]):
        top = to_numeric_series(out[col_toplam]).fillna(0)
        tamam = to_numeric_series(out[col_tamam]).fillna(0)
        gec = to_numeric_series(out[col_gec]).fillna(0)
        fail = to_numeric_series(out[col_tamamlayamadi]).fillna(0)

        denom = top.replace(0, np.nan)

        out["TamamlamaOrani"] = (tamam / denom).fillna(0.0)
        out["GecOrani"] = (gec / denom).fillna(0.0)
        out["BasarisizlikOrani"] = (fail / denom).fillna(0.0)

        # Basit proje başarı skoru: (tamam + 0.5*gec) / toplam  -  fail/total
        # 0..1 bandına kırp
        ps = ((tamam + 0.5 * gec) / denom).fillna(0.0) - (fail / denom).fillna(0.0)
        out["ProjeBasariSkoru"] = ps.clip(-1.0, 1.0)  # ham
        # 0..1'e taşı (rapor için okunabilir)
        out["ProjeBasariSkoru_01"] = ((out["ProjeBasariSkoru"] + 1.0) / 2.0).clip(0.0, 1.0)

    # Z-score kolonları (varsa)
    for base_col, z_col in [("CalSirketOmru", "CalSirketOmru_Z"), ("CalDepOmru", "CalDepOmru_Z")]:
        if base_col in out.columns:
            vals = to_numeric_series(out[base_col])
            mu = vals.mean(skipna=True)
            sd = vals.std(skipna=True)
            if sd and sd > 1e-9:
                out[z_col] = (vals - mu) / sd
            else:
                out[z_col] = 0.0

    return out


def dept_quantiles(df: pd.DataFrame, dept_col: str, col: str, q_low=0.25, q_med=0.5, q_high=0.75):
    """
    Departman bazlı quantile/median çıkarır.
    """
    g = df.groupby(dept_col)[col]
    ql = g.quantile(q_low)
    qm = g.quantile(q_med)
    qh = g.quantile(q_high)
    return ql, qm, qh


def normalized_gap_positive(x: pd.Series, bench: pd.Series, low: pd.Series) -> pd.Series:
    """
    Pozitif metriklerde (yüksek iyi):
    gap = max(0, bench - x) / max(bench - low, eps)
    0..1
    """
    eps = 1e-9
    denom = (bench - low).abs().replace(0, np.nan)
    out = (bench - x) / (denom + eps)
    out = out.fillna(0.0)
    return out.clip(0.0, 1.0)


def normalized_gap_negative(x: pd.Series, bench: pd.Series, high: pd.Series) -> pd.Series:
    """
    Negatif metriklerde (düşük iyi):
    gap = max(0, x - bench) / max(high - bench, eps)
    0..1
    """
    eps = 1e-9
    denom = (high - bench).abs().replace(0, np.nan)
    out = (x - bench) / (denom + eps)
    out = out.fillna(0.0)
    return out.clip(0.0, 1.0)


# -------------------------
# Türkçe eğitim etiketleri ve öneriler
# -------------------------
CATEGORY_TO_TRAINING_TR = {
    "Kalite/Hata": "Kalite ve Hata Azaltma Eğitimi",
    "SLA/Süre": "SLA Uyum ve Çözüm Süresi İyileştirme Eğitimi",
    "Eğitim Disiplini": "Zorunlu Eğitim Takibi ve Disiplin Eğitimi",
    "Sertifikasyon": "Sertifikasyon ve Yetkinlik Geliştirme Planı",
    "Süreç/Otomasyon": "Süreç İyileştirme ve Otomasyon Katkısı Eğitimi",
    "İş Tatmini": "Bağlılık ve İş Tatmini Destek Programı",
}

CATEGORY_TO_RECO_TR = {
    "Kalite/Hata": [
        "Hata oranı yüksek görünen çalışanlar için kalite checklist ve ikili kontrol (peer review) adımı eklendi.",
        "Rework sayısı yüksek görünen çalışanlar için tekrarlı iş nedenleri çıkarılıp kısa iyileştirme planı yazıldı."
    ],
    "SLA/Süre": [
        "SLA uyum oranı düşük olanlar için vaka/iş önceliklendirme düzeni belirlendi ve hedef aralıkları tanımlandı.",
        "Ortalama çözüm süresi yüksek olanlar için örnek vaka seti ile hızlandırma pratiği planlandı."
    ],
    "Eğitim Disiplini": [
        "Zorunlu eğitim tamamlama oranı düşük olanlar için eğitim takvimi ve hatırlatma akışı kurgulandı.",
        "Eğitim gecikme günü yüksek olanlar için gecikme eşiği belirlenip takip listesi çıkarıldı."
    ],
    "Sertifikasyon": [
        "Sertifika puanı düşük olanlar için departman rolüne uygun sertifika listesi ve hedef puan aralığı çıkarıldı."
    ],
    "Süreç/Otomasyon": [
        "Süreç iyileştirme skoru düşük olanlar için mini süreç analizi ve öneri şablonu ile hedefler yazıldı.",
        "Otomasyon katkısı düşük olanlar için küçük otomasyon görevleri ve mentör eşleştirme planlandı."
    ],
    "İş Tatmini": [
        "İş tatmini düşük olanlar için kısa geri bildirim görüşmesi ve takip planı (1-2 döngü) eklendi."
    ],
}

# Kategorilerin hangi kolonlarla ölçüleceği
CATEGORY_METRICS = {
    "Kalite/Hata": ["CalHataOrani", "CalReworkSayisi"],
    "SLA/Süre": ["CalSLAUyumOrani", "CalOrtCozumSuresi"],
    "Eğitim Disiplini": ["CalZorunluEgitimTamamlamaOrani", "CalEgitimGecikmeGunu"],
    "Sertifikasyon": ["CalSertifikaPuani"],
    "Süreç/Otomasyon": ["CalSurecIyilestirmeSkoru", "CalOtomasyonKatkisi"],
    "İş Tatmini": ["CalIsTatmini_M1"],
}

# Pozitif / Negatif metrik tanımı + ağırlık
# (Kolon yoksa otomatik atlanır.)
POS_METRICS = {
    "CalSLAUyumOrani": 1.0,                 # yüksek iyi (0..1)
    "CalZorunluEgitimTamamlamaOrani": 0.9,  # yüksek iyi (0..1)
    "CalSertifikaPuani": 0.7,               # yüksek iyi
    "CalSurecIyilestirmeSkoru": 0.6,        # yüksek iyi
    "CalOtomasyonKatkisi": 0.6,             # yüksek iyi
    "CalIsTatmini_M1": 0.5,                 # yüksek iyi (genelde 1..5)
    "ProjeBasariSkoru_01": 0.6,             # yüksek iyi (0..1) (türetildi)
    "TamamlamaOrani": 0.5,                  # yüksek iyi (0..1) (türetildi)
}

NEG_METRICS = {
    "CalHataOrani": 1.0,           # düşük iyi (oran)
    "CalReworkSayisi": 0.7,        # düşük iyi (adet)
    "CalOrtCozumSuresi": 0.7,      # düşük iyi
    "CalEgitimGecikmeGunu": 0.6,   # düşük iyi
    "BasarisizlikOrani": 0.6,      # düşük iyi (0..1) (türetildi)
    "GecOrani": 0.4,               # düşük iyi (0..1) (türetildi)
}


def main():
    os.makedirs(CIKTI_KLASOR, exist_ok=True)

    # -------- 1) Model + kolon standardı --------
    print("1) Model + kolon standardı yükleniyor...")
    if not MODEL_YOLU.exists():
        raise FileNotFoundError(f"Model bulunamadı: {MODEL_YOLU}")

    pipe = joblib.load(MODEL_YOLU)

    if not GIRDI_KOLONLARI_YOLU.exists():
        raise FileNotFoundError(f"Girdi kolon standardı bulunamadı: {GIRDI_KOLONLARI_YOLU}")

    girdi_kolonlari = joblib.load(GIRDI_KOLONLARI_YOLU)

    # -------- 2) 25k veri --------
    print("2) 25k veri okunuyor...")
    df = pd.read_excel(
        VERI_YOLU_25K,
        sheet_name=SAYFA_ADI,
        na_values=["x", "X", " x ", "X "]
    )

    # Etiket kolonunu drop
    if ETIKET_KOLONU in df.columns:
        df = df.drop(columns=[ETIKET_KOLONU], errors="ignore")

    # Destek departmanını tamamen dışarı al
    if "Departman" in df.columns:
        mask_destek = df["Departman"].astype("string").str.contains("destek", case=False, na=False)
        df = df[~mask_destek].copy()

    # Raporda göstereceğimiz kimlik kolonları
    rapor_kolonlari = safe_cols(df, ["Departman", "DepNo", "CalisanNo", "Ad", "Soyad", "MeslekiRolu"])

    # -------- 3) Rapor için türev özellikler (raporlama amaçlı) --------
    print("3) Türev kolonlar hazırlanıyor...")
    df = add_derived_features(df)

    # -------- 4) Model tahmini (ihtimal) --------
    print("4) Model ihtimalleri hesaplanıyor (predict_proba)...")

    # Modele girecek seti standardize et
    X = df.copy()

    # Eksik kolon varsa ekle
    missing = [c for c in girdi_kolonlari if c not in X.columns]
    for c in missing:
        X[c] = np.nan

    # Fazla kolon varsa at
    extra = [c for c in X.columns if c not in girdi_kolonlari]
    if extra:
        X = X.drop(columns=extra, errors="ignore")

    # kolon sırası sabit
    X = X[girdi_kolonlari]

    proba = pipe.predict_proba(X)  # 0=IsFeshi, 1=MevkiDevami, 2=Terfi

    # -------- 5) Gelişim ihtiyaç skoru (0..100) --------
    print("5) Gelişim ihtiyaç skoru (0..100) hesaplanıyor...")

    dept_col = "Departman" if "Departman" in df.columns else None
    if dept_col is None:
        raise ValueError("Departman kolonu yok. Gelişim raporu için 'Departman' gerekli.")

    # Önce metrik kolonlarını temizle/normalize et (varsa)
    # Oran beklenenler
    for c in ["CalHataOrani", "CalSLAUyumOrani", "CalZorunluEgitimTamamlamaOrani", "TamamlamaOrani", "GecOrani", "BasarisizlikOrani"]:
        if c in df.columns:
            df[c] = ensure_percent_0_1(df[c])

    # Sayısal beklenenler
    for c in ["CalReworkSayisi", "CalOrtCozumSuresi", "CalEgitimGecikmeGunu", "CalSertifikaPuani",
              "CalSurecIyilestirmeSkoru", "CalOtomasyonKatkisi", "CalIsTatmini_M1", "ProjeBasariSkoru_01"]:
        if c in df.columns:
            df[c] = to_numeric_series(df[c])

    # Kullanılabilir metrikler
    pos_cols = [c for c in POS_METRICS.keys() if c in df.columns]
    neg_cols = [c for c in NEG_METRICS.keys() if c in df.columns]

    if len(pos_cols) + len(neg_cols) == 0:
        raise ValueError("Gelişim skoru için metrik bulunamadı (kolon isimlerini kontrol et).")

    # Departman bazlı benchmark'lar
    # Pozitiflerde bench=median, low=q25
    pos_bench = {}
    pos_low = {}
    for c in pos_cols:
        ql, qm, _qh = dept_quantiles(df, dept_col, c, q_low=0.25, q_med=0.5, q_high=0.75)
        pos_bench[c] = qm
        pos_low[c] = ql

    # Negatiflerde bench=median, high=q75
    neg_bench = {}
    neg_high = {}
    for c in neg_cols:
        _ql, qm, qh = dept_quantiles(df, dept_col, c, q_low=0.25, q_med=0.5, q_high=0.75)
        neg_bench[c] = qm
        neg_high[c] = qh

    # Çalışan bazlı gap'ler
    gap_cols = {}
    for c in pos_cols:
        x = df[c]
        bench = df[dept_col].map(pos_bench[c])
        low = df[dept_col].map(pos_low[c])
        gap = normalized_gap_positive(x, bench, low)
        coln = f"gap_{c}"
        df[coln] = gap
        gap_cols[c] = coln

    for c in neg_cols:
        x = df[c]
        bench = df[dept_col].map(neg_bench[c])
        high = df[dept_col].map(neg_high[c])
        gap = normalized_gap_negative(x, bench, high)
        coln = f"gap_{c}"
        df[coln] = gap
        gap_cols[c] = coln

    # Ağırlıklı skor (0..100)
    weights = {}
    weights.update({c: POS_METRICS[c] for c in pos_cols})
    weights.update({c: NEG_METRICS[c] for c in neg_cols})
    w_sum = float(np.sum(list(weights.values())))

    score = 0.0
    for c, w in weights.items():
        score = score + w * df[gap_cols[c]]

    df["GelisimIhtiyacSkoru_0_100"] = (score / (w_sum if w_sum > 0 else 1.0)) * 100.0
    df["GelisimIhtiyacSkoru_0_100"] = df["GelisimIhtiyacSkoru_0_100"].clip(0.0, 100.0)

    # -------- 6) Kategori tespiti + eğitim etiketi + öneriler --------
    print("6) Kategori + eğitim etiketi + öneriler hazırlanıyor...")

    # kategori skorunu: kategorideki gap'lerin ortalaması (var olanlarla)
    cat_scores = {}
    for cat, cols in CATEGORY_METRICS.items():
        present = [c for c in cols if f"gap_{c}" in df.columns]
        if not present:
            continue
        df[f"cat_{cat}"] = df[[f"gap_{c}" for c in present]].mean(axis=1)
        cat_scores[cat] = f"cat_{cat}"

    # En büyük kategori
    if cat_scores:
        cat_cols_df = list(cat_scores.values())
        # idxmax satır bazında
        best_idx = df[cat_cols_df].values.argmax(axis=1)
        best_cat = [list(cat_scores.keys())[i] for i in best_idx]
        df["EnOnemliIhtiyacKategori"] = best_cat
    else:
        df["EnOnemliIhtiyacKategori"] = "Genel"

    # Eğitim etiketi + öneriler
    def training_label(cat: str) -> str:
        return CATEGORY_TO_TRAINING_TR.get(cat, "Genel Gelişim Planı")

    def recommendations(cat: str) -> str:
        recs = CATEGORY_TO_RECO_TR.get(cat, [])
        if not recs:
            return "Departman içi hedeflerle uyumlu olacak şekilde kısa gelişim planı çıkarıldı."
        return " | ".join(recs)

    df["EgitimEtiketi_TR"] = df["EnOnemliIhtiyacKategori"].apply(training_label)
    df["Oneri_TR"] = df["EnOnemliIhtiyacKategori"].apply(recommendations)

    # -------- 7) Çıktı dataframe'leri --------
    print("7) Çıktı tabloları hazırlanıyor...")

    out = pd.DataFrame()
    for c in rapor_kolonlari:
        out[c] = df[c]

    # İhtimal sütunlarını Türkçe yaz
    out["Ihtimal_IsFeshi"] = proba[:, 0]
    out["Ihtimal_MevkiDevami"] = proba[:, 1]
    out["Ihtimal_Terfi"] = proba[:, 2]

    # yüzdeye çevir (okunabilir)
    out["Ihtimal_IsFeshi_%"] = out["Ihtimal_IsFeshi"] * 100.0
    out["Ihtimal_MevkiDevami_%"] = out["Ihtimal_MevkiDevami"] * 100.0
    out["Ihtimal_Terfi_%"] = out["Ihtimal_Terfi"] * 100.0

    # Skorlar
    out["GelisimIhtiyacSkoru_0_100"] = df["GelisimIhtiyacSkoru_0_100"]
    out["EnOnemliIhtiyacKategori"] = df["EnOnemliIhtiyacKategori"]
    out["EgitimEtiketi"] = df["EgitimEtiketi_TR"]
    out["Oneriler"] = df["Oneri_TR"]

    # Departman içi gelişim sırası (skor yüksek = daha çok ihtiyaç)
    if "Departman" in out.columns:
        out["GelisimRank_Dep"] = (
            out.groupby("Departman")["GelisimIhtiyacSkoru_0_100"]
               .rank(method="first", ascending=False)
               .astype(int)
        )
    else:
        out["GelisimRank_Dep"] = np.nan

    out["GelisimRank_Global"] = out["GelisimIhtiyacSkoru_0_100"].rank(method="first", ascending=False).astype(int)

    # Top 50 ihtiyaç (her departman)
    top50_dep = None
    if "Departman" in out.columns:
        top50_dep = out[out["GelisimRank_Dep"] <= 50].copy()

    # Departman özet (makul aralıklar)
    dept_summary = (
        out.groupby("Departman", dropna=False)
           .agg(
               CalisanSayisi=("CalisanNo", "count") if "CalisanNo" in out.columns else ("GelisimIhtiyacSkoru_0_100", "size"),
               OrtalamaGelisimSkoru=("GelisimIhtiyacSkoru_0_100", "mean"),
               MedyanGelisimSkoru=("GelisimIhtiyacSkoru_0_100", "median"),
               OrtalamaTerfiIhtimalYuzde=("Ihtimal_Terfi_%", "mean"),
               OrtalamaIsFeshiIhtimalYuzde=("Ihtimal_IsFeshi_%", "mean"),
           )
           .reset_index()
    )

    # Eğitim ihtiyaç özeti: departman bazlı etiket sayıları
    train_counts = (
        out.groupby(["Departman", "EgitimEtiketi"], dropna=False)
           .size()
           .reset_index(name="KisiSayisi")
    )

    # Genel eğitim ihtiyaç özeti
    train_counts_all = (
        out.groupby(["EgitimEtiketi"], dropna=False)
           .size()
           .reset_index(name="KisiSayisi")
           .sort_values("KisiSayisi", ascending=False)
    )

    # -------- 8) Format: 2 ondalık --------
    print("8) Yuvarlama (2 ondalık) uygulanıyor...")

    round_cols = [
        "Ihtimal_IsFeshi", "Ihtimal_MevkiDevami", "Ihtimal_Terfi",
        "Ihtimal_IsFeshi_%", "Ihtimal_MevkiDevami_%", "Ihtimal_Terfi_%",
        "GelisimIhtiyacSkoru_0_100",
    ]
    for c in round_cols:
        if c in out.columns:
            out[c] = out[c].astype(float).round(2)

    for c in ["OrtalamaGelisimSkoru", "MedyanGelisimSkoru", "OrtalamaTerfiIhtimalYuzde", "OrtalamaIsFeshiIhtimalYuzde"]:
        if c in dept_summary.columns:
            dept_summary[c] = dept_summary[c].astype(float).round(2)

    # -------- 9) Excel yaz --------
    print("9) Excel yazılıyor:", CIKTI_EXCEL)

    # Sıralı görünüm
    sort_cols = ["Departman", "GelisimRank_Dep"] if "Departman" in out.columns else ["GelisimRank_Global"]
    out = out.sort_values(sort_cols, ascending=[True, True] if len(sort_cols) == 2 else [True])

    with pd.ExcelWriter(CIKTI_EXCEL, engine="openpyxl") as writer:
        out.to_excel(writer, index=False, sheet_name="Calisan_Gelisim")

        if top50_dep is not None:
            # yuvarlama
            for c in round_cols:
                if c in top50_dep.columns:
                    top50_dep[c] = top50_dep[c].astype(float).round(2)
            top50_dep.to_excel(writer, index=False, sheet_name="Top50_Ihtiyac_Dep")

        dept_summary.to_excel(writer, index=False, sheet_name="Departman_Ozet")
        train_counts.to_excel(writer, index=False, sheet_name="Egitim_Ihtiyac_Ozet")
        train_counts_all.to_excel(writer, index=False, sheet_name="Egitim_Ihtiyac_Genel")

    print("Bitti ✅ ->", CIKTI_EXCEL)


if __name__ == "__main__":
    main()
