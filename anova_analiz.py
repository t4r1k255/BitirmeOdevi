import os
import numpy as np
import pandas as pd
from scipy import stats

# =========================
# ayaları yaz
# =========================
GIRDI_EXCEL = "outputs/departman_ici_siralama.xlsx"
SHEET_NAME = "DepartmanIciSiralama"   # sende farklıysa değiştir (veya None yap)
CIKTI_KLASOR = "outputs"
CIKTI_EXCEL = os.path.join(CIKTI_KLASOR, "anova_raporu.xlsx")

# Gruplama kolonu (Departman bazlı analiz)
GROUP_COL = "Departman"

# Terfi ihtimali kolon adı (sende hangisi varsa otomatik bulmaya çalışacağız)
CANDIDATE_Y_COLS = [
    "ihtimal_terfi",
    "Ihtimal_Terfi",
    "p_Terfi",
    "proba_terfi",
    "proba_terfi",
    "p_terfi",
    "Terfi_Ihtimal",
]

DESTEK_HARIC = True  # Destek departmanı analize girmesin


def find_y_col(df: pd.DataFrame) -> str:
    # Önce aday isimlerden bul
    for c in CANDIDATE_Y_COLS:
        if c in df.columns:
            return c

    # Bulamazsa: "terfi" geçen numeric kolon ara
    for c in df.columns:
        if "terfi" in str(c).lower():
            if pd.api.types.is_numeric_dtype(df[c]):
                return c

    raise ValueError("Terfi ihtimali kolonu bulunamadı. Excel'deki kolon adını kontrol et.")


def eta_squared(y: np.ndarray, groups: list[np.ndarray]) -> float:
    # SS_between / SS_total
    grand_mean = np.mean(y)
    ss_total = np.sum((y - grand_mean) ** 2)
    ss_between = 0.0
    for g in groups:
        if len(g) == 0:
            continue
        ss_between += len(g) * (np.mean(g) - grand_mean) ** 2
    return float(ss_between / ss_total) if ss_total > 0 else np.nan


def main():
    os.makedirs(CIKTI_KLASOR, exist_ok=True)

    print("1) Excel okunuyor...")
    df = pd.read_excel(GIRDI_EXCEL, sheet_name=SHEET_NAME if SHEET_NAME else 0)
    df.columns = [str(c).strip() for c in df.columns]

    if GROUP_COL not in df.columns:
        raise ValueError(f"'{GROUP_COL}' kolonu yok. Excel'deki grup kolon adını kontrol et.")

    y_col = find_y_col(df)
    print("2) Kullanılacak hedef kolon (Y):", y_col)

    # Destek hariç
    if DESTEK_HARIC and GROUP_COL in df.columns:
        m = df[GROUP_COL].astype("string").fillna("").str.contains("destek", case=False, na=False)
        df = df[~m].copy()

    # Numeric çevir
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

    # NaN temizle
    df2 = df[[GROUP_COL, y_col]].dropna().copy()

    # Grup özet
    summary = (
        df2.groupby(GROUP_COL)[y_col]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={
            "count": "n",
            "mean": "ortalama_terfi_ihtimali",
            "std": "std",
            "min": "min",
            "max": "max"
        })
    )

    # Çok küçük grupları ele (ANOVA için en az 2 gözlem mantıklı)
    valid_groups = summary[summary["n"] >= 2][GROUP_COL].tolist()
    df2 = df2[df2[GROUP_COL].isin(valid_groups)].copy()

    # ANOVA grupları
    group_arrays = []
    group_names = []
    for gname, gdf in df2.groupby(GROUP_COL):
        arr = gdf[y_col].to_numpy(dtype=float)
        if len(arr) >= 2:
            group_arrays.append(arr)
            group_names.append(gname)

    print(f"3) Grup sayısı: {len(group_arrays)} (n>=2 olanlar)")

    if len(group_arrays) < 2:
        raise ValueError("ANOVA için en az 2 grup gerekli. Grupların n değerlerini kontrol et.")

    # One-way ANOVA
    print("4) ANOVA çalıştırılıyor...")
    F, p = stats.f_oneway(*group_arrays)

    y_all = df2[y_col].to_numpy(dtype=float)
    eta2 = eta_squared(y_all, group_arrays)

    anova_df = pd.DataFrame([{
        "Y": y_col,
        "Gruplama": GROUP_COL,
        "Grup Sayısı": len(group_arrays),
        "Toplam Gözlem": len(df2),
        "F": float(F),
        "p değeri": float(p),
        "eta-kare": float(eta2),
    }])

    # Pairwise (Bonferroni düzeltmeli) — dışarıdan bakan kişinin “hangi departmanlar ayrışıyor” kısmı için
    print("5) Pairwise testler hazırlanıyor (Welch t-test + Bonferroni)...")
    pairs = []
    m = 0
    for i in range(len(group_arrays)):
        for j in range(i + 1, len(group_arrays)):
            m += 1

    idx = 0
    for i in range(len(group_arrays)):
        for j in range(i + 1, len(group_arrays)):
            a = group_arrays[i]
            b = group_arrays[j]
            t, p2 = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
            mean_a = float(np.mean(a))
            mean_b = float(np.mean(b))
            diff = mean_a - mean_b
            p_adj = min(float(p2) * m, 1.0)  # Bonferroni

            pairs.append({
                "Grup_A": group_names[i],
                "Grup_B": group_names[j],
                "n_A": len(a),
                "n_B": len(b),
                "Ort_A": round(mean_a, 4),
                "Ort_B": round(mean_b, 4),
                "Fark_(A-B)": round(diff, 4),
                "t": float(t),
                "p": float(p2),
                "p_bonferroni": float(p_adj),
            })
            idx += 1

    pairwise_df = pd.DataFrame(pairs).sort_values("p_bonferroni")

    # Yuvarlama (rapor okunabilirliği)
    summary["ortalama_terfi_ihtimali"] = summary["ortalama_terfi_ihtimali"].round(4)
    summary["std"] = summary["std"].round(4)

    # Excel yaz
    print("6) Excel yazılıyor:", CIKTI_EXCEL)
    with pd.ExcelWriter(CIKTI_EXCEL, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="GrupOzet")
        anova_df.to_excel(writer, index=False, sheet_name="ANOVA")
        pairwise_df.to_excel(writer, index=False, sheet_name="Pairwise_Bonferroni")

    print("Bitti ✅ ->", CIKTI_EXCEL)
    print("ANOVA sonucu: F =", float(F), "| p =", float(p), "| eta^2 =", float(eta2))


if __name__ == "__main__":
    main()
