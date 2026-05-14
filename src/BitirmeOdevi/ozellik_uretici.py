import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class OzellikUretici(BaseEstimator, TransformerMixin):
    """
    - Proje oranları: tamam/gec/basarisiz oranları (+ opsiyonel skor)
    - Tenure (omur) mevki bazında normalize: aynı mevkide kıyas
    - MeslekiRolu sadece normalize için kullanılabilir, sonra drop edilir
    """
    def __init__(
        self,
        col_toplam="Toplam Proje",
        col_tamam="Tamamladı",
        col_gec="Geç Tamamladı",
        col_basarisiz="Tamamlayamadı",
        col_sirket="CalSirketOmru",
        col_dep="CalDepOmru",
        col_mevki="MeslekiRolu",
        drop_cols=None,
        drop_raw_counts=False,
        clip_z=3.0,
    ):
        self.col_toplam = col_toplam
        self.col_tamam = col_tamam
        self.col_gec = col_gec
        self.col_basarisiz = col_basarisiz
        self.col_sirket = col_sirket
        self.col_dep = col_dep
        self.col_mevki = col_mevki
        self.drop_cols = drop_cols or []
        self.drop_raw_counts = drop_raw_counts
        self.clip_z = clip_z

        self._role_stats_ = {}
        self._global_stats_ = {}

    def fit(self, X, y=None):
        df = X.copy()

        # numeric'e zorla
        for c in [self.col_sirket, self.col_dep]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # global mean/std
        for c in [self.col_sirket, self.col_dep]:
            if c in df.columns:
                mu = float(df[c].mean(skipna=True))
                sd = float(df[c].std(skipna=True))
                self._global_stats_[c] = (mu, sd if sd and sd > 1e-9 else 1.0)

        # role bazlı mean/std
        if self.col_mevki in df.columns:
            grp = df.groupby(self.col_mevki, dropna=False)
            for c in [self.col_sirket, self.col_dep]:
                if c in df.columns:
                    mu = grp[c].mean()
                    sd = grp[c].std().replace(0, np.nan).fillna(1.0)
                    self._role_stats_[c] = pd.DataFrame({"mu": mu, "sd": sd})

        return self

    def transform(self, X):
        df = X.copy()

        # --- Proje oranları ---
        for c in [self.col_toplam, self.col_tamam, self.col_gec, self.col_basarisiz]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        if self.col_toplam in df.columns:
            top = df[self.col_toplam].replace(0, np.nan)

            if self.col_tamam in df.columns:
                df["TamamlamaOrani"] = (df[self.col_tamam] / top).clip(0, 1)
            if self.col_gec in df.columns:
                df["GecOrani"] = (df[self.col_gec] / top).clip(0, 1)
            if self.col_basarisiz in df.columns:
                df["BasarisizlikOrani"] = (df[self.col_basarisiz] / top).clip(0, 1)

            if {"TamamlamaOrani", "GecOrani", "BasarisizlikOrani"} <= set(df.columns):
                df["ProjeBasariSkoru"] = (
                    1.00 * df["TamamlamaOrani"]
                    - 0.60 * df["GecOrani"]
                    - 1.20 * df["BasarisizlikOrani"]
                )

        # --- Tenure z-score (mevki bazlı) ---
        def zscore_by_role(col):
            if col not in df.columns:
                return
            df[col] = pd.to_numeric(df[col], errors="coerce")

            if self.col_mevki in df.columns and col in self._role_stats_:
                stat = self._role_stats_[col]
                keys = df[self.col_mevki]
                mu = keys.map(stat["mu"]).astype(float)
                sd = keys.map(stat["sd"]).astype(float)
                z = (df[col] - mu) / sd
            else:
                mu, sd = self._global_stats_.get(col, (0.0, 1.0))
                z = (df[col] - mu) / sd

            df[col + "_Z"] = z.clip(-self.clip_z, self.clip_z)

        zscore_by_role(self.col_sirket)
        zscore_by_role(self.col_dep)

        # ham count’ları istemiyorsan düşür
        if self.drop_raw_counts:
            drop_counts = [c for c in [self.col_tamam, self.col_gec, self.col_basarisiz] if c in df.columns]
            df = df.drop(columns=drop_counts, errors="ignore")

        # tanımsal kolonları drop et
        if self.drop_cols:
            df = df.drop(columns=[c for c in self.drop_cols if c in df.columns], errors="ignore")

        return df
