import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from .ozellik_uretici import OzellikUretici
from .konfigurasyon import VERI_YOLU, SAYFA_ADI, ETIKET_KOLONU, ETIKET_HARITASI, TANIMSAL_DROP

def modeli_egit_ve_kaydet(model_yolu="models/xgb_pipeline.joblib"):
    df = pd.read_excel(VERI_YOLU, sheet_name=SAYFA_ADI, na_values=["x", "X", " x ", "X "])

    # sadece etiketli
    df = df[df[ETIKET_KOLONU].isin(ETIKET_HARITASI.keys())].copy()
    y = df[ETIKET_KOLONU].map(ETIKET_HARITASI)

    # destek egitim disi (varsa)
    if "Departman" in df.columns:
        df = df[~df["Departman"].astype(str).str.contains("destek", case=False, na=False)].copy()
        y = df[ETIKET_KOLONU].map(ETIKET_HARITASI)

    X_raw = df.drop(columns=[ETIKET_KOLONU]).copy()

    # inference standardı için ham kolonları kaydet
    joblib.dump(list(X_raw.columns), "models/girdi_kolonlari.joblib")
    joblib.dump(ETIKET_HARITASI, "models/etiket_haritasi.joblib")

    drop_cols = [c for c in TANIMSAL_DROP if c in X_raw.columns]

    ozellik_uret = OzellikUretici(drop_cols=drop_cols)

    sayisal = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    kategorik = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    on_isleme = ColumnTransformer(
        transformers=[
            ("sayisal", sayisal, selector(dtype_include=np.number)),
            ("kategorik", kategorik, selector(dtype_exclude=np.number)),
        ],
        remainder="drop"
    )

    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="mlogloss"
    )

    pipe = Pipeline([
        ("ozellik_uret", ozellik_uret),
        ("on_isleme", on_isleme),
        ("model", xgb),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    classes = np.array([0, 1, 2])
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    w_map = {c: ww for c, ww in zip(classes, w)}
    w_map[2] *= 2.5  # Terfi
    w_map[0] *= 1.2  # IsFeshi (opsiyonel)
    sample_weight = y_train.map(w_map).values

    pipe.fit(X_train, y_train, model__sample_weight=sample_weight)

    pred = pipe.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
    print("\nReport:\n", classification_report(y_test, pred, digits=4))

    joblib.dump(pipe, model_yolu)
    print(f"Model kaydedildi: {model_yolu}")

    return pipe
