from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def stream_run(cmd: list[str], cwd: Path) -> int:
    """
    Komutu çalıştırır, stdout/stderr'ı canlı canlı ekrana basar.
    """
    print("\n$ " + " ".join(cmd))
    start = time.time()

    p = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert p.stdout is not None
    for line in p.stdout:
        print(line, end="")

    rc = p.wait()
    elapsed = time.time() - start
    print(f"\n[EXIT={rc}] ({elapsed:.1f}s)")
    return rc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Bir script hata verirse durma, devam et.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help=(
            "Sadece belirtilen script(ler)i çalıştır. "
            "Örn: python run_all_reports.py --only shap_analiz.py kolon_onemi.py"
        ),
    )
    args = parser.parse_args()

    # Proje kökü: run_all_reports.py hangi klasördeyse orası
    ROOT = Path(__file__).resolve().parent

    # PyCharm hangi interpreter ile çalışıyorsa onu kullan
    PY = sys.executable

    # outputs klasörü yoksa oluştur
    (ROOT / "outputs").mkdir(exist_ok=True)

    # Çalıştıracağım scriptler
    # Hepsi ROOT seviyesinde
    SCRIPTS: list[tuple[str, list[str]]] = [
        # Raporlar
        ("dep_siralama.py", []), #1
        ("gelisim_raporu.py", []), #2
        ("gelisim_oneri_raporu.py", []), #3
        ("anket_katilim_analizi.py", []), #4

        # Analiz çıktıları
        ("kolon_onemi.py", []),      # kolon_onemi_promote_recall.xlsx
        ("shap_analiz.py", []),      # shap_terfi_bar.png + shap_terfi_beeswarm.png
        ("anova_analiz.py", []),     # ANOVA + Pairwise_Bonferroni excel
    ]

    # --only filtrelemesi
    if args.only is not None and len(args.only) > 0:
        allowed = set(args.only)
        SCRIPTS = [x for x in SCRIPTS if x[0] in allowed]

    if not SCRIPTS:
        print("Çalıştırılacak script bulunamadı. --only ile yanlış isim vermiş olabilirsin.")
        sys.exit(1)

    # Ön kontrol: model dosyası (bazı scriptler ister)
    model_path = ROOT / "models" / "xgb_pipeline.joblib"
    if not model_path.exists():
        print(f"UYARI: Model bulunamadı -> {model_path}")
        print("SHAP/kolon önemi/raporların bazıları model ister.")
        print("Gerekirse önce train_model.py çalıştırıp modeli üret.\n")

    # Çalıştır
    for script_name, script_args in SCRIPTS:
        script_path = ROOT / script_name  # <-- DİKKAT: src'nin içinde değil, root'ta!

        if not script_path.exists():
            print(f"SKIP: {script_name} bulunamadı (dosya yok).")
            continue

        print("\n" + "=" * 70)
        print(f"RUN: {script_name}")
        print("=" * 70)

        cmd = [PY, str(script_path), *script_args]
        rc = stream_run(cmd, cwd=ROOT)

        if rc != 0 and not args.continue_on_error:
            print("\nHata alındı, runner durduruldu. Devam etmek istersen:")
            print("  python run_all_reports.py --continue-on-error")
            sys.exit(rc)

    print("\nHepsi bitti ✅")


if __name__ == "__main__":
    main()
