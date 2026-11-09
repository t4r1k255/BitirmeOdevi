# ML-Based Career Management for White-Collar Employees

Graduation thesis project. Industrial Engineering, Sakarya University (2026).

**[English](#english) · [Türkçe](#türkçe)**

---

## English

### What it does

Predicts the six-month career outcome of white-collar banking employees: **promotion**, **retention in current position** or **termination**. It uses an XGBoost classifier and turns the predictions into HR reports that managers can act on.

### Dataset

- 25,000 synthetic HR records across 16 banking departments
- 44 features per employee
- 8,000 labeled samples used for model training

All records are synthetic. No real employee data is included.

### Methods

- **Model:** XGBoost, 3-class classification (`IsFeshi` / `MevkiDevami` / `Terfi`)
- **Explainability:** SHAP (bar and beeswarm plots), feature importance focused on promotion recall
- **Statistical validation:** ANOVA (F-test), pairwise t-tests with Bonferroni correction, η² effect size

### Output reports

| File | Description |
|---|---|
| `outputs/departman_ici_siralama.xlsx` | Promotion-candidate ranking within each department |
| `outputs/gelisim_ihtiyaci_raporu.xlsx` | Development needs per employee |
| `outputs/gelisim_oneri_raporu.xlsx` | Improvement recommendations |
| `outputs/anket_katilim_analizi.xlsx` | Employee survey participation analysis |
| `outputs/anova_raporu.xlsx` | ANOVA and pairwise Bonferroni results |
| `outputs/kolon_onemi_promote_recall.xlsx` | Feature importance for promotion recall |
| `outputs/shap_terfi_bar.png`, `outputs/shap_terfi_beeswarm.png` | SHAP plots for the promotion class |

### How to run

The training code lives in `src/BitirmeOdevi`, so `src` must be on the Python path.

```bash
pip install -r requirements.txt

# Train the model (writes models/xgb_pipeline.joblib)
PYTHONPATH=src python train_model.py          # macOS / Linux
# set PYTHONPATH=src && python train_model.py # Windows (cmd)

# Generate every report into outputs/
python run_all_reports.py

# Optional: run only some scripts, or keep going after an error
python run_all_reports.py --only shap_analiz.py anova_analiz.py
python run_all_reports.py --continue-on-error
```

Pre-trained models and generated outputs are already included in `models/` and `outputs/`, so you can browse the results without running anything.

### Project structure

```
data/                  Synthetic employee list and training set (Excel)
models/                Trained pipeline and column/label mappings (joblib)
outputs/               Generated reports and SHAP plots
src/BitirmeOdevi/      Configuration, feature engineering, training
train_model.py         Model training entry point
run_all_reports.py     Runs all report and analysis scripts in order
*.py                   Individual report / analysis scripts
```

### Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · SHAP · SciPy · matplotlib · openpyxl · joblib

---

## Türkçe

### Ne yapar?

Bankacılık sektöründeki beyaz yakalı çalışanların 6 aylık kariyer sonucunu tahmin eder: **terfi**, **mevcut pozisyonda devam** veya **iş akdinin feshi**. XGBoost sınıflandırıcısı kullanır ve tahminleri yöneticilerin karar verirken kullanabileceği İK raporlarına dönüştürür.

### Veri seti

- 16 banka departmanında 25.000 sentetik çalışan kaydı
- Çalışan başına 44 değişken
- Model eğitimi için 8.000 etiketli örnek

Tüm kayıtlar sentetiktir, gerçek çalışan verisi içermez.

### Yöntemler

- **Model:** XGBoost, 3 sınıflı sınıflandırma (`IsFeshi` / `MevkiDevami` / `Terfi`)
- **Açıklanabilirlik:** SHAP (bar ve beeswarm grafikleri), terfi recall'una göre değişken önemi
- **İstatistiksel doğrulama:** ANOVA (F testi), Bonferroni düzeltmeli ikili t-testleri, η² etki büyüklüğü

### Çıktı raporları

| Dosya | Açıklama |
|---|---|
| `outputs/departman_ici_siralama.xlsx` | Departman içi terfi adayı sıralaması |
| `outputs/gelisim_ihtiyaci_raporu.xlsx` | Çalışan bazında gelişim ihtiyaçları |
| `outputs/gelisim_oneri_raporu.xlsx` | Gelişim önerileri |
| `outputs/anket_katilim_analizi.xlsx` | Çalışan anketi katılım analizi |
| `outputs/anova_raporu.xlsx` | ANOVA ve Bonferroni ikili karşılaştırma sonuçları |
| `outputs/kolon_onemi_promote_recall.xlsx` | Terfi recall'u için değişken önemi |
| `outputs/shap_terfi_bar.png`, `outputs/shap_terfi_beeswarm.png` | Terfi sınıfı için SHAP grafikleri |

### Nasıl çalıştırılır?

Eğitim kodu `src/BitirmeOdevi` içinde olduğu için `src` klasörünün Python yolunda olması gerekir.

```bash
pip install -r requirements.txt

# Modeli eğit (models/xgb_pipeline.joblib dosyasını üretir)
PYTHONPATH=src python train_model.py          # macOS / Linux
# set PYTHONPATH=src && python train_model.py # Windows (cmd)

# Tüm raporları outputs/ klasörüne üret
python run_all_reports.py

# İsteğe bağlı: sadece bazı scriptleri çalıştır ya da hata olsa da devam et
python run_all_reports.py --only shap_analiz.py anova_analiz.py
python run_all_reports.py --continue-on-error
```

Eğitilmiş modeller ve üretilmiş çıktılar `models/` ve `outputs/` klasörlerinde hazır durur. Hiçbir şey çalıştırmadan sonuçlara göz atabilirsin.

### Kullanılan teknolojiler

Python · pandas · NumPy · scikit-learn · XGBoost · SHAP · SciPy · matplotlib · openpyxl · joblib
