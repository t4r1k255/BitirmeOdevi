\# ML-Based Career Management for White-Collar Employees



Graduation thesis project — Sakarya University Industrial Engineering, 2026.



\## What it does

Predicts career outcomes (promotion / position retention / termination) for white-collar banking employees using an XGBoost classifier. Outputs actionable HR reports for management decisions.



\## Dataset

\- 25,000 synthetic HR records across 16 banking departments

\- 44 performance metrics per employee

\- 8,000 labeled samples used for training



\## Methods

\- XGBoost (3-class classification)

\- SHAP explainability (feature importance)

\- Statistical validation: ANOVA (F), paired t-test, Bonferroni correction, η² (eta-square)



\## Output Reports

| File | Description |

|------|-------------|

| `departman\_ici\_siralama.xlsx` | Promotion candidate ranking by department |

| `gelisim\_ihtiyaci\_raporu.xlsx` | Employee development needs analysis |

| `gelisim\_oneri\_raporu.xlsx` | Improvement recommendations |

| `anket\_katilim\_analizi.xlsx` | Survey participation analysis |

| `shap\_terfi\_bar.png` / `shap\_terfi\_beeswarm.png` | SHAP feature importance visualizations |



\## How to Run

```bash

pip install -r requirements.txt

python train\_model.py

python run\_all\_reports.py

```



\## Tech Stack

Python, XGBoost, scikit-learn, SHAP, pandas, matplotlib, seaborn, openpyxl

