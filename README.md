# Thin-File Credit Scoring: Lending to First-Time Borrowers

A credit-default model for applicants with little or no credit history, framed as a **business decision**: how to run a small-ticket consumer-lending product in India, such as a PayLater / BNPL line or a small personal loan at e-commerce checkout.

## The problem

Many applicants for small-ticket credit in India are **new to credit** and have no bureau record. A "no score, no loan" rule is safe but rejects all of them. This project asks:

> Using only data we can observe for a first-time borrower, can we approve a meaningful share of thin-file applicants while keeping expected losses at or below what we already accept on customers with a credit history?

Success is measured in **approval rate, expected loss and expected profit**, not only in AUC.

## Approach

| Step | What was done |
|---|---|
| Data | Kaggle [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data): `application_train.csv` (307,511 applications), plus `bureau.csv` and `previous_application.csv` |
| Thin-file definition | No record in the credit bureau: **14.3%** of applicants |
| Features | Affordability ratios, age and employment stability, document/contact/address aggregates, `EXT_SOURCE` summaries, bureau and previous-application aggregates |
| Core comparison | **Full** feature set (161 features) vs. **alternative-data only** (121 features: no bureau, prior-credit or `EXT_SOURCE` data). Gender is excluded from both and used only for the fairness audit |
| Split | Stratified 60/20/20 on target × thin-file status. LightGBM early stopping uses a slice of the training set. Validation is used for model comparison, calibration and threshold choice. **Test is scored once, at the end** |
| Models | Logistic regression (sklearn Pipeline: imputation, scaling, one-hot encoding) and LightGBM (4-point grid with early stopping) |
| Calibration | Reliability curves, Brier score and ECE; out-of-fold isotonic calibration (kept only if it lowers the Brier score) |
| Explainability | TreeSHAP summary, signal-family importance, and individual waterfalls for thin-file applicants |
| Decisions | Profit curves over PD cut-offs from 0 to 1, a profit-maximising and a conservative policy, the thin-file opportunity vs. a naive rule, LGD × margin sensitivity, and a segment fairness check |

**Business assumptions (illustrative, not real product economics):** average loan ₹25,000, 6-month tenor, 16% margin over the tenor, 10% annual cost of funds, 85% LGD. The implied **break-even PD is 10.9%**.

## Key results

All figures come from the executed notebook (`outputs/results.json`).

**Model performance on the test set (ROC-AUC)**

| Model | Feature set | All applicants | Thin-file only |
|---|---|---|---|
| Logistic regression | Full | 0.755 | 0.723 |
| Logistic regression | Alternative-data | 0.682 | 0.681 |
| LightGBM | Full | 0.769 | 0.750 |
| **LightGBM** | **Alternative-data** | **0.708** | **0.704** |

On test, the alternative-data LightGBM scores KS 0.304 and Gini 0.416; the full model scores KS 0.399 and Gini 0.537.

**Business outcomes**

- **Thin-file opportunity:** the naive rule rejects every thin-file applicant; its approved book defaults at 7.73%. The alternative-data model can approve **85.6% of thin-file applicants** at a 7.66% default rate, no worse than the naive book. That is **≈12,200 extra approvals per 100,000 applications** (validation).
- **Launch policy:** alternative-data LightGBM, approve if PD ≤ **0.075** (the conservative cut-off). It keeps 93% of the maximum expected profit, which needs a 0.110 cut-off. On the **held-out test set** it approved **61.8%** of applicants with a **4.48%** default rate. Realised profit was ₹9.99 lakh per 1,000 applicants, against ₹10.18 lakh expected from validation.
- **Calibration:** LightGBM without class weights was already well calibrated. Test mean predicted PD was 8.02% against an observed 8.07%, so isotonic calibration was not needed.
- **Threshold = break-even PD:** the profit-maximising cut-off (0.110) sits on the theoretical break-even PD (0.109). Across LGD 60–100% × margin 10–25% it ranges from 0.045 to 0.235, with approval rates from 33% to 97%.
- **What drives risk without a credit history:** loan structure (annuity-to-credit, credit-to-goods), employment tenure, age, occupation and organisation type, education, and phone/ID stability.
- **Fairness:** approval rates are much lower for under-25s (29% vs. 83% for 55+) and for region rating 3. Those approved in these groups still default *more*, so the gaps reflect measured risk. For the product's core young segment, the recommendation is start-small limits rather than declines.

## Limitations

- The data is **not Indian**. The model and cut-offs must be re-estimated on local data, such as a pilot cohort.
- The business assumptions are **illustrative**. See the sensitivity analysis.
- Labels exist only for loans Home Credit approved (**selection bias**). There is no out-of-time validation.
- `EXT_SOURCE_*` are opaque. They were treated as credit-history data; if they are partly alternative data, the alternative-data model is understated.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Data: needs a Kaggle account and one-time acceptance of the competition rules at
# https://www.kaggle.com/competitions/home-credit-default-risk/rules
# API token from https://www.kaggle.com/settings -> API
#   new "KGAT_..." tokens -> ~/.kaggle/access_token ; legacy tokens -> ~/.kaggle/kaggle.json
kaggle competitions download -c home-credit-default-risk -p data
unzip data/home-credit-default-risk.zip -d data           # PowerShell: Expand-Archive data\home-credit-default-risk.zip data

jupyter nbconvert --to notebook --execute --inplace thin_file_credit_scoring.ipynb
# or open it in Jupyter and Run All
```

The run takes about 2 minutes on a 12-core laptop CPU and is deterministic (seed 42). If `application_train.csv` is missing, the notebook prints these download instructions and stops cleanly.

## Files

| File | Contents |
|---|---|
| `thin_file_credit_scoring.ipynb` | The full analysis (executed, with outputs) |
| `requirements.txt` | Python dependencies (Python 3.10+) |
| `outputs/results.json` | Every headline number, written by the notebook |
| `data/` | Kaggle CSVs (not committed; the competition rules prohibit redistribution) |
