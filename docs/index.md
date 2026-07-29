# Capstone Report — Refresh Opportunity Scoring

- **Author:** Fahad Alam Jamal
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/Fahad-Alam-Jamal/Flyrank_ML_Internship
- **Date:** 2026-07-29

> This report uses the FlyRank internship dataset and the starter pipeline in `work/`. It compares a transparent Week-4 refresh baseline to a client-aware logistic model that ranks pages for refresh review.

## 0. Abstract

Can a model improve the ranking of refresh candidates compared to a transparent hand-crafted score? I use the FlyRank ML Internship dataset release and a starter `refresh_feature_vector.csv` to compare a Week-4 rule baseline against a logistic model trained on the same observable signals. The method is a client-aware holdout evaluation with the label `is_declining_label`, a simple feature matrix of current engagement and search performance metrics, and a baseline score built from visibility, freshness risk, position opportunity, and depth gap. On the held-out split, the learned model shows higher top-K precision and stronger ranking discrimination than the baseline, while the result is framed as decision support rather than a guaranteed production effect. The output is a ranked refresh review queue with clear recommended actions, and the repo includes the notebooks and scripts needed to reproduce the analysis.

## 1. Problem framing

The decision supported is which content item should be prioritized for refresh or review. The unit of analysis is a pseudonymized content item in the FlyRank dataset, and the output is a ranked refresh opportunity score that a human editor or analyst can use to inspect and act on the most promising candidates first. A wrong call means reviewing low-opportunity content instead of higher-opportunity content, which wastes editorial effort and fails to catch pages that are likely to decline. Data and ML help by turning observable search and engagement signals into a better ordering of candidates than a fixed hand-crafted rule.

## 2. Data safety

This work uses the FlyRank internship dataset release, specifically the starter processed feature vector in `work/refresh_feature_vector.csv` and the release documentation in `docs/ml-intern-dataset-and-lane-guide.md`. The full warehouse release is available from Hugging Face at `FlyRank/internship-warehouse`, with the freshest three days removed and the daily performance window ending at 2026-06-30. The dataset includes the pseudonymized content and client IDs, observable metrics such as impressions, clicks, sessions, average position, CTR, engagement rate, scroll rate, content age, and trend direction.

Deliberately excluded fields: any raw identifying strings, any client names, any raw URLs, any raw search queries, and any product-only decision scores. The model does not use `trend_direction` or any direct label-derived column as a feature. Leakage risks considered include feature fields that are derived from the same snapshot as the label; the final feature set was audited in `work/notebooks/w06_validation_audit.ipynb` and the model is presented as decision-support evidence rather than causal proof.

## 3. Baseline

The transparent baseline is a Week-4 hand-crafted refresh score built in `scripts/02_baseline_score.py`. It computes a baseline refresh score from:

- `visibility_score` — percentile rank of log impressions in 90 days,
- `freshness_risk_score` — percentile rank of days since last update,
- `position_opportunity_score` — a visibility-weighted function of average search position,
- `depth_gap_score` — a visibility-weighted score for shorter pages.

Those components are combined as 40% visibility, 30% freshness risk, 25% position opportunity, and 5% depth gap. The baseline is a fair comparison because it ranks the same content rows using observable signal only, and it is evaluated on the same held-out test split as the model.

## 4. Model / analysis

The method is a simple, interpretable logistic regression model trained to predict `is_declining_label` from current observable metrics. Features are drawn from `scripts/ml_utils.py` and include current page and traffic signals such as `search_volume`, `competition`, `cpc`, `word_count`, `log_impressions_90d`, `log_clicks_90d`, `log_sessions_90d`, `content_age_days`, `days_since_last_update`, `ctr`, `avg_position`, `engagement_rate`, `scroll_rate`, and `ai_traffic_pct`. Categorical features include `competition_level`, `content_type`, `main_intent`, `age_tier`, `freshness_tier`, `word_count_tier`, `impression_tier`, and `position_tier`.

The target is the boolean label `is_declining_label`, which represents a content item that is observed to be declining according to the released dataset. The model is trained on a client-aware holdout split to reduce client-level leakage. The baseline is the same ordered score from the rule model, and the evaluation uses the same test rows and ranking metrics for an honest comparison.

## 5. Evaluation

The split design is a client-aware holdout: roughly 20% of clients are held out as a test set, and the split is fixed with random seed 42. This avoids leaking near-duplicate client behavior from train to test. The actual evaluation set contains 2,325 held-out rows from 6 clients, with a base decline rate of 39.1%.

Model vs baseline on the same client-aware holdout split:

- precision@20: baseline 0.150 → model 0.350 (+0.200)
- precision@50: baseline 0.240 → model 0.400 (+0.160)
- precision@100: baseline 0.360 → model 0.440 (+0.080)
- ROC AUC: baseline 0.627 → model 0.702 (+0.075)
- average precision: baseline 0.468 → model 0.523 (+0.056)

This shows the learned ranking provides stronger top-K precision and better discrimination than the Week-4 rule on the same held-out split. A random row-level split context in the notebook is also shown for reference, but the client-aware split is the stronger, more honest validation for this problem.

## 6. Interpretation

The model finds that pages with higher visible demand, older content age, longer update gaps, and weaker current engagement are more likely to be declining. The most important signals are the current search and engagement metrics, not any label-derived fields. The error analysis shows the model tends to over-rank pages that are still active by volume but not yet clearly stale, and to miss some lower-volume cases where the decline signal is subtle.

That interpretation supports a practical use case: the model can surface refresh candidates that are not obvious from a single hand-crafted rule, while editors should still review the top-ranked pages and apply judgment before taking action.

## 7. Recommendation

Ranked recommendations for FlyRank refresh review:

1. Prioritize pages with high model probability and moderate-to-high impressions, because they are the best candidates to improve content impact quickly.
2. Review pages with long `days_since_last_update` and low `ctr` early, since they combine freshness risk with a clear opportunity to improve engagement.
3. Use the logistic model scores to supplement the Week-4 rule baseline, rather than replacing the rule outright; the model is best deployed as decision-support evidence in an editorial review workflow.
4. Monitor pages the model ranks highly but the baseline does not; these are the cases where the learned signal adds the most value and where human validation is most important.
5. Keep the model deployment under a periodic review process, because the dataset snapshot is fixed and the signal may shift over time.

## 8. Reproducibility

To reproduce this work from a fresh clone:

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. If you need the full Hugging Face warehouse release, read the token from `Ttt.txt` and use it with the Hugging Face client:

```python
with open('Ttt.txt', 'r') as f:
    t = f.read().strip()

try:
    from google.colab import userdata
    HF_TOKEN = userdata.get('HF_TOKEN')
except Exception:
    pass

HF_TOKEN = HF_TOKEN or t
```

3. Run the baseline and model pipeline from the repository root:

```bash
python scripts/02_baseline_score.py
python scripts/03_train_model.py
```

4. Open the key notebooks for the experiment and validation:

- `work/notebooks/w04_baseline_score.ipynb`
- `work/notebooks/w05_model.ipynb`
- `work/notebooks/w06_validation_audit.ipynb`
- `work/notebooks/capstone.ipynb`

5. The random seed is fixed at `42` in the scripts and notebooks.

The output artifacts include `work/outputs/baseline_rule_metrics.json` and the model comparison metrics generated by the notebooks.

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset; credit to FlyRank for the anonymized research release and the dataset design. For more information, see https://flyrank.ai.
