# Practical Application III: Comparing Classifiers

## Notebook

Main analysis notebook: [prompt_III.ipynb](./prompt_III.ipynb)

## Business Goal

The business goal is to help a bank improve the efficiency of its term-deposit marketing campaigns.  
Instead of calling clients at random, the bank wants to identify which clients are most likely to subscribe before a call is made, so that sales effort can be focused on higher-probability leads.

## Dataset

- Source: Portuguese bank direct-marketing campaign data from the UCI repository
- Time period described in the accompanying paper: May 2008 to November 2010
- Campaigns represented: 17
- Records in the working dataset used in the notebook: 41,188
- Working dataset used in the notebook: `data/bank-additional-full.csv`

## Project Files

- `prompt_III.ipynb`: full analysis, modeling, tuning, and conclusions
- `data/bank-additional-full.csv`: main dataset used in the notebook
- `data/bank-additional.csv`: smaller sample dataset
- `data/bank-additional-names.txt`: feature descriptions

## Concise Workflow

1. Read the data and review the paper’s business context.
2. Audit the features and identify placeholder values such as `unknown` and the sentinel value `pdays = 999`.
3. Drop `duration` from realistic modeling because it is only known after the call and causes target leakage.
4. Convert `pdays` into:
   - `pdays_was_contacted_before`
   - `pdays_clean`
5. Encode the target as `y = 1` for `yes` and `0` for `no`.
6. One-hot encode categorical predictors and standardize numeric predictors through a preprocessing pipeline.
7. Create an 80/20 stratified train/test split.
8. Establish a baseline with a majority-class dummy classifier.
9. Fit default models for:
   - Logistic Regression
   - K-Nearest Neighbors
   - Decision Tree
   - Support Vector Machine
10. Compare models using accuracy, precision, and recall.
11. Run grid search with cross-validation and parallel processing (`n_jobs=-1`) for the main classifiers.
12. Add an optional degree-2 polynomial-features logistic model.
13. Test whether removing low-signal features changes performance.
14. Translate the final model output into business terms: expected successful calls versus random calling.

## Feature Preparation

### Dropped Immediately

- `duration`

Reason: this variable is only known after the phone call ends, so it is not appropriate for a realistic predictive model.

### Features Marked as Low-Signal Candidates

- `loan`
- `housing`
- `day_of_week`
- `marital`
- `education`
- `campaign`

These were identified early in the EDA as weaker predictors and were later tested in a reduced-feature experiment.

### Encoding and Transformation Decisions

- One-hot encoded:
  - `job`
  - `marital`
  - `education`
  - `default`
  - `housing`
  - `loan`
  - `contact`
  - `month`
  - `day_of_week`
  - `poutcome`
- Kept numeric and scaled:
  - `age`
  - `campaign`
  - `previous`
  - `emp.var.rate`
  - `cons.price.idx`
  - `cons.conf.idx`
  - `euribor3m`
  - `nr.employed`
  - `pdays_clean`
  - `pdays_was_contacted_before`

## Initial / Default Models Used

### Baseline

- `DummyClassifier(strategy='most_frequent')`

### Simple Logistic Model

- `LogisticRegression(random_state=17, max_iter=1000)`

### Default Comparison Models

- `KNeighborsClassifier()`
- `DecisionTreeClassifier()`
- `SVC()`

All classifier models were used inside a preprocessing pipeline so that the same feature engineering logic was applied consistently.

## Grid Search and Cross-Validation

The notebook then moved from default models to tuned models using cross-validation and parallel execution.

### Cross-Validation Setup

- Stratified K-fold cross-validation
- Current notebook setting: `n_splits = 5`
- `shuffle=True`
- `random_state=17`
- `n_jobs=-1` to use all available CPU cores

### Scoring Tracked

- Accuracy
- Precision
- Recall

### Refit Metric

- Recall

Although the assignment prompt mentions accuracy, the current notebook selects the tuned model by recall. This is a deliberate business choice: with an imbalanced target, accuracy alone can favor models that predict mostly `no`, while recall measures how many real subscribers the campaign would actually find.

### Tuned Parameter Ranges in the Current Notebook
Initially, more values were tested, but with reruns, the choice was narrowed down and slightly adapted so as to search closer to the solutions and to discard ones that don't seem to give much chance of improvement. 
#### Logistic Regression

- Solver: `['liblinear']`
- C: `[0.001, 0.002]`
- Class weight: `['balanced']`

#### Polynomial Logistic Regression

- Degree-2 polynomial features applied to numeric features only
- Solver: `['liblinear']`
- C: `[0.001, 0.002]`
- Class weight: `['balanced']`

#### KNN

- Number of neighbors: `[3, 5, 7]`
- Weights: `['uniform', 'distance']`
- Distance metric: `['euclidean', 'manhattan']`

#### Decision Tree

- Max depth: `[None, 3, 5, 7]`
- Min samples split: `[2, 10, 20]`
- Min samples leaf: `[8, 10, 12]`
- Class weight: `['balanced']`

#### SVM

- Kernel: `['rbf']`
- C: `[1, 1.2]`
- Gamma: `['scale']`
- Class weight: `['balanced']`

## Key Results

### Baseline

- Accuracy: `0.887`
- Precision: `0.000`
- Recall: `0.000`

Interpretation: the baseline looks strong on accuracy only because most clients do not subscribe. It does not identify any subscribers at all, so it is not useful for campaign targeting.

### Simple Logistic Regression

- Test accuracy: `0.903`
- Test precision: `0.683`
- Test recall: `0.253`

Interpretation: the untuned logistic model is conservative. It is precise when it predicts a subscriber, but it misses many actual subscribers.

### Tuned Model Comparison

From the current notebook outputs:

| Model | Test Accuracy | Test Precision | Test Recall |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.789 | 0.307 | 0.692 |
| Polynomial Logistic Regression | 0.791 | 0.308 | 0.688 |
| Decision Tree | 0.732 | 0.243 | 0.652 |
| SVM | 0.850 | 0.397 | 0.638 |
| KNN | 0.876 | 0.436 | 0.337 |

### Best Tuned Model in the Current Notebook

- Model: `Logistic Regression`
- Best parameters:
  - `C = 0.001`
  - `class_weight = 'balanced'`
  - `solver = 'liblinear'`
- Refit metric: `recall`
- Cross-validated accuracy: `0.779`
- Cross-validated recall: `0.668`
- Cross-validated precision: `0.291`
- Test recall: `0.692`
- Test precision: `0.307`
- Test accuracy: `0.789`

## Discussion of Metrics

The notebook reports three main classification metrics:

- `Accuracy`: overall share of correct predictions. This is easy to understand, but on an imbalanced dataset it can be misleading because a model can score well by predicting mostly `no`.
- `Precision`: among the customers the model flags as likely subscribers, how many actually subscribe. This is useful when the bank wants a high hit rate among contacted clients.
- `Recall`: among all customers who really would subscribe, how many the model successfully finds. This is useful when the bank wants to avoid missing too many good leads.

In practical terms:

- a high-accuracy model can still be weak for campaign targeting
- a high-precision model may select too few customers to make the campaign large enough
- a high-recall model reaches more potential subscribers, but it usually requires more calls and lowers the hit rate

That tradeoff is why metric choice should follow the business objective.

## Business Interpretation

Using the best tuned model on the test set:

- Targeted calls predicted as likely subscribers: `2,091`
- Successful calls among those targeted calls: about `642`
- Success rate among targeted calls: `30.70%`
- Share of all actual subscribers captured: `69.18%`
- Actual subscribers still missed by the model: about `286`

If the bank called the same number of clients at random:

- Expected successful calls: about `236`
- Random success rate: `11.26%`

### Business Conclusion

The tuned logistic model would produce about `2.7x` as many successful calls as a random-calling strategy for the same number of calls.  
This is the main business value of the project: the model turns broad outreach into a more targeted campaign.

At the same time, the current threshold does not capture all potential subscribers. On the test set, the model reaches about `642` of roughly `928` actual subscribers, which means about `286` potential clients are still missed. That is why the model should be viewed as a ranking tool rather than a perfect yes/no rule.

## Findings and Actionable Insights

- The current tuned logistic regression model is a practical choice for campaign support because it improves expected successful calls substantially relative to random calling.
- The model should be used to rank customers by predicted likelihood of success, rather than as a rigid yes/no decision rule.
- If the bank wants to reduce wasted calls, it should contact the highest-ranked customers first.
- If the bank wants to capture more subscribers, it should lower the calling threshold and accept a lower hit rate.
- The current setup still misses about `286` potential subscribers on the test set, so campaign managers should treat the model as a prioritization tool, not a complete filter.
- The final operating threshold should be chosen only after comparing the cost of each call with the expected profit from a successful subscription.

## Ranking and Campaign Economics

The model produces a probability score for each customer, so customers can be ranked from most likely to least likely to subscribe. In practice, this is often more useful than treating the output as a simple binary decision.

This ranking can support several business strategies:

- call only the top-ranked customers if the campaign budget is tight
- lower the cutoff and call more customers if the bank wants to capture more subscribers
- compare different cutoffs based on expected number of successes, total call volume, and cost per campaign

The best model and threshold therefore depend on campaign economics:

- if the cost of each call is high relative to expected profit, the bank should prefer a stricter cutoff and higher precision
- if the profit from a successful subscription is high enough, it may be worth calling more customers and accepting lower precision in exchange for higher recall
- the most business-relevant metric is ultimately expected profit, not any single ML metric by itself

In other words, the current recall-focused model is sensible when the bank wants to find more subscribers, but the final operating threshold should be chosen only after comparing call cost against the expected value of a successful term-deposit sale.

## Reduced-Feature Test

The notebook also tested dropping the low-signal features:

- `loan`
- `housing`
- `day_of_week`
- `marital`
- `education`
- `campaign`

Impact in the current notebook:

- Change in test accuracy: `-0.012`
- Change in test precision: `-0.013`
- Change in test recall: `+0.004`

### Interpretation

Removing these features slightly improved recall, but reduced accuracy and precision.  
That suggests the low-signal variables are not completely useless, and dropping them does not clearly improve the model overall. If the business strongly prioritizes recall, the reduced version may still be worth considering, but the evidence does not show a decisive gain.

## Final Recommendation

For business use, the tuned logistic regression model is the strongest choice in the current notebook because:

- it captures a large share of likely subscribers
- it is easier to explain than KNN or SVM
- it clearly outperforms random calling in expected successful calls
- it gives a probability-based ranking that can be matched to campaign budget and business constraints

## Next Steps

1. Test several probability cutoffs and compare how many calls, successes, and missed subscribers each cutoff produces.
2. Add a profit-based decision rule that weighs call cost against the value of a successful subscription.
3. Validate the selected model on a future time period to confirm that performance is stable over time.
4. Re-run the reduced-feature experiment if model simplicity becomes more important than a small drop in precision.

## How to Reproduce

1. Open `prompt_III.ipynb`.
2. Run the notebook top to bottom.
3. Review:
   - EDA conclusions
   - baseline and default model comparison
   - grid search results
   - confusion matrices
   - successful-calls business summary
