# Understand the project

Imagine running a subscription music service. You can contact only 1,000 of 10,000
customers this month. Which customers should you contact, what evidence makes them
worth investigating, and what action could you offer?

## 1. Turn history into features
Transactions and listening logs have many rows per customer. Summarise them into
one row at a chosen prediction date: days listened recently, trend versus a prior
period, subscription history and known renewal settings.
These summaries are features. Only use data available at that moment.
Data after that moment would let the model peek into the future: leakage.

## 2. Define the answer you want to predict
A label says whether the customer later met the agreed churn definition.
The plan uses failure to renew within 30 days after expiry, subject to verifying
KKBox's actual rules. That means you must wait for the outcome window to close.
A label from the future cannot be used to train a model at an earlier historical date.

## 3. Learn patterns
Logistic regression is the first baseline. LightGBM then learns nonlinear patterns
and feature interactions. Both map features to churn probabilities.
An LLM is not needed for this step.
Fit preprocessing on training data only and keep the final test set untouched.

## 4. Check whether predictions are useful
Use earlier history to train and later eligible cohorts to evaluate, allowing labels
time to mature. Report ranking and probability quality and how many actual churners
appear in your top-budget group. Compare against simple baselines.
Repeated customers across time are not automatically leakage, but customer identifiers
must not become predictor shortcuts; document the deployment population.

## 5. Explain the prediction
SHAP allocates the model output among input features relative to a reference.
It can show that falling listening activity pushed the prediction upwards.
It cannot prove why the customer is leaving or which intervention will work.

## 6. Support a retention decision
An LLM takes the supplied profile, prediction drivers and approved actions and drafts
an action plus outreach text. Check its evidence, format and invented claims.
A human reviews the recommendation. The app does not automatically contact customers.

## 7. Estimate the economics
For a simple illustrative scenario, expected incremental value per contacted customer
could be p(churn) × assumed probability of preventing that churn × value retained
minus expected intervention cost. Define the value horizon and when costs are incurred.
This assumes an intervention effect the dataset does not measure. Run sensitivity
scenarios and compare targeting with repeated random samples at the same budget.
High churn risk is not the same as high responsiveness to an offer.

## Worked example (entirely fictional)
Customer demo-001 listened on 3 of the last 30 days versus 12 in the previous period.
A future model might estimate 72% risk, with declining listening a positive SHAP driver.
An approved action might be a music-discovery message. The assistant drafts it without
claiming to know why the customer has been listening less.
With an assumed 10% conditional rescue rate, 40 units of retained value and 1 unit of
cost, illustrative expected net value is 0.72 × 0.10 × 40 − 1 = 1.88 units.
None of these example numbers is a measured project result.

## What each teammate learns
Data owner: temporal data engineering, quality checks and business assumptions.
Model owner: baselines, evaluation, probabilities and model interpretation.
App owner: interfaces, grounded generation, evaluation and deployment.
All three should explain the full journey and review each other's work.

Read demo.py, run the CLI, then open the app. Next work through the first three backlog
tasks together before splitting implementation.
