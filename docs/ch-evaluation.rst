.. contents::
    :local:
    :depth: 3

Evaluation in Machine Learning
==============================

* **Offline evaluation**: offline evaluation is done to evaluate the performance of machine learning models. Offline metrics measure how close the predictions by the machine learning model on the evaluation dataset are to the graound truth dataset. 
* **Online evaluation**:

.. list-table:: Offline evaluation metrics
   :widths: 25 25
   :header-rows: 1

   * - Task
     - Metrics
   * - Classification
     - Precision, recall, F1 score, accuracy
   * - Regression
     - Mean squared error (MSE), root mean squared error (RMSE), MAE
   * - Ranking
     - Precision@k, recall@k, mean average precision (mAP), normalized discounted cumulative gain(nDCG), mean reciprocal rank (MRR)
   * - NLP
     - BLEU, ROUGE, METEOR, CIDEx, SPICE


.. list-table:: Online evaluation metrics
   :widths: 25 25
   :header-rows: 1

   * - Problem
     - Metrics
   * - Ad click prediction
     - Click-through rate, revenue lift.
   * - Harmful content detection
     - Prevalence, valid appeals.
   * - Video recommendation
     - Click-through rate, total watch time, number of completed videos.
   * - Friend recommendation
     - Number of requests sent per day, number of requests accepted per day.


.. _eval_precision:

Precision
---------
  In the context of binary classification (Yes/No), precision measures the model's performance at classifying positive observations (i.e. "Yes"). In other words, when a positive value is predicted, how often is the prediction correct? We could game this metric by only returning positive for the single observation we are most confident in.

  .. math::

    P = \frac{True Positives}{True Positives + False Positives}

.. _eval_recall:

Recall
---------
  Also called sensitivity. In the context of binary classification (Yes/No), recall measures how "sensitive" the classifier is at detecting positive instances. In other words, for all the true observations in our sample, how many did we "catch." We could game this metric by always classifying observations as positive.

  .. math::

    R = \frac{True Positives}{True Positives + False Negatives}

.. _eval_recall_vs_precision:

Recall vs Precision
-------------------
  Say we are analyzing Brain scans and trying to predict whether a person has a tumor (True) or not (False). We feed it into our model and our model starts guessing.

  - **Precision** is the % of True guesses that were actually correct! If we guess 1 image is True out of 100 images and that image is actually True, then our precision is 100%! Our results aren't helpful however because we missed 10 brain tumors! We were super precise when we tried, but we didn’t try hard enough.

  - **Recall**, or Sensitivity, provides another lens which with to view how good our model is. Again let’s say there are 100 images, 10 with brain tumors, and we correctly guessed 1 had a brain tumor. Precision is 100%, but recall is 10%. Perfect recall requires that we catch all 10 tumors!
