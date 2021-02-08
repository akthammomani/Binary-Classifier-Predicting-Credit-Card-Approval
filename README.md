# Predicting Credit Card Approval Using Binary Classifiers

## 1. Introduction

Commercial banks receive a lot of applications for credit cards. Many of them get rejected for many reasons, like high loan balances, low income levels, or too many inquiries on an individual's credit report, for example. Manually analyzing these applications is mundane, error-prone, and time-consuming (and time is money!). Luckily, this task can be automated with the power of machine learning and pretty much every commercial bank does so nowadays. In this notebook, we will build an automatic credit card approval predictor using machine learning techniques, just like the real banks do.


<p align="center">
  <img width="750" height="500" src="https://user-images.githubusercontent.com/67468718/106730102-10cf9600-65c3-11eb-91d4-bc15e9b37ed9.jpeg">
</p>

## 2. Dataset

We'll be using the Credit Card Approval dataset from the UCI Machine Learning Repository [location](http://archive.ics.uci.edu/ml/datasets/credit+approval).

This file concerns credit card applications. All attribute names and values have been changed to meaningless symbols to protect confidentiality of the data.

Furthermore, this dataset is interesting because there is a good mix of attributes: continuous, nominal with small numbers of values, and nominal with larger numbers of values. There are also a few missing values.


## 3. Objective

The structure of this notebook is as follows:

 * First, we will start off by loading and viewing the dataset.
 * We will see that the dataset has a mixture of both numerical and categorical features, that it contains values from different ranges, plus that it contains a number of missing entries.
We will have to preprocess the dataset to ensure the machine learning model we choose can make good predictions.
 * After our data is in good shape, we will do some exploratory data analysis to build our intuitions.
 * Finally, we will build a machine learning model that can predict if an individual's application for a credit card will be accepted.
 
 
 ## 4. Inspecting the applications

<p>The output may appear a bit confusing at its first sight, but let's try to figure out the most important features of a credit card application. The features of this dataset have been anonymized to protect the privacy, but <a href="http://rstudio-pubs-static.s3.amazonaws.com/73039_9946de135c0a49daa7a0a9eda4a67a72.html">this blog</a> gives us a pretty good overview of the probable features. The probable features in a typical credit card application are <code>Gender</code>, <code>Age</code>, <code>Debt</code>, <code>Married</code>, <code>BankCustomer</code>, <code>EducationLevel</code>, <code>Ethnicity</code>, <code>YearsEmployed</code>, <code>PriorDefault</code>, <code>Employed</code>, <code>CreditScore</code>, <code>DriversLicense</code>, <code>Citizen</code>, <code>ZipCode</code>, <code>Income</code> and finally the <code>ApprovalStatus</code>. This gives us a pretty good starting point, and we can map these features with respect to the columns in the output.   </p>
<p>As we can see from our first glance at the data, the dataset has a mixture of numerical and categorical features. This can be fixed with some preprocessing, but before we do that, let's learn about the dataset a bit more to see if there are other dataset issues that need to be fixed.</p>

## 5. Handling Missing Values

  <p>We've uncovered some issues that will affect the performance of our machine learning model(s) if they go unchanged:</p>
<ul>
<li>Our dataset contains both numerical and categorical data (specifically data that are of <code>float64</code>, <code>int64</code> and <code>object</code> types). Specifically, the features:'dept', 'years_employed', 'CreditScore' and 'income' contain numeric values (of types float64, float64, int64 and int64 respectively) and all the other features contain categorical features values.</li>
<li>The dataset also contains values from several ranges. Some features have a value range of 0 - 28, some have a range of 2 - 67, and some have a range of 1017 - 100000. Apart from these, we can get useful statistical information (like <code>mean</code>, <code>max</code>, and <code>min</code>) about the features that have numerical values. </li>
<li>Two features defined as an object and those need to be converted to float64: 'age' & 'zipcode'.    
<li>Finally, the dataset has missing values, which we'll take care of next. The missing values in the dataset are labeled with '?', which can be seen in the most categorical features. So temporarily we'll replace these missing value question marks with NaN.</li>
</ul>


## 6. Preprocessing the data

<p>The missing values are now successfully handled.</p>
<p>There is still some minor but essential data preprocessing needed before we proceed towards building our machine learning model. We are going to divide these remaining preprocessing steps into three main tasks:</p>
<ol>
<li>Convert the non-numeric data into numeric.</li>
<li>Split the data into train and test sets. </li>
<li>Scale the feature values to a uniform range.</li>
</ol>
<p>First, we will be converting all the categorical values into numeric ones. We do this because not only it results in a faster computation but also many machine learning models (like XGBoost) (and especially the ones developed using scikit-learn) require the data to be in a strictly numeric format. We will do this by using a technique called <a href="http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html">label encoding</a>.</p>


## 7. Modelling

Essentially, predicting if a credit card application will be approved or not is a binary-classification task. According to UCI, our dataset contains more instances that correspond to "Denied" status than instances corresponding to "Approved" status. Specifically, out of 690 instances, there are 383 (55.5%) applications that got denied and 307 (44.5%) applications that got approved.

This gives us a benchmark. A good machine learning model should be able to accurately predict the status of the applications with respect to these statistics.

Which model should we pick? A question to ask is: are the features that affect the credit card approval decision process correlated with each other? Although we can measure correlation, that is outside the scope of this notebook, so we'll rely on our intuition that they indeed are correlated for now, so let's explore below classifiers:

 * Logistic Regression model.
 * Decision Tree - Entropy - No Max depth.
 * Decision Tree - Gini - max_depth=5.
 * Random Forest.
 * Gradient Boosting.
 * Extreme Gradient Boosting.
 
## 8. Model Comparison and the Winner !!!

The Random Forests Model shows an overall **Accuracy 89.47% and AUC 94%**, which is great and indicates that our model outperforms all other Models and was effectively able to  predict if an individual's application for a credit card will be accepted or rejected.

[Single Decision Tree Visualization from our Random Forests Model](https://github.com/akthammomani/Binary-Classifier-Predicting-Credit-Card-Approval/blob/main/Single_DT_Visualization_RF.md)

### Accuracy vs. AUC for All Models:

| <code>Model</code>|<code>Accuracy</code>|<code>AUC</code>|
|:------------------|:------------------:|:------------------:|
|Logistic Regression model|87.28070 %    |90.92996 %    |
|Decision Tree - Entropy - No Max depth|83.77193 %    |82.93233 %    |
|Decision Tree - Gini - max_depth=5|86.40351 %    |91.89553 %    |
|<code>Random Forest - max_depth=5</code>|<code>89.47368 %   </code> | <code>94.09577 %  </code> |
|Gradient Boosting - max_depth=5|87.28070 %      | 92.79778 %   |
|Extreme Gradient Boosting| 87.28070 %    | 94.21053 %   |

### Confusion Matrix comparison between All Models:

| <code>Model</code>| <code>Metrics</code> | <code>value</code>  |<code>Metrics </code>  | <code>value</code>  | <code>Total</code>  |<code> value </code> |
|:-----------------|:-------:|:------:|:--------:|:------:|:------:|:------:|
|Logistic Regression model | TP |  116 | FN  | 17  |Total   | 133  |
|Logistic Regression model| FP |   12|  TN | 83  |Total   | 95  |   
|Decision Tree - Entropy - No Max depth| TP |  117 | FN  | 16  |Total   | 133  |
|Decision Tree - Entropy - No Max depth| FP |   21|  TN | 74  |Total   | 95|   
|Decision Tree - Gini - max_depth=5| TP |  112 | FN  | 21  |Total   | 133 |
|Decision Tree - Gini - max_depth=5| FP |   10|  TN | 85  |Total   | 95|   
|<code>Random Forest - max_depth=5</code>| <code>TP </code>|<code> 124</code>|<code> FN  </code>| <code>9 </code>|<code>Total </code>  | <code>133</code> |
|<code>Random Forest - max_depth=5</code>|<code> FP </code>|  <code> 15</code>| <code> TN </code>|<code>80 </code>|<code>Total   </code>| <code>95  </code>|   
|Gradient Boosting - max_depth=5| TP |  117 | FN  | 16  |Total   | 133|
|Gradient Boosting - max_depth=5| FP |   13|  TN | 82  |Total   | 95|  
|Extreme Gradient Boosting| TP |  117 | FN  | 16  |Total   | 133 |
|Extreme Gradient Boosting| FP |   13|  TN | 82  |Total   | 95  | 


| <code>Model</code>|<code>Accuracy</code>|<code>AUC</code>|
|:------------------|:------------------:|:------------------:|
|Logistic Regression model|87.28070 %    |90.92996 %    |
|Decision Tree - Entropy - max_depth=5|86.84211 %    |88.96715 %    |
|Decision Tree - Gini - max_depth=5|86.40351 %    |91.89553 %    |
|<code>Random Forest - max_depth=5</code>|<code>89.47368 %   </code> | <code>94.09577 %  </code> |
|Gradient Boosting - max_depth=5|88.15789 %      | 94.03245 %   |
|Extreme Gradient Boosting| 85.96491 %    | 93.56549 %   |



| <code>Model</code>|<code>Class 0 precision</code>|<code>Class 1 precision</code>|<code>Class 0 recall</code>|<code>Class 1 recall</code>|
|:------------------|:------------------:|:------------------:|:------------------:|:------------------:|
|Logistic Regression model|0.906250      |0.830000      | 0.872180  |0.873684  |
|Decision Tree - Entropy - max_depth=5| 0.881481       | 0.849462       | 0.894737     |0.831579       |
|Decision Tree - Gini - max_depth=5|  0.918033      | 0.801887       |  0.842105    |  0.894737     |
|<code>Random Forest - max_depth=5</code>|<code> 0.892086   </code>    | <code>0.898876  </code>     | <code>0.932331 </code>    |<code> 0.842105 </code>     |
|Gradient Boosting - max_depth=5| 0.901515       | 0.854167       | 0.894737     | 0.863158      |
|Extreme Gradient Boosting| 0.874074       | 0.838710       | 0.887218   |0.821053  |

  


