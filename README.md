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

