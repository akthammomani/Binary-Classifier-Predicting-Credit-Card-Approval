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

