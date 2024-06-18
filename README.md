# Pancreatic-Cancer-prediction-models

BACKGROUND

With a 5-year survival rate of only 12%, pancreatic cancer is one of the deadliest diseases worldwide.

Early diagnosis is vital for patient survival given the absence of effective therapies and diagnostic tests.

The prospect of detecting pancreatic cancer through non-invasive diagnostic approaches, such as blood or urine tests to measure cancer-specific biomarkers, is often regarded as the "holy grail" of cancer diagnostics. However, this area of research faces significant challenges due to the complexity of cancer types, patient heterogeneity and lack of biomarker specificity.

Achieving high accuracy in diagnostic tests remains a major obstacle in this endeavor.

OBJECTIVE

In this notebook I have applied some classical maschine learning models to a clinical dataset to predict pancreatic cancer based on clinical data.

THE DATA

The dataset is very well balanced as it includes 103 healthy controls, 208 non-cancer conditions and 199 cancer cases. It includes features such as patient demographics (sex and age) and levels of protein biomarkers (creatinine, LYVE1, REG1B, and TFF1). It is important to note that features directly related to cancer, such as the stage of progression, are excluded from the prediction model.

FINAL REMARKS AND CONCLUSIONS

* Multiclass classification shows modest accuracy (over 70%) across different models, including logistic regression, SVM, and XGBoost. Although the classes were well-balanced, the number of observations was relatively small (around 200 per class). A larger dataset would likely result in improved performance.
* Simplifying the classification to a binary model increased the accuracy up to 90% and yielded an excellent ROC curve (AUC of 0.94).
* I analyzed feature importance using XGBoost Gain, which demonstrated that the feature CA19.9 significantly improves model accuracy when used for splits. For this reason, removing the CA19.9 column would likely be detrimental. Since the distribution of CA19.9 is not normal, NAs were replaced with the median. This choice could be further debated in future analyses.


REFERENCES

* https://www.kaggle.com/datasets/johnjdavisiv/urinary-biomarkers-for-pancreatic-cancer

* https://www.hopkinsmedicine.org/health/conditions-and-diseases/pancreatic-cancer/pancreatic-cancer-prognosis

* O'Neill et al. "Biomarkers in the diagnosis of pancreatic cancer: Are we closer to finding the golden ticket?" World J Gastroenterol. 2021 Jul 14;27(26):4045-4087. doi: 10.3748/wjg.v27.i26.4045.


**ACKNOWLEDGEMENTS**

I would like to express my gratitude to Bernardi and colleagues for their generous data sharing, which promotes the principles of open science.
