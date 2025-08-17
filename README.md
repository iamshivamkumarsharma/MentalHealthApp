# Mental Health Prediction Project 🧠

This project builds a machine learning model to predict whether an individual is likely to seek treatment for mental health issues based on survey responses. It demonstrates the **end-to-end ML lifecycle**, from preprocessing and model comparison to deployment with a user-friendly interface.

---

## Objective

The goal is to create an ML-based system that predicts mental health treatment likelihood using a **public survey dataset**.  
**Note:** This is for educational/demo purposes only — **not medical advice**.

---

## Dataset

* Source: `survey.csv` (employee workplace & personal survey)
* Target variable: `treatment` (Yes/No — whether the person sought treatment)
* Features: ~26 columns including demographics (Age, Gender, Country), workplace factors (remote work, company size, supervisor support), and perception-based responses (benefits, anonymity, wellness programs, etc.)

---

## Preprocessing

* **Imputation**: Missing values handled (`SimpleImputer`) — median for numeric, most frequent for categorical
* **Encoding**: Categorical variables encoded using `OrdinalEncoder` with `handle_unknown='use_encoded_value'`
* **ColumnTransformer**: Combines numeric + categorical preprocessing for consistent transformations

---

## Model Training

Multiple models were trained and compared:

* Logistic Regression  
* Decision Tree  
* Random Forest  
* Extra Trees  
* K-Nearest Neighbors  
* Gaussian Naive Bayes  
* Linear Discriminant Analysis  
* SVC (RBF kernel)  
* AdaBoost  
* LightGBM  

Each wrapped in a **scikit-learn pipeline** with preprocessing included.

---

## Evaluation

Metrics used:

* Accuracy  
* Precision (weighted)  
* Recall (weighted)  
* F1-score (weighted)  

**Results**:

* LightGBM: best overall accuracy  
* AdaBoost: strong recall/F1 performance  

---

## Deployment

* **Pipeline saved**: `mh_pipeline.pkl` (preprocessing + model) via `cloudpickle`  
* **Metadata saved**: `meta.json` (feature columns, target variable)  
* Streamlit app (`app.py`) collects user input, predicts treatment likelihood, and displays results with **ethical disclaimers**

---

## Technical Stack

* **Python**: Pandas, NumPy, scikit-learn, LightGBM, cloudpickle  
* **Streamlit**: UI deployment  
* **Git/GitHub**: Version control  
* **Virtual Environment (venv)**: Reproducibility  

---

## Key Learnings

* Building reusable ML pipelines (preprocessing + modeling)  
* Comparing multiple ML algorithms with various metrics  
* Handling real-world survey data (missing values, categorical encoding, imbalanced labels)  
* Deploying ML models with a user-friendly interface  
* Awareness of AI ethics in sensitive domains like mental health  

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/mental-health-prediction.git
cd mental-health-prediction



================================================FAQ=================================================
What kind of machine learning task is performed in this project?

This project performs a supervised classification task, predicting whether an individual has sought mental health treatment based on survey features.

Which feature(s) do you expect to be most predictive of treatment-seeking?

Features like family history, work interference, supervisor and coworkers support, and ease of leave tend to be most predictive due to their direct effect on attitudes towards seeking help.

How do you handle missing or inconsistent data in this survey-based project?

Typical approaches include imputation for numeric/categorical missing values, normalization of inconsistent category labels, and omitting rows with excessive missingness.

Why do you use multiple models, and how do you select the best one?

Using multiple models allows comparison and selection of the best-performing algorithm via validation metrics like accuracy, precision, recall, and F1 score. The model with the highest balanced performance is chosen.

How does workplace support affect the likelihood of seeking treatment according to the data?

Workplace support (supportive coworkers/supervisors, easy leave) positively correlates with seeking treatment; stigma and lack of support discourage it.

How do you validate the reliability of your model results?

Reliability is established by splitting the dataset into training/testing sets, using cross-validation, and comparing multiple metrics (accuracy, precision, recall, F1) across models.

What limitations do you see in using this dataset for real-world conclusions?

Limitations include bias and underreporting due to stigma, limited demographic representation, survey self-selection, and relatively small dataset size.

How might class imbalance affect your results, and how do you manage it?

If far more people answered one way, models could be biased; methods such as rebalancing (resampling), weighted loss functions, or evaluation with balanced metrics are used.

If you had access to additional features, what would improve the model further?

Additional features like clinical history, company policies, recent stressors, and access to external support resources could enhance model accuracy.

