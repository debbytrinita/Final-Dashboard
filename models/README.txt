Top 3 model churn menurut CV roc_auc

1. Logistic Regression                  CV roc_auc 0.7998  -> churn_model_logreg.joblib
2. Logistic Regression + Oversampling   CV roc_auc 0.7990  -> churn_model_logreg_ros.joblib
3. Gradient Boosting                    CV roc_auc 0.7923  -> churn_model.joblib

Cara memuat (scikit-learn==1.9.1; model oversampling juga butuh imbalanced-learn):
    import joblib, json, pandas as pd
    model = joblib.load('<file>.joblib')
    meta = json.load(open('model_metadata.json', encoding='utf-8'))
    proba = model.predict_proba(df[meta['feature_order']])[:, 1]

Hanya muat file .joblib dari sumber tepercaya (joblib memakai pickle).
