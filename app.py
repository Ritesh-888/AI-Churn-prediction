import os
from flask import Flask, render_template, request, send_file, redirect, url_for
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load model and feature columns
clf = joblib.load('model.pkl')
feature_columns = joblib.load('columns.pkl')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'

# Home & upload
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        f = request.files['file']
        df = pd.read_csv(f)
        df_proc = pd.get_dummies(df.drop(['customerID'], axis=1))
        # Align columns
        for col in feature_columns:
            if col not in df_proc:
                df_proc[col] = 0
        df_proc = df_proc[feature_columns]

        # Predict
        probs = clf.predict_proba(df_proc)[:, 1]
        df['churn_prob'] = probs

        # Plot distribution
        plt.figure()
        plt.hist(probs, bins=20)
        plt.title('Churn Probability Distribution')
        plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'prob_dist.png'))
        plt.close()

        # Pie chart churn vs retain
        churn_count = (probs >= 0.5).sum()
        retain_count = len(probs) - churn_count
        plt.figure()
        plt.pie([churn_count, retain_count], labels=['Churn','Retain'], autopct='%1.1f%%')
        plt.title('Churn vs Retain')
        plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'pie_chart.png'))
        plt.close()

        # Top 10 risk
        top10 = df.sort_values('churn_prob', ascending=False).head(10)
        top10_html = top10[['customerID', 'churn_prob']].to_html(index=False, classes='table table-striped')

        # Save predictions
        pred_path = os.path.join('static', 'predictions.csv')
        df.to_csv(pred_path, index=False)

        return render_template('index.html',
                               prob_img=url_for('static', filename='images/prob_dist.png'),
                               pie_img=url_for('static', filename='images/pie_chart.png'),
                               top10_table=top10_html,
                               download_link=url_for('static', filename='predictions.csv'))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)