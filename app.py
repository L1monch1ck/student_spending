from flask import Flask, request, render_template_string
import pandas as pd
import joblib
import os
import plotly.express as px
from sklearn.decomposition import PCA

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(BASE_DIR, 'data', 'student_spending.csv')
model_path = os.path.join(BASE_DIR, 'models', 'kmeans_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

df = pd.read_csv(data_path)
kmeans = joblib.load(model_path)
scaler = joblib.load(scaler_path)

features = [
    'monthly_income','financial_aid','tuition','housing',
    'food','transportation','books_supplies','entertainment',
    'personal_care','technology','health_wellness','miscellaneous'
]

X_scaled = scaler.transform(df[features])

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]

df['Cluster'] = kmeans.labels_

DESCRIPTIONS = {
    0: "Экономные студенты",
    1: "Средний уровень расходов",
    2: "Высокие расходы на базовые нужды",
    3: "High-income lifestyle студенты"
}

def generate_plot(user_point=None):
    fig = px.scatter(
        df,
        x='PC1',
        y='PC2',
        color='Cluster',
        title='2D Student Spending Clusters (PCA)',
        hover_data=features
    )

    if user_point:
        scaled = scaler.transform([user_point])
        p = pca.transform(scaled)

        fig.add_scatter(
            x=[p[0][0]],
            y=[p[0][1]],
            mode='markers',
            marker=dict(size=14, color='red', symbol='x'),
            name='YOU'
        )

    fig.update_layout(
        paper_bgcolor='#0b0f1a',
        plot_bgcolor='#0b0f1a',
        font=dict(color='white')
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')


@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    user_point = None

    if request.method == 'POST':
        values = [float(request.form[f]) for f in features]

        cluster = int(kmeans.predict(scaler.transform([values]))[0])
        result = f"Cluster {cluster}: {DESCRIPTIONS[cluster]}"

        user_point = values

    table = df.head(10).to_html(classes='table table-dark table-striped', index=False)
    plot = generate_plot(user_point)

    return render_template_string("index.html", result=result, table=table, plot=plot)


if __name__ == '__main__':
    app.run(debug=True)