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

    return render_template_string(TEMPLATE, result=result, table=table, plot=plot)


TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Student Spending AI</title>

    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">

    <style>
        body {
            background: #0b0f1a;
            color: white;
        }

        .card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 15px;
            backdrop-filter: blur(10px);
            margin-bottom: 20px;
            color: white;
        }

        .form-control {
            background-color: rgba(255,255,255,0.08) !important;
            color: white !important;
            border: 1px solid rgba(255,255,255,0.2) !important;
        }

        .form-control:focus {
            background-color: rgba(255,255,255,0.12) !important;
            color: white !important;
            box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25);
        }

        .form-control::placeholder {
            color: rgba(255,255,255,0.5) !important;
        }

        label {
            font-size: 12px;
            opacity: 0.8;
            color: rgba(255,255,255,0.8);
        }

        .table {
            color: white !important;
        }
    </style>
</head>

<body>

<div class="container py-4">

    <h1 class="text-center mb-4">🎓 Student Spending Clustering AI</h1>

    <div class="card p-4">

        <form method="POST">

            {% set labels = {
            'monthly_income': '💰 Monthly Income',
            'financial_aid': '🎓 Financial Aid',
            'tuition': '🏫 Tuition',
            'housing': '🏠 Housing',
            'food': '🍔 Food',
            'transportation': '🚗 Transport',
            'books_supplies': '📚 Books',
            'entertainment': '🎮 Entertainment',
            'personal_care': '🧴 Care',
            'technology': '💻 Tech',
            'health_wellness': '💊 Health',
            'miscellaneous': '📦 Other'
            } %}

            <div class="row g-2">

            {% for field in labels %}
                <div class="col-md-3">
                    <label>{{ labels[field] }}</label>
                    <input type="number" step="any" name="{{field}}" class="form-control" required>
                </div>
            {% endfor %}

            </div>

            <button class="btn btn-info w-100 mt-3">Predict Cluster</button>
        </form>

        {% if result %}
        <div class="alert alert-success mt-3">
            {{ result }}
        </div>
        {% endif %}

    </div>

    <div class="card p-4">
        <h5>📊 2D Clusters (PCA)</h5>
        {{ plot|safe }}
    </div>

    <div class="card p-4">
        <h5>📋 Dataset</h5>
        <div class="table-responsive">
            {{ table|safe }}
        </div>
    </div>

</div>

</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True)