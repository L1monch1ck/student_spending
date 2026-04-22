import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import joblib
import os

print("🚀 Запуск обучения K-Means Clustering...")

df = pd.read_csv('data/student_spending.csv')
print(f"✅ Датасет успешно загружен. Размер: {df.shape}")

features = ['monthly_income', 'financial_aid', 'tuition', 'housing',
            'food', 'transportation', 'books_supplies', 'entertainment',
            'personal_care', 'technology', 'health_wellness', 'miscellaneous']

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Обучение модели K-Means с 4 кластерами...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_scaled)

os.makedirs('models', exist_ok=True)

# Сохранение модели и scaler
joblib.dump(kmeans, 'models/kmeans_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("✅ Модель K-Means успешно обучена и сохранена в папку 'models'!")

# Elbow Method для визуализации
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o', linewidth=2)
plt.title('Elbow Method - Определение оптимального количества кластеров')
plt.xlabel('Количество кластеров (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.savefig('elbow_method.png')
plt.close()

df['Cluster'] = kmeans.labels_
df.to_csv('data/student_spending_with_clusters.csv', index=False)

print("✅ Кластеры добавлены в файл 'data/student_spending_with_clusters.csv'")
print("\nРаспределение студентов по кластерам:")
print(df['Cluster'].value_counts().sort_index())