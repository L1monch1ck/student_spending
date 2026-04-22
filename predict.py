import joblib
import numpy as np

kmeans = joblib.load('models/kmeans_model.pkl')
scaler = joblib.load('models/scaler.pkl')

print("=== Student Spending - K-Means Clustering ===\n")
print("Введите 12 значений через пробел:")
print(
    "monthly_income financial_aid tuition housing food transportation books_supplies entertainment personal_care technology health_wellness miscellaneous")

try:
    values = list(map(float, input().strip().split()))

    if len(values) != 12:
        print("❌ Ошибка: Нужно ввести ровно 12 чисел!")
    else:
        input_scaled = scaler.transform([values])
        cluster = kmeans.predict(input_scaled)[0]

        interpretations = {
            0: "Экономные студенты — низкие траты почти во всех категориях",
            1: "Студенты со средним уровнем расходов",
            2: "Студенты с высокими тратами на жильё и питание",
            3: "Студенты с высоким доходом и большими тратами на развлечения и технологии"
        }

        print(f"\n✅ Результат: Студент относится к **Кластеру {cluster}**")
        print(f"Описание: {interpretations[cluster]}")

except Exception as e:
    print(f"❌ Ошибка: {e}")