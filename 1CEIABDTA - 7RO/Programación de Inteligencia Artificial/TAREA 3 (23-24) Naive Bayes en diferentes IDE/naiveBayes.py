import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

# Cargar los datos
vote_data = pd.read_csv("house-votes-84.data", header=None, na_values="?")
header = ["NAME", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16"]
vote_data.columns = header

# Reemplazar valores faltantes
vote_data.fillna("", inplace=True)

# Convertir todas las columnas categóricas a numéricas
le = LabelEncoder()
for column in vote_data.columns:
    vote_data[column] = le.fit_transform(vote_data[column].astype(str))

# Dividir los datos en entrenamiento y prueba
vote_raw_train, vote_raw_test = train_test_split(vote_data, test_size=0.15, random_state=42)

# Entrenar el clasificador Naive Bayes
vote_classifier = MultinomialNB(alpha=1.0)
vote_classifier.fit(vote_raw_train.drop(columns=['NAME']), vote_raw_train['NAME'])

# Predecir la clase más probable
vote_test_pred = vote_classifier.predict(vote_raw_test.drop(columns=['NAME']))

# Matriz de confusión y reporte de clasificación
print(confusion_matrix(vote_raw_test['NAME'], vote_test_pred))
print(classification_report(vote_raw_test['NAME'], vote_test_pred))

# Predicciones donde se ha equivocado el modelo
errores = vote_raw_test[vote_raw_test['NAME'] != vote_test_pred]
print(errores)

# Predicciones en formato raw
vote_test_pred_proba = vote_classifier.predict_proba(vote_raw_test.drop(columns=['NAME']))
pred = pd.DataFrame(vote_test_pred_proba, columns=le.classes_[:vote_test_pred_proba.shape[1]])

# Curvas ROC, Precisión-Recall y Sensibilidad-Especificidad
fpr, tpr, _ = roc_curve(vote_raw_test['NAME'], vote_test_pred_proba[:, 1], pos_label=1)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

precision, recall, _ = precision_recall_curve(vote_raw_test['NAME'], vote_test_pred_proba[:, 1], pos_label=1)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Sensibilidad y Especificidad
plt.plot(tpr, 1-fpr)
plt.xlabel('Sensitivity')
plt.ylabel('1 - Specificity')
plt.title('Sensitivity-Specificity Curve')
plt.show()
