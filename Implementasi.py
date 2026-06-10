import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Membaca dataset
data = pd.read_csv('mahasiswa.csv')

# Fitur
X = data[['IPK','Kehadiran','SKS_Lulus','Organisasi']]

# Target
y = data['Status']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Membuat model
model = DecisionTreeClassifier()

# Training
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Akurasi
accuracy = accuracy_score(y_test, y_pred)

print("Akurasi:", accuracy)