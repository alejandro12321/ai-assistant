import cv2
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Preprocesar imágenes del dataset
def load_images_from_folders(base_folder):
    images = []
    labels = []
    for label, folder_name in enumerate(['reconocidos', 'noReconocidos']):
        folder_path = os.path.join(base_folder, folder_name)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            image = cv2.imread(img_path)
            if image is not None:
                image = cv2.resize(image, (128, 128))
                images.append(image)
                labels.append(label)  # 0 para no reconocidos y 1 para reconocidos
    return np.array(images, dtype='float32') / 255.0, np.array(labels)

# Ruta a tu dataset
data_path = 'dataSetRose'
images, labels = load_images_from_folders(data_path)

# Crear el modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Añadido Dropout para evitar sobreajuste
    Dense(1, activation='sigmoid')  # Clasificación binaria
])

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Dividir los datos en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_val, y_val))

# Función para preprocesar imágenes capturadas por la cámara
def preprocess_image(image):
    image = cv2.resize(image, (128, 128))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Capturar y reconocer rostros en tiempo real
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        image = preprocess_image(roi)
        prediction = model.predict(image)
        
        # Ajusta el umbral si es necesario
        threshold = 0.9  # Aumenta el umbral para mayor precisión
        if prediction[0] > threshold:
            text = "Rostro reconocido"
        else:
            text = "Rostro no reconocido"
        
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
