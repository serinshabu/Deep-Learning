# heart_disease_dnn.py
# Deep Neural Network for Heart Disease Prediction
# Imbalanced vs Balanced Dataset Comparison

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score


# -------------------------
# 1. Set Random Seed
# -------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def train_and_evaluate(csv_path, epochs=50, title="EXPERIMENT"):
    """
    Train and evaluate a DNN model on a given dataset
    """

    print("\n" + "=" * 60)
    print(f"{title}")
    print("=" * 60)

    # -------------------------
    # 2. Load Dataset
    # -------------------------
    df = pd.read_csv(csv_path)
    print("Dataset shape (before cleaning):", df.shape)

    # -------------------------
    # 3. Handle Missing Values
    # -------------------------
    df = df.dropna()
    print("Dataset shape (after cleaning):", df.shape)

    # -------------------------
    # 4. Features & Target
    # -------------------------
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # -------------------------
    # 5. Train-Test Split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )

    # -------------------------
    # 6. Feature Scaling
    # -------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------------------------
    # 7. Build Model
    # -------------------------
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # -------------------------
    # 8. Train Model
    # -------------------------
    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=32,
        verbose=1
    )

    # -------------------------
    # 9. Evaluate Model
    # -------------------------
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n--- Test Results ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    return model


def main():
    """
    Main execution function
    """

    # -------------------------
    # IMBALANCED DATASET
    # -------------------------
    train_and_evaluate(
        csv_path="framingham.csv",
        epochs=50,
        title="IMBALANCED DATASET"
    )

    # -------------------------
    # BALANCED DATASET
    # -------------------------
    train_and_evaluate(
        csv_path="framingham MERGED.csv",
        epochs=100,
        title="BALANCED DATASET"
    )


if __name__ == "__main__":
    main()