# -*- coding: utf-8 -*-
"""
Переобучение LSTM для МЕЖЛАМЕЛЬНОГО НАПРЯЖЕНИЯ - оптимизированная модель
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==================== ФУНКЦИИ РАСЧЁТА ====================
def calculate_interlamella_nb514e(I1, Ib, Ud1):
    """
    Расчёт межламельного напряжения для NB-514E
    Формула: ((2*3*Ud1)/(0.7*348)) * ((Ib/I1)*9+0.7*9.67-8+1/8*0.7*9.67)/((Ib/I1)*9)
    """
    try:
        if pd.isna(I1) or pd.isna(Ib) or pd.isna(Ud1):
            return 0
        if I1 == 0 or Ib == 0:
            return 0
        
        ratio = Ib / I1
        numerator = (2 * 3 * Ud1) / (0.7 * 348)
        multiplier_num = (ratio * 9 + 0.7 * 9.67 - 8 + (1/8) * 0.7 * 9.67)
        multiplier_den = ratio * 9
        
        if multiplier_den == 0:
            return 0
        
        multiplier = multiplier_num / multiplier_den
        U_inter = numerator * multiplier
        
        return float(U_inter) if not np.isnan(U_inter) and not np.isinf(U_inter) else 0
    except:
        return 0

# ==================== ЗАГРУЗКА ДАННЫХ ====================
print("="*70)
print("🚂 ПЕРЕОБУЧЕНИЕ LSTM ДЛЯ МЕЖЛАМЕЛЬНОГО НАПРЯЖЕНИЯ")
print("="*70)
print("\n📁 Загрузка данных...")
df = pd.read_excel('Big-Data-MSUD.xlsx')
print(f"✅ Загружено {len(df):,} записей")

# Расчёт межламельного
print("🔧 Расчёт межламельного напряжения...")
df['Interlamella'] = df.apply(
    lambda row: calculate_interlamella_nb514e(row.get('I1[1]', 0), row.get('Ib[1]', 0), row.get('Ud1[1]', 0)), 
    axis=1
)

# Удаление NaN и inf
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=['Interlamella'])

print(f"\n📊 Статистика межламельного напряжения:")
print(f"   Среднее: {df['Interlamella'].mean():.4f} В")
print(f"   Мин: {df['Interlamella'].min():.4f} В")
print(f"   Макс: {df['Interlamella'].max():.4f} В")
print(f"   Станд. откл.: {df['Interlamella'].std():.4f} В")

# ==================== ПОДГОТОВКА ДАННЫХ ====================
def create_sequences(data, lookback=30):
    """Создание последовательностей для LSTM"""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Используем только межламельное
data = df['Interlamella'].values.reshape(-1, 1)

print("\n🔄 Нормализация данных...")
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Создание последовательностей
lookback = 30
print(f"📦 Создание последовательностей (lookback={lookback})...")
X, y = create_sequences(data_scaled, lookback)

print(f"   Форма X: {X.shape}")
print(f"   Форма y: {y.shape}")

# Разделение на train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

print(f"\n📊 Разделение данных:")
print(f"   Train: {len(X_train)} записей ({len(X_train)/(len(X_train)+len(X_test))*100:.1f}%)")
print(f"   Test:  {len(X_test)} записей ({len(X_test)/(len(X_train)+len(X_test))*100:.1f}%)")

# ==================== СОЗДАНИЕ ОПТИМИЗИРОВАННОЙ МОДЕЛИ ====================
print("\n🧠 Создание оптимизированной LSTM модели для межламельного...")

model = Sequential([
    # Первый слой LSTM
    LSTM(256, return_sequences=True, input_shape=(lookback, 1)),
    BatchNormalization(),
    Dropout(0.2),
    
    # Второй слой LSTM
    LSTM(128, return_sequences=True),
    BatchNormalization(),
    Dropout(0.2),
    
    # Третий слой LSTM
    LSTM(64, return_sequences=True),
    BatchNormalization(),
    Dropout(0.2),
    
    # Четвёртый слой LSTM
    LSTM(32, return_sequences=False),
    BatchNormalization(),
    Dropout(0.1),
    
    # Полносвязные слои
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("\n📋 Архитектура модели:")
model.summary()

# ==================== ОБУЧЕНИЕ ====================
print("\n🚀 Начало обучения (может занять ~5-15 минут)...")

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=25,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=12,
    min_lr=0.00001,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    'lstm_inter_best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=0
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=200,
    batch_size=16,
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    verbose=1
)

# ==================== ОЦЕНКА ====================
print("\n📈 Оценка модели на тестовых данных:")
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"   Test Loss (MSE): {test_loss:.8f}")
print(f"   Test MAE: {test_mae:.8f}")

# Прогнозирование
y_pred = model.predict(X_test, verbose=0)

# Денормализация
y_test_denorm = scaler.inverse_transform(y_test)
y_pred_denorm = scaler.inverse_transform(y_pred)

# Метрики точности
mae = np.mean(np.abs(y_test_denorm - y_pred_denorm))
rmse = np.sqrt(np.mean((y_test_denorm - y_pred_denorm)**2))
mape = np.mean(np.abs((y_test_denorm - y_pred_denorm) / (y_test_denorm + 1e-8))) * 100

print(f"\n✅ Результаты на реальных значениях межламельного:")
print(f"   MAE:  {mae:.6f} В")
print(f"   RMSE: {rmse:.6f} В")
print(f"   MAPE: {mape:.2f}%")

# Анализ распределения ошибок
errors = np.abs(y_test_denorm.flatten() - y_pred_denorm.flatten())
print(f"\n📊 Распределение ошибок:")
print(f"   Средняя ошибка: {np.mean(errors):.6f} В")
print(f"   Макс ошибка: {np.max(errors):.6f} В")
print(f"   90-й перцентиль: {np.percentile(errors, 90):.6f} В")

# ==================== СОХРАНЕНИЕ ====================
print("\n💾 Сохранение модели и скалера...")

model.save('lstm_inter_model.keras')
joblib.dump(scaler, 'lstm_inter_scaler.pkl')

print("\n✅ Готово!")
print(f"   📁 Модель сохранена: lstm_inter_model.keras")
print(f"   📁 Скалер сохранён: lstm_inter_scaler.pkl")

# ==================== ТЕСТ ПРОГНОЗА ====================
print("\n🧪 Тест прогноза на последних данных:")

# Берём последние 30 точек
test_sequence = data_scaled[-lookback:].reshape(1, lookback, 1)

# Прогноз на 20 шагов вперёд
forecast_steps = 20
forecast = []

current_seq = test_sequence.copy()
for i in range(forecast_steps):
    next_pred = model.predict(current_seq, verbose=0)[0, 0]
    forecast.append(next_pred)
    current_seq = np.append(current_seq[:, 1:, :], [[[next_pred]]], axis=1)

# Денормализация прогноза
forecast = np.array(forecast).reshape(-1, 1)
forecast_denorm = scaler.inverse_transform(forecast)

print(f"\n   Последние 5 реальных значений межламельного:")
for i, val in enumerate(data[-5:].flatten()):
    print(f"      {i+1}. {val:.6f} В")

print(f"\n   Прогноз на следующие {forecast_steps} шагов:")
for i, val in enumerate(forecast_denorm.flatten()):
    print(f"      {i+1}. {val:.6f} В")

print("\n" + "="*70)
print("🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
print("="*70)
print("\n📝 ИНСТРУКЦИЯ ПО ПРИМЕНЕНИЮ:")
print("   1. Замените файлы в C:\\ted_app:")
print("      - lstm_interlamella_model.keras → lstm_inter_model.keras")
print("      - lstm_interlamella_scaler.pkl → lstm_inter_scaler.pkl")
print("   2. Обновите app.py с новыми именами моделей")
