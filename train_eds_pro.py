import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚂 ПЕРЕОБУЧЕНИЕ LSTM ДЛЯ РЕАКТИВНОЙ ЭДС (ПО ФОРМУЛЕ)")
print("="*70)

# 1. ЗАГРУЗКА ДАННЫХ
print("\n📁 Загрузка данных...")
try:
    df = pd.read_excel('Big-Data-MSUD.xlsx')
    print(f"✅ Загружено {len(df)} записей")
except Exception as e:
    print(f"❌ Ошибка загрузки: {e}")
    exit()

# 2. РАССЧЁТ РЕАКТИВНОЙ ЭДС ПО ФОРМУЛЕ
print("\n⚡ Рассчёт реактивной ЭДС по физической формуле...")

# Извлекаем нужные колонки
try:
    I1 = df['I1[1]'].values     # Намагничивающий ток
except KeyError as e:
    print(f"❌ Колонка не найдена: {e}")
    print(f"Доступные колонки: {df.columns.tolist()}")
    exit()

# Формула расчёта реактивной ЭДС (ваша формула):
# e_p = (8*(I1[1]/6)*4*1*37.8*10^-7*0.4*26.9)/(0.00522*9.79)
eds_calculated = (8*(I1/6)*4*1*37.8*1e-7*0.4*26.9)/(0.00522*9.79)

print(f"✅ Рассчитано {len(eds_calculated)} значений ЭДС")
print(f"   Min: {eds_calculated.min():.4f}")
print(f"   Max: {eds_calculated.max():.4f}")
print(f"   Mean: {eds_calculated.mean():.4f}")

# 3. НОРМАЛИЗАЦИЯ
print("\n🔧 Нормализация данных...")
scaler = MinMaxScaler()
eds_scaled = scaler.fit_transform(eds_calculated.reshape(-1, 1))
print(f"✅ Данные нормализованы в диапазон [0, 1]")

# 4. СОЗДАНИЕ ПОСЛЕДОВАТЕЛЬНОСТЕЙ
print("\n📊 Создание временных последовательностей...")
lookback = 30  # 30 шагов = 3 минуты реального времени (6 сек * 30)
X, y = [], []

for i in range(len(eds_scaled) - lookback):
    X.append(eds_scaled[i:i+lookback, 0])
    y.append(eds_scaled[i+lookback, 0])

X = np.array(X)
y = np.array(y)

# Разделение на обучающий и валидационный наборы
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"✅ Обучающих примеров: {len(X_train)}")
print(f"✅ Валидационных примеров: {len(X_test)}")

# 5. ПОСТРОЕНИЕ МОДЕЛИ LSTM
print("\n🧠 Построение архитектуры LSTM...")
model = keras.Sequential([
    # LSTM блок 1
    layers.LSTM(256, return_sequences=True, input_shape=(lookback, 1)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    # LSTM блок 2
    layers.LSTM(128, return_sequences=True),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    # LSTM блок 3
    layers.LSTM(64, return_sequences=True),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    # LSTM блок 4
    layers.LSTM(32, return_sequences=False),
    layers.BatchNormalization(),
    layers.Dropout(0.1),
    
    # Dense блоки
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)  # Output
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae', 'mse']
)

print("✅ Модель построена успешно")
print(f"✅ Всего параметров: {model.count_params():,}")

# 6. CALLBACKS
print("\n⚙️  Настройка callbacks...")
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

checkpoint = callbacks.ModelCheckpoint(
    'lstm_eds_best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=0
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=0
)

print("✅ Callbacks готовы")

# 7. ОБУЧЕНИЕ
print("\n🚀 Начало обучения (может занять ~10-15 минут)...")
print("="*70)

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

print("="*70)

# 8. ОЦЕНКА
print("\n📈 Оценка качества модели...")
train_loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]
train_mae = history.history['mae'][-1]
val_mae = history.history['val_mae'][-1]

print(f"✅ Финальная loss (обучение): {train_loss:.6f}")
print(f"✅ Финальная loss (валидация): {val_loss:.6f}")
print(f"✅ Финальная MAE (обучение): {train_mae:.6f}")
print(f"✅ Финальная MAE (валидация): {val_mae:.6f}")

# 9. СОХРАНЕНИЕ
print("\n💾 Сохранение модели и скалера...")
model.save('lstm_eds_model.keras')
print("✅ Модель сохранена: lstm_eds_model.keras")

joblib.dump(scaler, 'lstm_eds_scaler.pkl')
print("✅ Скалер сохранён: lstm_eds_scaler.pkl")

print("\n" + "="*70)
print("✅ ГОТОВО!")
print("="*70)
print("\nМодель полностью обучена и готова к использованию в приложении!")
print(f"Рассчитано на основе физической формулы ЭДС")
print(f"Средняя ошибка прогноза (MAE): {val_mae:.4f}")
print("\nДанные файлы созданы:")
print("  📁 lstm_eds_model.keras")
print("  📁 lstm_eds_best_model.keras")
print("  📁 lstm_eds_scaler.pkl")
