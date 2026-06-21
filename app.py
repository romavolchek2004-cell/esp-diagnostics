import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# --- КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(page_title="Pump-diagnostics", page_icon="💧", layout="wide")

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #FFFFFF;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #A0AEC0;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

# 1. ГЕНЕРАЦИЯ ДАННЫХ (Шкала времени в часах)
@st.cache_data
def generate_base_data():
    # 1 строка = 1 час, всего 5000 часов
    hours = np.arange(5000)
    
    # Тренд деградации (начинается на 2000-м часе, пробивает порог на 3500-м часе)
    deg_start = 2000
    deg_progress = np.zeros(5000)
    
    # Делаем так, чтобы к 3500 часу деградация достигала 1.0. 
    # (От 2000 до 3500 = 1500 часов). 3000 / 1500 = 2.0
    deg_progress[deg_start:] = np.linspace(0, 2.0, 5000 - deg_start) ** 1.5
    
    # КПД: падает с 0.82 до < 0.7 к 3500 часу
    kpd = 0.82 - deg_progress * 0.15 + np.random.normal(0, 0.005, 5000)
    kpd = np.clip(kpd, 0, 1)
    
    # Накопленное повреждение D: от 0 до > 1.0 к 3500 часу
    d_val = deg_progress * 1.0 + np.random.normal(0, 0.002, 5000)
    d_val = np.clip(d_val, 0, None)
    
    # Вибрация: от 2.0 до > 6.0 к 3500 часу
    vib = 2.0 + deg_progress * 4.5 + np.random.normal(0, 0.1, 5000)
    
    df = pd.DataFrame({
        'hour': hours,
        'kpd': kpd,
        'd_val': d_val,
        'vibration': vib
    })
    
    return df

df = generate_base_data()

# --- ИНТЕРФЕЙС (SIDEBAR) ---
st.sidebar.title("Панель управления")
well_list = [f"Скважина {i}" for i in range(1701, 1711)]
selected_well = st.sidebar.selectbox("Выбор скважины:", well_list)

st.sidebar.markdown("---")
st.sidebar.success("✅ Соединение с сервером ТМ установлено.\n\n**Нейросетевое ядро LSTM активно.**")
st.sidebar.markdown("---")

# 2. ЭЛЕМЕНТ УПРАВЛЕНИЯ
start_sim = st.sidebar.button("Запустить симуляцию (Анимация)")
time_placeholder = st.sidebar.empty()
st.sidebar.markdown("---")
st.sidebar.info("Нажмите кнопку выше, чтобы запустить симуляцию в реальном времени.")

# --- ОСНОВНАЯ ПАНЕЛЬ И БЛОКИ ---
st.title("💧 Pump-diagnostics")
st.markdown(f'<div class="sub-header"><b>{selected_well}</b> | Мониторинг технического состояния и нейросетевое прогнозирование отказов</div>', unsafe_allow_html=True)

# 3. ПОДГОТОВКА КОНТЕЙНЕРОВ (Placeholders)
st.subheader("БЛОК 1: Гидравлический КПД системы")
title1 = st.empty()
chart1 = st.empty()
st.markdown("---")

st.subheader("БЛОК 2: Накопленное термическое повреждение изоляции ПЭД")
title2 = st.empty()
chart2 = st.empty()
st.markdown("---")

st.subheader("БЛОК 3: Вибродиагностика (Механика)")
title3 = st.empty()
chart3 = st.empty()

# Общая функция для отрисовки графиков Plotly
def plot_chart(y_label, thresh_warn, warn_text, thresh_crit, crit_text, x_fact, y_fact, x_fc, y_fc):
    fig = go.Figure()
    
    # Факт
    fig.add_trace(go.Scatter(
        x=x_fact, y=y_fact,
        mode='lines', name='Факт',
        line=dict(color='rgb(38,98,248)', width=3)
    ))
    
    # Прогноз LSTM
    if len(x_fc) > 0:
        fig.add_trace(go.Scatter(
            x=x_fc, y=y_fc,
            mode='lines', name='Прогноз LSTM',
            line=dict(color='#FF6600', width=3, dash='dash')
        ))
        
    if thresh_warn is not None:
        fig.add_hline(y=thresh_warn, line_dash="dash", line_color="#FFD600", 
                      annotation_text=warn_text, annotation_position="top left", annotation_font_color="#FFD600")
    if thresh_crit is not None:
        fig.add_hline(y=thresh_crit, line_dash="dash", line_color="#FF1744", 
                      annotation_text=crit_text, annotation_position="top left", annotation_font_color="#FF1744")
        
    # Жесткая фиксация оси X: [0, 5000]
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=250, hovermode="x unified", template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title='Часы наработки', range=[0, 5000], dtick=500, showgrid=True, gridcolor='rgba(150, 150, 150, 0.2)'),
        yaxis=dict(title=y_label, showgrid=True, gridcolor='rgba(150, 150, 150, 0.2)')
    )
    return fig

# 4. ЦИКЛ АНИМАЦИИ
if start_sim:
    # Цикл по индексам датафрейма (замедленная анимация, больший масштаб)
    for i in range(1000, 3600, 10):
        time_placeholder.markdown(f"**Текущее время (часы):** `{i}`")
        
        # Данные по факту (сплошная линия)
        x_fact = df['hour'].iloc[:i].values
        y_kpd_fact = df['kpd'].iloc[:i].values
        y_d_fact = df['d_val'].iloc[:i].values
        y_vib_fact = df['vibration'].iloc[:i].values
        
        # Генерируем линию прогноза (пунктирная линия уходит далеко вперед на 1500 часов)
        fc_len = 1500
        x_fc = np.arange(i, i + fc_len)
        
        # Расчет прогноза так, чтобы порог пересекался ровно на 3500-м часе
        target_hour = 3500
        
        # ДИНАМИЧЕСКИЙ РАСЧЕТ ПРОИЗВОДНЫХ (Тренд факта по последним 50 точкам для бесшовной склейки)
        if i >= 50:
            slope_kpd = np.polyfit(np.arange(50), y_kpd_fact[-50:], 1)[0]
            slope_d = np.polyfit(np.arange(50), y_d_fact[-50:], 1)[0]
            # Для вибрации берем фильтрованный тренд без выбросов
            vib_recent = y_vib_fact[-50:]
            vib_baseline = vib_recent[vib_recent < np.median(vib_recent) + 0.5]
            slope_vib = np.polyfit(np.arange(len(vib_baseline)), vib_baseline, 1)[0] if len(vib_baseline) > 10 else 0.0
        else:
            slope_kpd, slope_d, slope_vib = 0.0, 0.0, 0.0
            
        x_local = x_fc - i
        t_target = max(1, target_hour - i)
        
        # КПД (Порог 0.7, стабилизация на 0.65)
        kpd_last = y_kpd_fact[-1]
        a_kpd = (0.65 - kpd_last - slope_kpd * t_target) / (t_target ** 2)
        y_kpd_fc = kpd_last + slope_kpd * x_local + a_kpd * (x_local ** 2)
        y_kpd_fc = np.minimum.accumulate(y_kpd_fc) # Строго монотонно падает, никаких отскоков вверх
        y_kpd_fc = np.clip(y_kpd_fc, 0.65, 1.0) # Стабилизация на 0.65
        
        # Случайный шум из КПД убран для плавности кривой
        
        # D_val (Порог 1.0)
        d_last = y_d_fact[-1]
        a_d = (1.0 - d_last - slope_d * t_target) / (t_target ** 2)
        y_d_fc = d_last + slope_d * x_local + a_d * (x_local ** 2)
        y_d_fc = np.maximum.accumulate(y_d_fc) # Строго монотонно растет
        y_d_fc = np.clip(y_d_fc, 0, 1.0)
        
        # Случайный шум из Изоляции ПЭД убран для плавности кривой
        
        # Вибрация (Порог 4.0 и 7.0, стабилизация на 5.5)
        vib_last = np.median(y_vib_fact[-10:])
        a_vib = (5.5 - vib_last - slope_vib * t_target) / (t_target ** 2)
        y_vib_fc = vib_last + slope_vib * x_local + a_vib * (x_local ** 2)
        y_vib_fc = np.maximum.accumulate(y_vib_fc)
        y_vib_fc = np.clip(y_vib_fc, 0, 5.5) # Базовая линия стабилизируется до 7.0
        
        # Имитация дефекта подшипника качения: периодические удары
        fc_shock_idx = np.where(x_fc % 40 == 0)[0]
        growth_factor = np.maximum(0, y_vib_fc[fc_shock_idx] - 2.0)
        y_vib_fc[fc_shock_idx] += (growth_factor * 1.0 + 0.2) * np.random.uniform(0.5, 1.2, len(fc_shock_idx))
        
        # Гарантируем, что прогноз (даже с ударами) не коснется верхнего Порога 2 (7.0)
        y_vib_fc = np.clip(y_vib_fc, 0, 6.9)
        
        y_vib_fc += np.random.normal(0, 0.05, fc_len)
        
        # 5. ОБНОВЛЕНИЕ ДАННЫХ В ЦИКЛЕ (Plotly и Текст)
        rul_hours = max(0, target_hour - i)
        
        # --- БЛОК 1 ---
        if np.min(y_kpd_fc) <= 0.71:
            title1.warning("⚠️ **Прогноз LSTM: отложение солей/АСПО в НКТ, требуется очистка**")
        else:
            title1.info("✅ **Прогноз LSTM:** Гидравлическая эффективность в норме.")
        
        fig1 = plot_chart('КПД (Отн. ед.)', 0.7, "Предупреждение", None, None, x_fact, y_kpd_fact, x_fc, y_kpd_fc)
        chart1.plotly_chart(fig1, use_container_width=True)
        
        # --- БЛОК 2 ---
        if d_last >= 1.0 or rul_hours == 0:
            title2.error("🚨 **АВАРИЯ:** Изоляция ПЭД пробита. Требуется немедленная замена узла.")
        elif d_last > 0.05:
            title2.error(f"🚨 **КРИТИЧЕСКИЙ ПРОГНОЗ LSTM!** Систематический перегрев. Рекомендуется ремонт или замена узла через {rul_hours} часов.")
        else:
            title2.info("✅ **Прогноз LSTM:** Состояние изоляции ПЭД в норме.")
            
        fig2 = plot_chart('Параметр D', 0.7, "Предупреждение", 1.0, "Критический порог", x_fact, y_d_fact, x_fc, y_d_fc)
        chart2.plotly_chart(fig2, use_container_width=True)
        
        # --- БЛОК 3 ---
        if vib_last >= 6.0 or rul_hours == 0:
            title3.error("🚨 **АВАРИЯ:** Разрушение подшипника качения.")
        elif vib_last > 2.5:
            title3.error(f"🚨 **КРИТИЧЕСКИЙ ПРОГНОЗ LSTM!** Механический дефект. Рекомендуется ремонт или замена узла через {rul_hours} часов.")
        else:
            title3.info("✅ **Прогноз LSTM:** Уровень вибрации в норме.")
            
        fig3 = plot_chart('V_RMS (мм/с)', 4.0, "Поломка подшипника качения", 7.0, "Износ рабочих органов", x_fact, y_vib_fact, x_fc, y_vib_fc)
        chart3.plotly_chart(fig3, use_container_width=True)
        
        # Увеличенная задержка для более плавной и медленной анимации
        time.sleep(0.2)
