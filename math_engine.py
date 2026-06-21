import numpy as np
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis

class ESP_Math_Apparatus:
    """
    Математическая модель расчета прогнозируемых компонентов предиктивной диагностики
    для оценки технического состояния установки электроцентробежного насоса (УЭЦН).
    
    Подготавливает вектор признаков для последующей передачи в архитектуру нейросети (LSTM).
    """
    
    def __init__(self):
        # Универсальная газовая постоянная R, Дж/(моль*К)
        self.R = 8.314
        # Перевод энергии активации из эВ в Дж/моль (1 эВ = 96 485 Дж/моль по тексту)
        self.eV_to_J_mol = 96485.0
        
    def calculate_hydraulic_power_kw(self, Q_sut: float, P_vyk: float, P_pr: float) -> float:
        """
        Расчет полезной гидравлической мощности.
        Q_sut: объемный расход жидкости, м³/сут
        P_vyk: давление на выкиде насоса, МПа
        P_pr: давление на приеме насоса, МПа
        Возвращает: гидравлическую мощность, кВт
        """
        # Защита от математических ошибок: исключаем отрицательный расход и отрицательный перепад давления
        Q_sut = max(0.0, Q_sut)
        delta_P = max(0.0, P_vyk - P_pr)
        
        # Расчет объемного расхода жидкости в м³/с по формуле (3.14)
        Q_sec = Q_sut / 86400.0
        
        # Перевод перепада давлений из МПа в Па по формуле (3.15)
        delta_P_Pa = delta_P * 1e6
        
        # Базовая формула гидравлической мощности (Вт) по формуле (3.13) и (3.16)
        # P_гидр = (Q_сут / 86400) * (P_вык - P_пр) * 10^6
        P_hydr_w = Q_sec * delta_P_Pa
        
        # Перевод результата в киловатты по формуле (3.17)
        # P_гидр = (Q_сут * (P_вык - P_пр) * 10^3) / 86400
        P_hydr_kw = P_hydr_w / 1000.0
        
        return P_hydr_kw

    def calculate_efficiency(self, P_hydr_kw: float, P_act: float) -> float:
        """
        Расчет коэффициента полезного действия (КПД) системы.
        P_hydr_kw: гидравлическая мощность, кВт
        P_act: фактическая активная электрическая мощность, кВт
        Возвращает: КПД системы
        """
        # Защита от деления на ноль при отсутствии потребляемой мощности
        if P_act <= 1e-6:
            return 0.0
            
        # Определение коэффициента полезного действия системы по формуле (3.18)
        # eta_сист = P_гидр / P_акт
        eta_syst = P_hydr_kw / P_act
        
        # Ограничиваем КПД максимальным значением 1.0 (или 100%)
        return min(eta_syst, 1.0)

    def calculate_resource_at_temp(self, T: float, L_nom: float, T_nom: float, E_a_ev: float = 1.0) -> float:
        """
        Определение времени до отказа при постоянной температуре (термокинетическая модель).
        T: абсолютная температура изоляции, К
        L_nom: номинальный срок службы изоляции, часов
        T_nom: эталонная температура, К
        E_a_ev: энергия активации для изоляции ПЭД, эВ
        Возвращает: время до отказа L(T), часов
        """
        # Защита от деления на ноль при околонулевой температуре
        if T < 1e-3 or T_nom < 1e-3:
            return float('inf')
            
        # Перевод энергии активации из эВ в Дж/моль
        E_a_J = E_a_ev * self.eV_to_J_mol
        
        # Расчет времени до отказа при постоянной температуре по формуле (3.20)
        # L(T) = L_ном * exp[ (E_a / R) * (1 / T - 1 / T_ном) ]
        exponent_val = (E_a_J / self.R) * (1.0 / T - 1.0 / T_nom)
        
        # Защита от математической ошибки переполнения (overflow) экспоненты
        exponent_val = np.clip(exponent_val, -700, 700)
        
        # Вычисление итогового ресурса
        L_T = L_nom * np.exp(exponent_val)
        
        return max(0.0, float(L_T))

    def calculate_accumulated_damage(self, time_intervals_h: np.ndarray, temperatures_K: np.ndarray, 
                                     L_nom: float, T_nom: float, E_a_ev: float = 1.0) -> float:
        """
        Расчет накопленного термического повреждения изоляции ПЭД D (правило Майнера).
        time_intervals_h: массив длительностей интервалов, часов
        temperatures_K: массив средних температур на интервалах, К
        Возвращает: D (накопленное повреждение)
        """
        if len(time_intervals_h) != len(temperatures_K):
            raise ValueError("Размерности массивов времени и температуры должны совпадать.")
            
        damage = 0.0
        
        # Интеграл доли использованного ресурса по формуле (3.21) заменяется суммой 
        # для дискретных отсчетов телеметрии по формуле (3.22)
        for dt_i, T_i in zip(time_intervals_h, temperatures_K):
            # Расчет ресурса для текущего интервала (L(T_i))
            L_Ti = self.calculate_resource_at_temp(T_i, L_nom, T_nom, E_a_ev)
            if L_Ti > 0:
                # Суммирование приращения повреждения по формуле (3.22)
                # D = sum( delta_t_i / L(T_i) )
                damage += dt_i / L_Ti
                
        return float(damage)

    def calculate_vrms(self, v_array: np.ndarray) -> float:
        """
        Расчет среднеквадратичного значения (RMS) виброскорости.
        v_array: массив мгновенных значений виброскорости, мм/с
        Возвращает: V_RMS, мм/с
        """
        N = len(v_array)
        if N == 0:
            return 0.0
            
        # Среднеквадратичное значение виброскорости рассчитывается по формуле (3.25)
        # V_RMS = sqrt( (1 / N) * sum(v_i^2) )
        v_rms = np.sqrt(np.mean(v_array**2))
        
        return float(v_rms)

    def calculate_crest_factor(self, v_array: np.ndarray, v_rms: float) -> float:
        """
        Расчет пик-фактора (Crest Factor) вибрационного сигнала.
        v_array: массив мгновенных значений виброскорости, мм/с
        v_rms: среднеквадратичное значение виброскорости, мм/с
        Возвращает: пик-фактор CF
        """
        # Защита от деления на ноль при отсутствии вибрации
        if v_rms <= 1e-6 or len(v_array) == 0:
            return 0.0
            
        # Пик-фактор рассчитывается по формуле (3.26)
        # CF = max(|v_i|) / V_RMS
        max_abs_v = np.max(np.abs(v_array))
        cf = max_abs_v / v_rms
        
        return float(cf)

    def calculate_kurtosis(self, v_array: np.ndarray) -> float:
        """
        Расчет эксцесса (Kurtosis) формы вибросигнала.
        v_array: массив мгновенных значений виброскорости, мм/с
        Возвращает: эксцесс (K)
        """
        if len(v_array) < 2:
            return 0.0
        
        # Защита от деления на ноль при нулевой дисперсии сигнала
        if np.var(v_array) <= 1e-12:
             # Для нормального оборудования эксцесс близок к 3
             return 3.0 
             
        # Эксцесс рассчитывается по формуле (3.27)
        # K = [ (1/N) * sum(v_i - v_mean)^4 ] / [ (1/N) * sum(v_i - v_mean)^2 ]^2
        # fisher=False использует определение Пирсона (без вычитания 3)
        k_val = kurtosis(v_array, fisher=False)
        
        return float(k_val)

    def calculate_fft_and_amplitudes(self, v_array: np.ndarray, fs: float):
        """
        Расчет амплитудного спектра с использованием дискретного преобразования Фурье.
        v_array: массив мгновенных значений виброскорости, мм/с
        fs: частота дискретизации, Гц
        Возвращает: freqs (массив частот), amplitudes (амплитуды спектра)
        """
        N = len(v_array)
        if N == 0:
            return np.array([]), np.array([])
            
        # Дискретное преобразование Фурье (ДПФ) вычисляется по формуле (3.30)
        # V_k = sum( v_n * exp(-j * 2 * pi * k * n / N) )
        V_k = fft(v_array)
        
        # Получение частот (f_k = k * f_д / N)
        freqs = fftfreq(N, 1/fs)
        
        # Отбор только положительных частот для амплитудного спектра
        pos_mask = freqs >= 0
        freqs_pos = freqs[pos_mask]
        V_k_pos = V_k[pos_mask]
        
        # Односторонний амплитудный спектр получают по формуле (3.31)
        # A_k = (2 / N) * |V_k|
        amplitudes = (2.0 / N) * np.abs(V_k_pos)
        
        return freqs_pos, amplitudes

    def calculate_unbalance_indicator(self, vx_array: np.ndarray, vy_array: np.ndarray, f_rot: float, fs: float) -> float:
        """
        Расчет индикатора дисбаланса (UI).
        vx_array, vy_array: массивы виброскорости по ортогональным осям X и Y
        f_rot: частота вращения вала, Гц
        fs: частота дискретизации, Гц
        Возвращает: индикатор дисбаланса UI
        """
        if len(vx_array) != len(vy_array) or len(vx_array) == 0:
            return 0.0
            
        N = len(vx_array)
        freqs = fftfreq(N, 1/fs)
        
        # Нахождение индекса частоты, наиболее близкой к частоте вращения вала f_rot
        idx_rot = np.argmin(np.abs(freqs - f_rot))
        
        # Получение комплексных спектров V_x(f) и V_y(f) через ДПФ
        V_x = fft(vx_array)
        V_y = fft(vy_array)
        
        # Фазовые углы комплексной амплитуды (arg(V))
        phase_x = np.angle(V_x[idx_rot], deg=True)
        phase_y = np.angle(V_y[idx_rot], deg=True)
        
        # Разность фаз между сигналами на частоте вращения вала по формуле (3.28)
        # delta_phi_xy = arg(V_x(f_rot)) - arg(V_y(f_rot))
        delta_phi_xy = phase_x - phase_y
        
        # Приведение разности фаз к диапазону [-180, 180]
        delta_phi_xy = (delta_phi_xy + 180) % 360 - 180
        
        # Индикатор дисбаланса рассчитывают по формуле (3.29)
        # UI = |delta_phi_xy - 90| / 90
        ui = np.abs(np.abs(delta_phi_xy) - 90.0) / 90.0
        
        return float(ui)

    def calculate_high_freq_energy(self, amplitudes: np.ndarray, freqs: np.ndarray, f_low: float, f_high: float) -> float:
        """
        Расчет суммарной энергии в высокочастотном диапазоне.
        amplitudes: амплитудный спектр
        freqs: массив частот
        f_low, f_high: границы диапазона, Гц
        Возвращает: энергия E_HF
        """
        # Фильтрация частот, попадающих в заданный диапазон [f_низ; f_верх]
        mask = (freqs >= f_low) & (freqs <= f_high)
        amps_in_range = amplitudes[mask]
        
        # Суммарная энергия в высокочастотном диапазоне вычисляется по формуле (3.32)
        # E_HF = sum( A_f^2 )
        E_HF = np.sum(amps_in_range**2)
        
        return float(E_HF)
        
    def get_feature_vector(self,
                           Q_sut: float, P_vyk: float, P_pr: float, P_act: float,
                           time_intervals_h: np.ndarray, temperatures_K: np.ndarray, 
                           L_nom: float, T_nom: float, E_a_ev: float,
                           v_array_z: np.ndarray, vx_array: np.ndarray, vy_array: np.ndarray,
                           fs: float, f_rot: float, f_hf_low: float, f_hf_high: float) -> np.ndarray:
        """
        Финальный метод для формирования вектора признаков X на вход нейросети (LSTM).
        Собирает все рассчитанные показатели в единый вектор.
        """
        # 1. Расчет системного коэффициента полезного действия (формулы 3.16, 3.17, 3.18)
        P_hydr_kw = self.calculate_hydraulic_power_kw(Q_sut, P_vyk, P_pr)
        eta_syst = self.calculate_efficiency(P_hydr_kw, P_act)
        
        # 2. Расчет накопленного повреждения D (термокинетическая модель, формулы 3.20, 3.22)
        D = self.calculate_accumulated_damage(time_intervals_h, temperatures_K, L_nom, T_nom, E_a_ev)
        
        # 3. Расчет интегральных параметров вибродиагностики (формулы 3.25, 3.26, 3.27)
        v_rms = self.calculate_vrms(v_array_z)
        cf = self.calculate_crest_factor(v_array_z, v_rms)
        k_val = self.calculate_kurtosis(v_array_z)
        
        # 4. Расчет индикатора дисбаланса (формулы 3.28, 3.29)
        ui = self.calculate_unbalance_indicator(vx_array, vy_array, f_rot, fs)
        
        # 5. Расчет амплитуд гармоник и энергии ВЧ-вибрации (формулы 3.30, 3.31, 3.32)
        freqs, amplitudes = self.calculate_fft_and_amplitudes(v_array_z, fs)
        
        if len(freqs) > 0:
            # Поиск амплитуд 1-й, 2-й и 3-й гармоник (A_1, A_2, A_3)
            idx_1 = np.argmin(np.abs(freqs - f_rot))
            idx_2 = np.argmin(np.abs(freqs - 2*f_rot))
            idx_3 = np.argmin(np.abs(freqs - 3*f_rot))
            A_1 = amplitudes[idx_1]
            A_2 = amplitudes[idx_2]
            A_3 = amplitudes[idx_3]
        else:
            A_1 = A_2 = A_3 = 0.0
            
        E_HF = self.calculate_high_freq_energy(amplitudes, freqs, f_hf_low, f_hf_high)
        
        # 6. Формирование входа для нейросети по формуле (3.33)
        # X = [eta_сист, D, V_RMS, CF, K, UI, A_1, A_2, A_3, E_HF]
        X = np.array([
            eta_syst,
            D,
            v_rms,
            cf,
            k_val,
            ui,
            A_1,
            A_2,
            A_3,
            E_HF
        ], dtype=float)
        
        return X
