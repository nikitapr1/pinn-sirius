import os
import copy
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
from PINN_tools import PINN_with_params, get_region_dates

def sir_ode(t, y):
    S, E, I, R = y
    beta_val = beta_interp(t) / (N_POP)
    sigma_val = sigma_interp(t)
    gamma_val = gamma_interp(t)
    dS = -beta_val * S * I
    dE = beta_val * S * I - sigma_val * E
    dI = sigma_val * E - gamma_val * I
    dR = gamma_val * I
    return [dS, dE, dI, dR]


if __name__ == "__main__":
    N_POP = 200.0          # Масштаб популяции внутри PINN
    POP_TOTAL = 2_780_292  # Население Новосибирской области
    DISEASE_DURATION = 10  # Длительность болезни для скользящей суммы (дней)

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    df_nsk = pd.read_csv('NSK_data.csv')
    date_col = 'Date' if 'Date' in df_nsk.columns else 'Data'

    start_idx, end_idx = 44, 1007
    df_slice = df_nsk.iloc[start_idx:end_idx].copy().reset_index(drop=True)
    dates_nsk = pd.to_datetime(df_slice[date_col].values)

    # Исходный ряд выявлений
    I_raw = df_slice['new_diagnoses'].fillna(0).to_numpy(dtype=float)

    # 1) Сглаживание 7-дневным окном
    cases_smooth = pd.Series(I_raw).rolling(window=7, min_periods=1, center=True).mean().to_numpy()

    # 2) Скользящая сумма за 10 дней
    I_window_sum = pd.Series(cases_smooth).rolling(window=DISEASE_DURATION, min_periods=1).sum().to_numpy()
    I_window_sum = np.maximum(I_window_sum, 1.0)

    # Масштабированные данные для PINN (в долях от N_POP)
    I_data = (I_window_sum[:, None] / POP_TOTAL) * N_POP
    T = len(I_data)
    t_days = np.arange(T, dtype=float)

    print(f"Загружено {T} дней: {dates_nsk[0].date()} — {dates_nsk[-1].date()}")
    print(f"Пик скользящей суммы (10 дней): {I_window_sum.max():.0f} чел.")
    print(f"Пик на графике: {(I_window_sum).max():.0f}")

    
    plt.plot(I_window_sum)
    plt.show()

    dates_nsk, dates_pred_nsk = get_region_dates(df_nsk, date_col='Date', start_idx=44, end_idx=1007, n_points=T)

    # Инициализация сети
    model_pinn = PINN_with_params([1, 128, 128, 128, 128, 128, 3], [1, 128, 128, 128, 3], N_POP, T)

    t_norm_all = torch.linspace(-1.0, 1.0, T).reshape(-1,1)

    checkpoint = torch.load('pinn_seir_1.pth')
    model_pinn.load_state_dict(checkpoint['model_state_dict'])
    loss_w = checkpoint['loss_weights']
    loss_weights = (loss_w['w_pde_S'], loss_w['w_pde_E'], loss_w['w_pde_I'], loss_w['w_pde_R'], loss_w['w_ic_E'], loss_w['w_ic_I'], loss_w['w_ic_R'], loss_w['w_data_I'], loss_w['w_mono_S'], loss_w['w_mono_R'])

    E_pred, I_pred, R_pred, beta_pred, sigma_pred, gamma_pred, E0_pred = model_pinn(t_norm_all)
    S_pred = N_POP - E_pred - I_pred - R_pred

    S_pred = S_pred.detach().numpy()
    E_pred = E_pred.detach().numpy()    
    I_pred = I_pred.detach().numpy()
    R_pred = R_pred.detach().numpy()
    beta_pred_nsk = beta_pred.detach().numpy()
    sigma_pred_nsk = sigma_pred.detach().numpy()
    gamma_pred_nsk = gamma_pred.detach().numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(dates_nsk, I_pred*POP_TOTAL/N_POP, label="I_pred", linewidth=2, color='red')
    plt.scatter(dates_nsk, (I_window_sum)[:, None].flatten(), s=20, label="I_true", alpha=0.5, color='black')
    # plt.legend()
    plt.xlabel("t")
    plt.ylabel("I(t)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.tick_params(labelsize=15)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(dates_nsk, beta_pred_nsk/T*2, label="β(t)", color='black', marker='*', markevery=30)
    plt.grid(True, alpha=0.3)
    plt.plot(dates_nsk, sigma_pred_nsk/T*2, label="σ(t)", color='black', marker='.', markevery=30)
    plt.plot(dates_nsk, gamma_pred_nsk/T*2, label="γ(t)", color='black')
    plt.tick_params(labelsize=15)
    plt.legend()


    # Создаем интерполяционные функции для beta и gamma
    beta_interp = interp1d(t_norm_all.numpy().flatten(), beta_pred_nsk.flatten(), kind='linear', bounds_error=False, fill_value='extrapolate')
    sigma_interp = interp1d(t_norm_all.numpy().flatten(), sigma_pred_nsk.flatten(), kind='linear', bounds_error=False, fill_value='extrapolate')
    gamma_interp = interp1d(t_norm_all.numpy().flatten(), gamma_pred_nsk.flatten(), kind='linear', bounds_error=False, fill_value='extrapolate')

    E0_found = 5*torch.sigmoid(model_pinn.E0_raw).item()

    I0 = I_data[0].item()
    E0 = E0_found
    R0 = 0.0
    S0 = N_POP - I0 - E0 - R0
    t_eval = np.linspace(-1, 1, T)
    sol = solve_ivp(sir_ode, (-1, 1), [S0, E0, I0, R0], t_eval=t_eval, rtol=1e-8)

    # Получаем решение
    S_ode = sol.y[0]
    E_ode = sol.y[1]
    I_ode = sol.y[2]
    R_ode = sol.y[3]

    plt.figure(figsize=(10, 6))
    plt.plot(dates_nsk, I_ode*POP_TOTAL/N_POP, label="I_pred", linewidth=2, color='red')
    # plt.plot(t_test, R_pred, label="R_pred", linewidth=2, color='black')
    plt.scatter(dates_nsk, (I_window_sum)[:, None].flatten(), s=20, label="I_true", alpha=0.5, color='black')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.tick_params(labelsize=15)
    plt.show()

    R0_nsk = beta_pred_nsk / gamma_pred_nsk
    plt.figure(figsize=(10, 6))
    plt.grid(True, alpha=0.3)
    plt.plot(dates_nsk, R0_nsk, label="Новосибирская область", color='#0072B2')
    plt.tight_layout()
    plt.tick_params(labelsize=13)