import copy
import os
import warnings
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PINN_tools import PINN_with_params, device, loss_fn_params
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import torch

warnings.filterwarnings('ignore')

# ==============================================================================
# === БЛОК НАСТРОЕК ЭКСПЕРИМЕНТА ДЛЯ ШКОЛЬНИКОВ (МЕНЯТЬ ЗДЕСЬ!) ===
# ==============================================================================

# 1. Выберите регион из списка доступных регионов Казахстана ниже:
#    'Almaty'                   (г. Алматы)
#    'Nur-Sultan'               (г. Астана / Нур-Султан)
#    'Shymkent'                 (г. Шымкент)
#    'Karaganda Region'         (Карагандинская область)
#    'Pavlodar Region'          (Павлодарская область)
#    'East Kazakhstan Region'   (Восточно-Казахстанская область)
#    'Akmola Region'            (Акмолинская область)
#    'Almaty Region'            (Алматинская область)
#    'Atyrau Region'            (Атырауская область)
#    'Kostanay Region'          (Костанайская область)
#    'Aktobe Region'            (Актюбинская область)
#    'West Kazakhstan Region'   (Западно-Казахстанская область)
#    'North Kazakhstan Region'  (Северо-Казахстанская область)
#    'Jambyl Region'            (Жамбылская область)
#    'Mangystau Region'         (Мангистауская область)
#    'Kyzylorda Region'         (Кызылординская область)
#    'Turkistan Region'         (Туркестанская область)

REGION_NAME = 'Almaty'

# 2. Количество шагов (эпох) дообучения нейросети на новом регионе:
#    Попробуйте: 0 (без дообучения), 300 (быстрое), 1500 (полное дообучение)
EPOCHS = 1500

# 3. Средняя длительность болезни одного человека (в днях):
DISEASE_DURATION = 10

# 4. Скорость обучения нейросети (Learning Rate):
LEARNING_RATE = 1e-4

# ==============================================================================
# СПРАВОЧНИК РЕГИОНОВ КАЗАХСТАНА (НАСЕЛЕНИЕ И НАЗВАНИЯ НА РУССКОМ)
# ==============================================================================
REGIONS_INFO_KZ = {
    'Almaty': {'pop': 2_000_000, 'ru': 'г. Алматы'},
    'Nur-Sultan': {'pop': 1_200_000, 'ru': 'г. Астана (Нур-Султан)'},
    'Shymkent': {'pop': 1_100_000, 'ru': 'г. Шымкент'},
    'Karaganda Region': {'pop': 1_375_000, 'ru': 'Карагандинская область'},
    'Almaty Region': {'pop': 2_050_000, 'ru': 'Алматинская область'},
    'Turkistan Region': {'pop': 2_040_000, 'ru': 'Туркестанская область'},
    'East Kazakhstan Region': {
        'pop': 1_360_000,
        'ru': 'Восточно-Казахстанская область',
    },
    'Jambyl Region': {'pop': 1_140_000, 'ru': 'Жамбылская область'},
    'Akmola Region': {'pop': 735_000, 'ru': 'Акмолинская область'},
    'Kostanay Region': {'pop': 865_000, 'ru': 'Костанайская область'},
    'Aktobe Region': {'pop': 890_000, 'ru': 'Актюбинская область'},
    'Pavlodar Region': {'pop': 750_000, 'ru': 'Павлодарская область'},
    'Kyzylorda Region': {'pop': 815_000, 'ru': 'Кызылординская область'},
    'Mangystau Region': {'pop': 720_000, 'ru': 'Мангистауская область'},
    'Atyrau Region': {'pop': 660_000, 'ru': 'Атырауская область'},
    'West Kazakhstan Region': {
        'pop': 660_000,
        'ru': 'Западно-Казахстанская область',
    },
    'North Kazakhstan Region': {
        'pop': 540_000,
        'ru': 'Северо-Казахстанская область',
    },
}


def load_region_from_kz_csv(
    csv_path, region_name, pop_total, n_pop=200.0, disease_duration=10
):
  """Загружает данные выбранного региона из cases_kazakhstan.csv,

  восстанавливает пропущенные даты и считает активных больных I(t).
  """
  if region_name not in REGIONS_INFO_KZ:
    raise ValueError(
        f"Неизвестный регион '{region_name}'! Выберите регион из списка в"
        ' начале файла.'
    )

  df = pd.read_csv(csv_path)
  sub = df[df['name'] == region_name].copy()
  sub['date'] = pd.to_datetime(sub['date'])
  sub = sub.sort_values('date').drop_duplicates('date')

  # Срез 963 дня (21 апреля 2020 — 9 декабря 2022), совпадающий с базовой моделью
  full_dates = pd.date_range('2020-04-21', '2022-12-09', freq='D')
  grid = pd.DataFrame({'date': full_dates}).merge(
      sub[['date', 'cumulative_cases']], on='date', how='left'
  )
  grid['cumulative_cases'] = (
      grid['cumulative_cases'].interpolate(method='linear').bfill().ffill()
  )

  # Вычисляем ежедневные выявления без скачков после выходных
  I_raw = (
      grid['cumulative_cases']
      .diff()
      .fillna(0.0)
      .clip(lower=0.0)
      .to_numpy(dtype=float)
  )

  # 1) Сглаживание 7-дневным окном
  cases_smooth = (
      pd.Series(I_raw)
      .rolling(window=7, min_periods=1, center=True)
      .mean()
      .to_numpy()
  )

  # 2) Скользящая сумма за время болезни (переход к активным больным I(t))
  I_window_sum = (
      pd.Series(cases_smooth)
      .rolling(window=disease_duration, min_periods=1)
      .sum()
      .to_numpy()
  )
  I_window_sum = np.maximum(I_window_sum, 1.0)

  # 3) Масштабирование для PINN
  I_data = (I_window_sum[:, None] / pop_total) * n_pop
  return grid['date'].to_numpy(), I_window_sum, I_data


if __name__ == '__main__':
  SEED = 42
  np.random.seed(SEED)
  torch.manual_seed(SEED)

  N_POP = 200.0
  POP_TOTAL = REGIONS_INFO_KZ[REGION_NAME]['pop']
  REGION_RU = REGIONS_INFO_KZ[REGION_NAME]['ru']

  print('=' * 70)
  print(f' ЗАПУСК ЭКСПЕРИМЕНТА PINN-SEIR: {REGION_RU} ({REGION_NAME})')
  print(
      f' Население: {POP_TOTAL:,} чел. | Эпох дообучения: {EPOCHS} |'
      f' Длительность болезни: {DISEASE_DURATION} дн.'
  )
  print('=' * 70)

  # 1. Загружаем данные региона Казахстана
  dates_reg, I_window_sum, I_data = load_region_from_kz_csv(
      'cases_kazakhstan.csv',
      REGION_NAME,
      POP_TOTAL,
      N_POP,
      DISEASE_DURATION,
  )
  T = len(I_data)
  print(
      f'Загружено {T} дней:'
      f' {pd.to_datetime(dates_reg[0]).date()} —'
      f' {pd.to_datetime(dates_reg[-1]).date()}'
  )
  print(
      'Реальный пик одновременно болеющих в регионе:'
      f' {I_window_sum.max():.0f} чел.'
  )

  # 2. Инициализируем нейросеть и загружаем предобученные веса (Новосибирск)
  model_pinn = PINN_with_params(
      [1, 128, 128, 128, 128, 128, 3],
      [1, 128, 128, 128, 3],
      N_POP,
      T,
  ).to(device)

  checkpoint = torch.load('pinn_seir_1.pth', map_location=device)
  model_pinn.load_state_dict(checkpoint['model_state_dict'])
  loss_w = checkpoint['loss_weights']
  loss_weights = (
      loss_w['w_pde_S'],
      loss_w['w_pde_E'],
      loss_w['w_pde_I'],
      loss_w['w_pde_R'],
      loss_w['w_ic_E'],
      loss_w['w_ic_I'],
      loss_w['w_ic_R'],
      loss_w['w_data_I'],
      loss_w['w_mono_S'],
      loss_w['w_mono_R'],
  )

  t_norm_all = torch.linspace(-1.0, 1.0, T, device=device).reshape(-1, 1)
  I_data_t = torch.tensor(I_data, dtype=torch.float32, device=device).reshape(
      -1, 1
  )

  # 3. Дообучение (Fine-tuning) на данных выбранного региона Казахстана
  if EPOCHS > 0:
    print(f"\nНачинаем дообучение нейросети на регионе '{REGION_RU}'...")
    optimizer = torch.optim.Adam(
        model_pinn.parameters(), lr=LEARNING_RATE, weight_decay=5.8e-5
    )

    t_colloc = t_norm_all.clone().detach()
    t_data_tensor = t_norm_all.clone().detach()
    I0_t = torch.tensor([[I_data[0, 0]]], dtype=torch.float32, device=device)
    R0_t = torch.tensor([[0.0]], dtype=torch.float32, device=device)
    S0_t = torch.tensor(
        [[N_POP - I_data[0, 0]]], dtype=torch.float32, device=device
    )

    best_loss = float('inf')
    best_state = copy.deepcopy(model_pinn.state_dict())

    for epoch in range(1, EPOCHS + 1):
      optimizer.zero_grad()
      loss, loss_data_val = loss_fn_params(
          model_pinn,
          t_colloc,
          t_data_tensor,
          I_data_t,
          S0_t,
          I0_t,
          R0_t,
          loss_weights,
      )
      loss.backward()
      optimizer.step()

      if loss.item() < best_loss:
        best_loss = loss.item()
        best_state = copy.deepcopy(model_pinn.state_dict())

      if epoch % max(1, EPOCHS // 5) == 0 or epoch == 1:
        print(
            f'  Шаг {epoch:4d}/{EPOCHS} | Общая ошибка: {loss.item():.4f} |'
            f' Ошибка по данным: {loss_data_val:.4f}'
        )

    model_pinn.load_state_dict(best_state)
    print('Дообучение успешно завершено!')
  else:
    print(
        '\nДообучение отключено (EPOCHS = 0). Используются исходные веса'
        ' Новосибирска.'
    )

  # 4. Получаем предсказание нейросети и скрытые параметры
  model_pinn.eval()
  with torch.no_grad():
    E_pred, I_pred, R_pred, beta_pred, sigma_pred, gamma_pred, E0_pred = (
        model_pinn(t_norm_all)
    )

  I_pred_np = I_pred.cpu().numpy().flatten()
  beta_np = beta_pred.cpu().numpy().flatten()
  sigma_np = sigma_pred.cpu().numpy().flatten()
  gamma_np = gamma_pred.cpu().numpy().flatten()

  # 5. Проверяем найденные параметры через решение дифференциальных уравнений (ОДУ)
  t_grid = t_norm_all.cpu().numpy().flatten()
  beta_interp = interp1d(
      t_grid, beta_np, kind='linear', bounds_error=False, fill_value='extrapolate'
  )
  sigma_interp = interp1d(
      t_grid,
      sigma_np,
      kind='linear',
      bounds_error=False,
      fill_value='extrapolate',
  )
  gamma_interp = interp1d(
      t_grid,
      gamma_np,
      kind='linear',
      bounds_error=False,
      fill_value='extrapolate',
  )

  def seir_ode(t, y):
    S, E, I, R = y
    b_val = beta_interp(t) / N_POP
    s_val = sigma_interp(t)
    g_val = gamma_interp(t)
    dS = -b_val * S * I  # Исправлен знак минус!
    dE = b_val * S * I - s_val * E
    dI = s_val * E - g_val * I
    dR = g_val * I
    return [dS, dE, dI, dR]

  E0_found = 5.0 * torch.sigmoid(model_pinn.E0_raw).item()
  I0_val = I_data[0].item()
  S0_val = N_POP - I0_val - E0_found
  sol = solve_ivp(
      seir_ode, (-1.0, 1.0), [S0_val, E0_found, I0_val, 0.0], t_eval=t_grid, rtol=1e-8
  )
  I_ode_people = sol.y[2] * (POP_TOTAL / N_POP)
  I_pinn_people = I_pred_np * (POP_TOTAL / N_POP)

  # 6. Считаем метрики точности и эпидемиологические показатели
  mape_pinn = (
      np.mean(np.abs(I_window_sum - I_pinn_people) / np.maximum(I_window_sum, 1.0))
      * 100
  )
  mape_ode = (
      np.mean(np.abs(I_window_sum - I_ode_people) / np.maximum(I_window_sum, 1.0))
      * 100
  )
  R0_curve = beta_np / gamma_np

  print('\n' + '=' * 70)
  print(f' ИТОГОВЫЕ РЕЗУЛЬТАТЫ ДЛЯ РЕГИОНА: {REGION_RU} (EPOCHS = {EPOCHS})')
  print('=' * 70)
  print(f'  • Средняя ошибка нейросети (MAPE PINN):     {mape_pinn:.2f} %')
  print(f'  • Средняя ошибка мат. модели (MAPE ОДУ):    {mape_ode:.2f} %')
  print(
      '  • Пик болеющих (реальные данные):          '
      f' {I_window_sum.max():.0f} чел.'
  )
  print(
      '  • Пик болеющих (расчет модели):            '
      f' {I_ode_people.max():.0f} чел.'
  )
  print(f'  • Максимальный индекс репродукции R0(t):    {R0_curve.max():.2f}')
  print(f'  • Средний индекс репродукции R0(t):         {R0_curve.mean():.2f}')
  print('=' * 70)

  # 7. Строим наглядный дашборд из 3 графиков
  fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)

  # График 1: Число инфицированных I(t)
  axes[0].scatter(
      dates_reg,
      I_window_sum,
      s=12,
      color='black',
      alpha=0.45,
      label='Реальные данные (I_true)',
  )
  axes[0].plot(
      dates_reg,
      I_pinn_people,
      color='#D55E00',
      linewidth=2.2,
      label=f'Нейросеть PINN (MAPE={mape_pinn:.1f}%)',
  )
  axes[0].plot(
      dates_reg,
      I_ode_people,
      color='#0072B2',
      linestyle='--',
      linewidth=2.0,
      label=f'Решение уравнений SEIR (MAPE={mape_ode:.1f}%)',
  )
  axes[0].set_title(
      f'1. Динамика числа активно болеющих людей в регионе: {REGION_RU}'
      f' (Дообучение: {EPOCHS} эпох)',
      fontsize=13,
      fontweight='bold',
  )
  axes[0].set_ylabel('Число болеющих (чел.)', fontsize=11)
  axes[0].grid(True, alpha=0.3)
  axes[0].legend(fontsize=10, loc='upper left')

  # График 2: Скрытые параметры вируса (в суточной шкале 1/день)
  scale_to_days = 2.0 / T
  axes[1].plot(
      dates_reg,
      beta_np * scale_to_days,
      label='Заразность вируса β(t)',
      color='#E69F00',
      linewidth=2,
  )
  axes[1].plot(
      dates_reg,
      sigma_np * scale_to_days,
      label='Переход из инкубации σ(t)',
      color='#009E73',
      linewidth=2,
  )
  axes[1].plot(
      dates_reg,
      gamma_np * scale_to_days,
      label='Скорость выздоровления γ(t)',
      color='#CC79A7',
      linewidth=2,
  )
  axes[1].set_title(
      '2. Восстановленные нейросетью скрытые параметры эпидемии (1/день)',
      fontsize=13,
      fontweight='bold',
  )
  axes[1].set_ylabel('Значение параметра', fontsize=11)
  axes[1].grid(True, alpha=0.3)
  axes[1].legend(fontsize=10, loc='upper left')

  # График 3: Базовое репродуктивное число R0(t)
  axes[2].plot(
      dates_reg,
      R0_curve,
      color='#0072B2',
      linewidth=2.2,
      label=f'Индекс репродукции R₀(t) — {REGION_RU}',
  )
  axes[2].axhline(
      1.0,
      color='red',
      linestyle='--',
      linewidth=1.5,
      label='Порог эпидемии (R₀ = 1.0)',
  )
  axes[2].fill_between(
      dates_reg,
      1.0,
      R0_curve,
      where=(R0_curve >= 1.0),
      color='red',
      alpha=0.15,
      label='Зона роста вспышки (R₀ > 1)',
  )
  axes[2].set_title(
      '3. Базовое репродуктивное число R₀(t) = β(t) / γ(t)',
      fontsize=13,
      fontweight='bold',
  )
  axes[2].set_ylabel('R₀(t)', fontsize=11)
  axes[2].set_xlabel('Дата', fontsize=11)
  axes[2].grid(True, alpha=0.3)
  axes[2].legend(fontsize=10, loc='upper left')

  plt.tight_layout()
  out_filename = f"result_{REGION_NAME.replace(' ', '_')}_{EPOCHS}ep.png"
  plt.savefig(out_filename, dpi=150)
  print(f'\nГрафик сохранен в файл: {out_filename}')
  plt.show()