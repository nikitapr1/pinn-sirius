# Рабочий проект PINN-SEIR Almaty

Код извлечён из `PINN_data_cases.zip → PINN_SEIR/PINN_SEIR_Almaty.rar`.
Три Python-файла перенесены без изменений; контрольные суммы исходных файлов
находятся в `archive-manifest.json`.

- `PINN_school.py` — дообучение по выбранному региону Казахстана. Сейчас
  `REGION_NAME = 'Almaty'`, `EPOCHS = 1500` и `LEARNING_RATE = 1e-4`.
- `PINN_tools.py` — архитектура `PINN_with_params`, SEIR loss и работа с датами.
- `PINN_SEIR_example.py` — пример применения предобученной модели к Новосибирску.

Зависимости по исходным импортам: Python, torch, numpy, pandas, matplotlib, scipy.
Совместимость конкретных версий с предобученным checkpoint ещё не проверена.
В Kaggle сначала использовать встроенные библиотеки, без переустановки torch/CUDA.

## Входные файлы вне GitHub

Оригинал содержит CSV и `pinn_seir_1.pth`; они не опубликованы в публичном
репозитории. Полностью извлечённая папка в текущем рабочем окружении:
`/workspace/scratch/pinn-import/extracted/PINN_SEIR_Almaty`.
Долгосрочный исходник — предоставленный пользователем архив Google Drive.
Для Kaggle эти файлы следует подключить как приватный Dataset:

- `cases_kazakhstan.csv` и `pinn_seir_1.pth` нужны `PINN_school.py`;
- `NSK_data.csv` и тот же checkpoint нужны `PINN_SEIR_example.py`;
- также в архиве есть `altay-region-data.csv`, `mos_data.csv`,
  `omsk-region-data.csv`, `spb_data.csv`.

После подготовки совместимого окружения и помещения входов в рабочую папку
исходный запуск предполагает текущую директорию проекта:

```bash
cd PINN_SEIR_Almaty
python PINN_school.py
```

## Статус интеграции

Сейчас выполнен импорт исходников и проверка синтаксиса. Полное дообучение не
запускалось. Автоматический workflow по-прежнему запускает короткий тест
`kaggle/smoke_test.py`, а не этот 1500-эпоховый эксперимент.

Перед реальным запуском в Kaggle нужно адаптировать пути к `/kaggle/input`,
выходы к `OUTPUT_DIR`, добавить аргументы для короткого теста, метрики и
сохранение resumable checkpoint. Сейчас исходник сохраняет итоговый PNG и
вызывает `plt.show()`, но не сохраняет веса дообученной модели. Автозапуск
полного обучения при каждом коммите не включать.
