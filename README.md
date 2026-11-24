# Рекомендательная система товаров интернет-магазина

Курсовая работа по дисциплине ПИС.

## Данные для запуска

Для варианта из листа задания нужен файл отзывов Amazon Reviews 2023, категория Video Games, выборка 5-core, в формате JSON Lines. Поместите его в папку `data` под именем `Video_Games_5.jsonl.gz` или `Video_Games_5.jsonl`. Нужны поля `user_id`, `parent_asin`, `rating`, `title` и `text`. Источник данных и описание контрольных выборок опубликованы [McAuley Lab](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).

Для воспроизведения исходного прототипа можно использовать прежнюю выборку `reviews_Toys_and_Games_5.json.gz`. Команды для PowerShell:

```powershell
New-Item -ItemType Directory -Force data | Out-Null
curl.exe -fL "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz" -o data/reviews_Toys_and_Games_5.json.gz
```

Если в `data` есть обе выборки, приложение выбирает Video Games. Чтобы задать другой путь, установите переменную `RECOMMENDER_DATA`. Для Compose путь должен быть виден внутри контейнера, например `/app/data/reviews.jsonl.gz`. Данные подключаются в режиме чтения и хранятся отдельно от образа.

## Запуск через Docker Compose

Из корня репозитория при запущенном Docker:

```bash
docker compose up --build
```

Откройте [Streamlit](http://localhost:8501), дождитесь обучения моделей, выберите пользователя и нажмите «Показать рекомендации». Для завершения нажмите Ctrl+C, затем выполните `docker compose down`.

Для запуска контейнера без Compose:

```bash
docker build -t recommender-app .
docker run --rm -p 8501:8501 --mount "type=bind,source=${PWD}/data,target=/app/data,readonly" recommender-app
```

Выполняйте команды из корня проекта. Каталог `data` должен существовать на момент запуска.

## Запуск без Docker

В PowerShell с установленным Python 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Автоматическая проверка

GitHub Actions при каждом push и обновлении pull request проверяет конфигурацию Compose, собирает образ, проверяет выдачу десяти рекомендаций на контрольных данных и проверяет HTTP-ответ запущенного Streamlit. Тест приложения можно запустить локально командой `python tests/smoke.py` из корня проекта после установки зависимостей.
