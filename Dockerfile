FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --only-binary=:all: -r requirements.txt \
    && useradd --create-home --uid 10001 streamlit \
    && mkdir /app/data \
    && chown streamlit:streamlit /app/data

COPY app.py .
USER streamlit
EXPOSE 8501
HEALTHCHECK --interval=5s --timeout=3s --start-period=10s --retries=6 \
    CMD ["python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2).close()"]
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.headless=true", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
