# CPU inspection service: OpenVINO INT8 models, no torch. Build after `python bench/deploy.py`.
FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODELS_DIR=/app/deploy
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY edgeinspect/__init__.py edgeinspect/serving.py edgeinspect/roi.py edgeinspect/
COPY serve/ serve/
COPY deploy/ deploy/
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,sys; sys.exit(0 if b'ok' in urllib.request.urlopen('http://localhost:8000/health').read() else 1)"
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
