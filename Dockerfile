# API + dashboard image (works on Hugging Face Spaces [Docker], Render, Fly, or locally)
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --user -r requirements.txt
COPY --chown=user . .
# bake a small demo model set so the public demo works without an external database
RUN python -m csu.cli run-all --customers 1500 --no-mlflow
EXPOSE 7860
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
