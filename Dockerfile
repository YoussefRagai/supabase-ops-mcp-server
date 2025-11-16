FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY supabase_ops_server.py main.py ./

RUN mkdir -p /app/output && \
    useradd -m -u 1000 mcpuser && chown -R mcpuser:mcpuser /app

USER mcpuser

CMD ["python", "main.py"]
