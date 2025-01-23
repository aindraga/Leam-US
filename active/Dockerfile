FROM python:3.9

WORKDIR /app
ENV PIP_PROGRESS_BAR=off
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "leam_us_active_lig.py", "--config_filename=data/model/dcrnn_cov.yaml"]