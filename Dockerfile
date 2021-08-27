FROM python:3.6
WORKDIR /app
COPY src /app/src
COPY run_analysis.py requirements.txt /app/
RUN pip install -r requirements.txt
ENTRYPOINT ["python", "run_analysis.py"]