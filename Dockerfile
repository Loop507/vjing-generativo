FROM python:3.9
WORKDIR /app
COPY . /app
RUN apt-get update && apt-get install -y ffmpeg
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
