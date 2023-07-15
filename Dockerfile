FROM docker.jfrog.booking.com/projects/content-ml/streamlit:1.1.0

# copy the application code
COPY main.py main.py

CMD ["streamlit", "run", "main.py"]
