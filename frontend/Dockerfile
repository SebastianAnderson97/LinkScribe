FROM python:3.11 as build-stage
WORKDIR /tmp
RUN pip install poetry
COPY ./pyproject.toml /tmp/
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

FROM python:3.11
RUN apt-get update

WORKDIR /app
COPY --from=build-stage /tmp/requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY . .

RUN pwd && ls -la 

EXPOSE 8501
CMD ["streamlit", "run", "home.py"]