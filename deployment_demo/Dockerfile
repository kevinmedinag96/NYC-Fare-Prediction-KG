FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

WORKDIR /app

COPY ./model /app/model

COPY ./requirements.txt /app/src/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/src/requirements.txt

COPY ./src /app/src

#EXPOSE 80

CMD [ "uvicorn",  "src.api:app", "--host", "0.0.0.0","--port", "80", "--reload" ]
#correct cmd in terminal to run container -> docker run --name <cnt-name> -p 8000:80 <img-name>

#in web-> http://localhost:8000