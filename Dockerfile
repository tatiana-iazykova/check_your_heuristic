FROM tiangolo/uwsgi-nginx-flask:python3.7

COPY requirements.txt ./

COPY package.json ./
COPY package-lock.json ./

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./app.py" ]
