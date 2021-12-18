FROM tiangolo/uwsgi-nginx-flask:python3.7

COPY requirements.txt ./


RUN apt install curl
RUN curl -sL https://deb.nodesource.com/setup_6.x | bash
RUN apt-get install -y nodejs
RUN apt-get install -y npm

COPY package.json ./
COPY package-lock.json ./
RUN npm install

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./app.py" ]
