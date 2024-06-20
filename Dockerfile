FROM python:3.8

WORKDIR /usr/src/app

# Do not write pyc files: ----
ENV PYTHONDONTWRITEBYTECODE 1
# Do not buffer to stdout/stderr: ----
ENV PYTHONUNBUFFERED 1

COPY ./requirements.txt /usr/src/app/requirements.txt

# Install the requirements with all depencies:
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && rm -rf /root/.cache/pip

COPY ./ /usr/src/app