#!/bin/sh
set -e

python manage.py collectstatic --noinput
python manage.py migrate --noinput

exec gunicorn ask_me.wsgi:application --bind 0.0.0.0:${PORT:-8000}