#!/bin/sh
set -e

echo "Running collectstatic..."
python manage.py collectstatic --noinput

echo "Running migrations..."
python manage.py migrate --noinput

# Render sets $PORT; locally we fall back to 8000.
# Use ${PORT:-8000} so this works in both environments without special-casing.
BIND_PORT="${PORT:-8000}"

echo "Starting Gunicorn on 0.0.0.0:${BIND_PORT}..."
exec gunicorn ask_me.wsgi:application \
    --bind "0.0.0.0:${BIND_PORT}" \
    --workers 2 \
    --threads 2 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile -
