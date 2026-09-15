"""WSGI entry point for the roto training service."""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "roto_app.settings")

application = get_wsgi_application()
