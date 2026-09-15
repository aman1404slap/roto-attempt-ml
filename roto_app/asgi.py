"""ASGI entry point for the roto training service."""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "roto_app.settings")

application = get_asgi_application()
