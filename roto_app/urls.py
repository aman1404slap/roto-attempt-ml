"""URL configuration for the roto training service.

``URL_PREFIX`` mirrors the sibling services: behind an ALB the service is routed by path, so
the same patterns are mounted twice -- bare for direct access and health checks, prefixed for
the load balancer.
"""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

from roto_app.views import views

api_url_patterns = [
    path("runs", views.create_run, name="create_run"),
    path("runs/list", views.list_runs, name="list_runs"),
    path("runs/<uuid:run_id>", views.run_status, name="run_status"),
    path("runs/<uuid:run_id>/stop", views.stop_run, name="stop_run"),
    path("seed-data", views.seed_data, name="seed_data"),
]

base_url_patterns = [
    path("admin/", admin.site.urls),
    path("health-check", views.health_check, name="health_check"),
    path("api/", include(api_url_patterns)),
]

urlpatterns = base_url_patterns.copy()

if getattr(settings, "URL_PREFIX", ""):
    urlpatterns = [path(f"{settings.URL_PREFIX}", include(base_url_patterns))] + urlpatterns

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
