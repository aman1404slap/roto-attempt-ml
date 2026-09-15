"""Small shared helpers: response shape, API-key auth, timing."""

import os

from django.conf import settings
from django.http import JsonResponse

STATUS_CODE_200 = 200
STATUS_CODE_400 = 400
STATUS_CODE_401 = 401
STATUS_CODE_404 = 404


def format_response(message="", status_code=STATUS_CODE_200, data=None):
    return JsonResponse({"message": message, "data": data}, status=status_code)


def authenticate(request):
    """Shared-secret auth, matching the sibling services' header.

    This service can start GPU instances, so an unauthenticated ``POST /api/runs`` is a way to
    spend money. The key is required in every environment including local -- an auth path that
    is only exercised in production is an auth path nobody has tested.
    """
    api_key = request.headers.get("x-task-api-key", None)
    if not api_key or api_key != settings.ROTO_APP_TASK_API_KEY:
        raise PermissionError("Invalid API key")


def time_difference_in_seconds(end_time, start_time):
    return round((end_time - start_time).total_seconds(), 2)


def ensure_suffix(string, suffix):
    return string if string.endswith(suffix) else f"{string}{suffix}"


def ensure_prefix(string, prefix):
    return string if string.startswith(prefix) else f"{prefix}{string}"


def free_disk_gb(path="/") -> float:
    """Free space where datasets and runs are written.

    Worth logging rather than assuming: a 1,000-shot dataset is ~1.9 GB derived and the raw
    archive it is built from is 11 GB for 50 shots, so the build step is the one that runs out
    of disk, and it does so a long way into the job.
    """
    st = os.statvfs(path)
    return round(st.f_bavail * st.f_frsize / 1024**3, 2)
