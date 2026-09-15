"""Django settings for the roto training service.

The service is orchestration and nothing else: it accepts a run, launches it, and reports what
happened. The science lives in ``src/roto`` and is a plain package with no Django in it, which
is what keeps ``pytest`` and the existing scripts working unchanged and what makes the training
path independent of this one.

Config follows the sibling services: ``django-environ``, every value from the environment with
a default, no credentials in the file.
"""

import logging
import os
import sys

import environ
import sentry_sdk
from sentry_sdk.integrations.django import DjangoIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

root = environ.Path(__file__) - 2
DEFAULT_ENV_FILE = os.path.join(root(), ".env")

# ``src`` on the path, not ``pip install -e .``. The training package is imported the same way
# in every context -- here, in ``scripts/*.py``, in the image (which sets PYTHONPATH=/code/src)
# and in pytest (which sets ``pythonpath = ["src"]``) -- so there is no installed copy that can
# drift from the working tree, and no state on a machine that a fresh checkout would not have.
_SRC = os.path.join(root(), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

env = environ.Env(
    ENV=(str, "local"),
    DJANGO_DEBUG=(bool, False),
    SECRET_KEY=(str, ""),
    DATABASE_URL=(str, "postgres://roto:roto@localhost:5432/roto"),
    AWS_DEFAULT_REGION=(str, "us-east-2"),
    # Our bucket: read/write, everything we derive goes here.
    AWS_DEFAULT_BUCKET=(str, ""),
    # The archive, in another account. Read only, and nothing ever writes to it.
    SOURCE_S3_BUCKET=(str, "production-citadel"),
    SOURCE_S3_PREFIX=(str, ""),
    STORAGE_ROOT=(str, ""),
    SOURCE_ROOT=(str, ""),
    ROTO_ENVIRONMENT=(str, ""),
    RUN_EXECUTOR=(str, ""),
    SCRATCH_DIR=(str, ""),
    LOCAL_USER=(str, ""),
    DEFAULT_DATASET_VERSION=(str, "v003"),
    ECS_RUN_TASK_CLUSTER_NAME=(str, ""),
    ECS_RUN_TASK_CONTAINER_NAME=(str, "roto"),
    ECS_RUN_TASK_DEFINITION=(str, ""),
    ECS_RUN_TASK_SUBNET_IDS=(str, ""),
    ECS_RUN_TASK_SECURITY_GROUP_IDS=(str, ""),
    ECS_TASK_LOG_GROUP=(str, ""),
    SENTRY_DSN=(str, ""),
    ROTO_APP_TASK_API_KEY=(str, ""),
    URL_PREFIX=(str, ""),
)

environ.Env.read_env(env.str("ENV_PATH", DEFAULT_ENV_FILE))

ENV = env("ENV")
BASE_DIR = root()

SECRET_KEY = env("SECRET_KEY")
DEBUG = env("DJANGO_DEBUG")

ALLOWED_HOSTS = ["*"]
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
USE_X_FORWARDED_HOST = True
USE_X_FORWARDED_PORT = True

INSTALLED_APPS = [
    "roto_app",
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django_linear_migrations",
    "django_json_widget",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "roto_app.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "roto_app.wsgi.application"

DATABASES = {"default": env.db()}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/roto/static/"
STATIC_ROOT = os.path.join(BASE_DIR, "staticfiles")

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# --- AWS ---------------------------------------------------------------------------------

AWS_DEFAULT_REGION = env("AWS_DEFAULT_REGION")

AWS_DEFAULT_BUCKET = env("AWS_DEFAULT_BUCKET")
"""Our bucket, in our account, read/write. Datasets and every run land here."""

SOURCE_S3_BUCKET = env("SOURCE_S3_BUCKET")
SOURCE_S3_PREFIX = env("SOURCE_S3_PREFIX")
"""The archive, in another account, read only. Only the dataset build reads it; training never
touches it, which is why the GPU task never has to hold client footage."""


# --- Where datasets and runs are stored ---------------------------------------------------

STORAGE_ROOT = env("STORAGE_ROOT") or (
    f"s3://{AWS_DEFAULT_BUCKET}" if ENV != "local" and AWS_DEFAULT_BUCKET else "abc"
)
"""The root everything derived is read from and written to.

``s3://<bucket>`` in staging; a gitignored folder -- ``abc`` -- locally, because until the
infrastructure exists there is no bucket to write to and waiting for one would mean the service
could not be run at all. **The layout inside is identical either way** (``datasets/<version>/``,
``runs/staging/<run>/``, ``runs/local/<user>/<run>/``), so local is staging with a different
root rather than a second design, and switching back is this one variable.

Set explicitly to override -- a local process can point at the real bucket once it exists, and
it should be possible to say so without also claiming to be staging."""

SOURCE_ROOT = env("SOURCE_ROOT") or (
    f"s3://{SOURCE_S3_BUCKET}/{SOURCE_S3_PREFIX}".rstrip("/")
    if ENV != "local"
    else "data/spline_dataset_08_25_26"
)
"""The archive the dataset build reads. Read-only, and nothing here ever writes to it.

The cross-account bucket once that read exists; the delivery already on disk until then."""


# --- How a run is executed ---------------------------------------------------------------

ROTO_ENVIRONMENT = env("ROTO_ENVIRONMENT") or ("staging" if ENV != "local" else "local")
"""Stamped onto every run and used to pick its S3 prefix. ``local`` is a smoke test whose
number can never be quoted -- ``roto.v2.provenance`` enforces that at the scoring end, and this
is where the value comes from."""

RUN_EXECUTOR = env("RUN_EXECUTOR") or ("local" if ENV == "local" else "ecs")
"""``ecs`` launches a GPU task; ``local`` runs the same management command as a subprocess on
this machine. The command it runs is identical, so a local run is a staging run with a smaller
argument list rather than a different code path."""

SCRATCH_DIR = env("SCRATCH_DIR") or (
    os.path.join(BASE_DIR, ".scratch") if ENV == "local" else "/scratch"
)
"""Where datasets are synced to and runs are written before being synced back. Local disk in
both environments -- the training code reads plain paths and nothing about that changes when
the data arrives from S3."""

LOCAL_USER = env("LOCAL_USER") or os.environ.get("USER") or "unknown"
"""Namespaces ``runs/local/<user>/`` so two people smoke-testing do not overwrite each other."""

DEFAULT_DATASET_VERSION = env("DEFAULT_DATASET_VERSION")

ECS_RUN_TASK_CLUSTER_NAME = env("ECS_RUN_TASK_CLUSTER_NAME")
ECS_RUN_TASK_CONTAINER_NAME = env("ECS_RUN_TASK_CONTAINER_NAME")
ECS_RUN_TASK_DEFINITION = env("ECS_RUN_TASK_DEFINITION")
ECS_RUN_TASK_SUBNET_IDS = env("ECS_RUN_TASK_SUBNET_IDS")
ECS_RUN_TASK_SECURITY_GROUP_IDS = env("ECS_RUN_TASK_SECURITY_GROUP_IDS")
ECS_TASK_LOG_GROUP = env("ECS_TASK_LOG_GROUP")


# --- Auth, logging, errors ---------------------------------------------------------------

ROTO_APP_TASK_API_KEY = env("ROTO_APP_TASK_API_KEY")
URL_PREFIX = env("URL_PREFIX")

SENTRY_DSN = env("SENTRY_DSN")
if ENV != "local":
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        integrations=[DjangoIntegration(), LoggingIntegration(level=logging.INFO)],
        environment=ENV,
        send_default_pii=True,
    )

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "root": {"handlers": ["console"], "level": "INFO"},
    "loggers": {
        "django": {"handlers": ["console"], "level": "INFO", "propagate": True},
        "django.request": {"handlers": ["console"], "level": "INFO", "propagate": False},
    },
}
