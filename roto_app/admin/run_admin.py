"""Django admin for runs.

Read-mostly by design. The fields that make a run reproducible are set once, by the request
that created it, and editing them afterwards would leave a row describing a run that never
happened -- so they are read-only here and the admin is a window rather than a control panel.
"""

from django.contrib import admin
from django_json_widget.widgets import JSONEditorWidget

from roto_app.models.run import Run


@admin.register(Run)
class RunAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "environment",
        "status",
        "stage",
        "dataset_version",
        "steps",
        "executor",
        "created",
        "completed_at",
    )
    list_filter = ("environment", "status", "stage", "rung", "dataset_version", "executor")
    search_fields = ("reference_id", "ecs_task_arn", "external_id")
    date_hierarchy = "created"
    ordering = ("-created",)

    readonly_fields = (
        "external_id",
        "reference_id",
        "rung",
        "dataset_version",
        "seed",
        "steps",
        "environment",
        "executor",
        "ecs_cluster",
        "ecs_task_arn",
        "created",
        "updated",
        "execution_started_at",
        "completed_at",
        "request_payload",
        "result_data",
        "error_logs",
        "time_logs",
        "run_logs",
    )

    def has_add_permission(self, request):
        """A run is created by launching one. A row with no task behind it is a lie."""
        return False

    formfield_overrides = {}

    def formfield_for_dbfield(self, db_field, request, **kwargs):
        from django.db import models

        if isinstance(db_field, models.JSONField):
            kwargs["widget"] = JSONEditorWidget
        return super().formfield_for_dbfield(db_field, request, **kwargs)

    @admin.display(description="run")
    def name(self, obj):
        return obj.name
