import uuid
from typing import Collection, Optional

from django.db import models
from django.utils import timezone
from safedelete.models import HARD_DELETE, SafeDeleteModel


class ExternalIdMixin(models.Model):
    class Meta:
        abstract = True

    external_id = models.UUIDField(editable=False, default=uuid.uuid4, unique=True)

    def full_clean(
        self, exclude: Optional[Collection[str]] = None, validate_unique: bool = True
    ) -> None:
        exclude = exclude or []
        # print(exclude)
        # Do not perform unique check for external_id
        # exclude.append("external_id")
        super().full_clean(exclude, validate_unique)


class BaseModel(ExternalIdMixin, SafeDeleteModel):
    """Inherit from this when creating new models."""

    class Meta:
        abstract = True
        ordering = ("created",)

    _safedelete_policy = HARD_DELETE

    # These are for tracking the internal state of our models and to help with filtering
    created = models.DateTimeField(auto_now_add=True, db_index=True, editable=False)
    updated = models.DateTimeField(auto_now=True, db_index=True, editable=False)

    # Enable logical deletes.
    deleted = models.DateTimeField(null=True, blank=True, db_index=True, editable=False)

    def __str__(self):
        return repr(self)

    __REPR__ = ("id",)

    def __repr__(self):
        props = [f"{prop}={getattr(self, prop)!r}" for prop in self.__REPR__ if hasattr(self, prop)]
        return f'{self.__class__.__name__}({", ".join(props)})'

    def save(
        self,
        force_insert=False,
        force_update=False,
        using=None,
        update_fields=None,
        **kwargs,
    ):
        self.full_clean()

        # Manually set updated timestamp
        self.updated = timezone.now()

        # If using update_fields, make sure 'updated' is included
        if update_fields is not None and "updated" not in update_fields:
            update_fields = list(update_fields) + ["updated"]

        return super().save(
            force_insert=force_insert,
            force_update=force_update,
            using=using,
            update_fields=update_fields,
            **kwargs,
        )
