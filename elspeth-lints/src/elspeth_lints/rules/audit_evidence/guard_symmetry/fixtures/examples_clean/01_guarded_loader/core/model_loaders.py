from contracts.audit import Widget


class AuditIntegrityError(Exception):
    pass


class WidgetLoader:
    def load(self, row):
        if row.size is None:
            raise AuditIntegrityError("missing size")
        return Widget(size=row.size)
