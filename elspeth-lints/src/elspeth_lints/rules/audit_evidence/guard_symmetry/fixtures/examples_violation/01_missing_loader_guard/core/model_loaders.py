from contracts.audit import Widget


class WidgetLoader:
    def load(self, row):
        return Widget(size=row.size)
