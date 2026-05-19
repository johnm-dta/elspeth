from elspeth.contracts.errors import tier_1_error


@tier_1_error(reason="registered")
class WidgetError(Exception):
    pass
