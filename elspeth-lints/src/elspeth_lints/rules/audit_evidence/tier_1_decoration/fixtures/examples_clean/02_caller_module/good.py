from elspeth.contracts.errors import tier_1_error


@tier_1_error(reason="registered", caller_module=__name__)
class WidgetError(Exception):
    pass
