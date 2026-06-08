from elspeth.contracts.tier_registry import tier_1_error


@tier_1_error(caller_module=__name__)
class MissingReasonError(Exception):
    pass
