from elspeth.contracts.declaration_contracts import (
    DeclarationContract,
    implements_dispatch_site,
)


def register_declaration_contract(contract):
    return None


class ShadowContract(DeclarationContract):
    name = "shadowed"

    @implements_dispatch_site("post_emission_check")
    def post_emission_check(self, inputs, outputs):
        raise NotImplementedError


register_declaration_contract(ShadowContract())
