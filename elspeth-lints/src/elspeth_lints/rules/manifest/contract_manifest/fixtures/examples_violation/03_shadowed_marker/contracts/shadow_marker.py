from elspeth.contracts.declaration_contracts import (
    DeclarationContract,
    register_declaration_contract,
)


def implements_dispatch_site(site):
    def decorate(fn):
        return fn

    return decorate


class ShadowMarkerContract(DeclarationContract):
    name = "shadow_marker"

    @implements_dispatch_site("post_emission_check")
    def post_emission_check(self, inputs, outputs):
        raise NotImplementedError


register_declaration_contract(ShadowMarkerContract())
