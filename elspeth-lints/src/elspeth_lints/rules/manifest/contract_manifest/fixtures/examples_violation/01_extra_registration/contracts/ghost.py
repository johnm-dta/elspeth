from elspeth.contracts.declaration_contracts import (
    DeclarationContract,
    implements_dispatch_site,
    register_declaration_contract,
)


class GhostContract(DeclarationContract):
    name = "ghost_not_in_manifest"

    @implements_dispatch_site("post_emission_check")
    def post_emission_check(self, inputs, outputs):
        raise NotImplementedError


register_declaration_contract(GhostContract())
