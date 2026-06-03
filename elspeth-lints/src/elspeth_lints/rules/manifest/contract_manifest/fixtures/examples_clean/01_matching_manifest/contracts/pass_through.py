from elspeth.contracts.declaration_contracts import (
    DeclarationContract,
    implements_dispatch_site,
    register_declaration_contract,
)


class PassThroughContract(DeclarationContract):
    name = "passes_through_input"

    @implements_dispatch_site("post_emission_check")
    def post_emission_check(self, inputs, outputs):
        raise NotImplementedError


register_declaration_contract(PassThroughContract())
