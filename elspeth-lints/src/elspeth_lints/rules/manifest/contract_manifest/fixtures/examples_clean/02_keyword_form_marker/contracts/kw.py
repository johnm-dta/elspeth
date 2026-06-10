from elspeth.contracts.declaration_contracts import (
    DeclarationContract,
    implements_dispatch_site,
    register_declaration_contract,
)


class KeywordContract(DeclarationContract):
    name = "kw"

    @implements_dispatch_site(site_name="post_emission_check")
    def post_emission_check(self, inputs, outputs):
        raise NotImplementedError


register_declaration_contract(KeywordContract())
