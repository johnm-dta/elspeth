from elspeth_lints.rules.plugin_contract.plugin_hashes.rule import PluginHashesRule

# Fixture trees hold exactly one plugin; the shipped rule's whole-repo
# discovery floor (min_plugins=37) would mask every real assertion here.
RULE = PluginHashesRule(min_plugins=1)
