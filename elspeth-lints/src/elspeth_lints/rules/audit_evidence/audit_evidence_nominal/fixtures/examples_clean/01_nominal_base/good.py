from elspeth.contracts.audit_evidence import AuditEvidenceBase


class Ok(AuditEvidenceBase, RuntimeError):
    def to_audit_dict(self):
        return {}
