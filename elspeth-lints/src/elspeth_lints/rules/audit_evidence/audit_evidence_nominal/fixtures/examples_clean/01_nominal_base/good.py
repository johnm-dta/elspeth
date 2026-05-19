class AuditEvidenceBase:
    pass


class Ok(AuditEvidenceBase, RuntimeError):
    def to_audit_dict(self):
        return {}
