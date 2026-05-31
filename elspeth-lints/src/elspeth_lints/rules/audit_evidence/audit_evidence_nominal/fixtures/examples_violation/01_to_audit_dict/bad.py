class Mimic(RuntimeError):
    def to_audit_dict(self):
        return {}
