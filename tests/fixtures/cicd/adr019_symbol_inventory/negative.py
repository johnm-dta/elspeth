def clean(outcome, status):
    terminal = True
    payload = {"completed": True}
    if status in {"completed", "failed"}:
        return terminal
    return payload
