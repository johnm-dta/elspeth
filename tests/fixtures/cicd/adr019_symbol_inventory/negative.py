def clean(outcome):
    terminal = True
    payload = {"completed": True}
    if outcome in {"completed", "failed"}:
        return terminal
    return payload
