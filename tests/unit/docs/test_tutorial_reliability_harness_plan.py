from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
STAGING_CREDENTIAL_PLACEHOLDERS = ("dta_user", "dta_pass")


def test_public_docs_do_not_embed_staging_credential_placeholders() -> None:
    offenders: list[str] = []

    for path in [*REPO_ROOT.glob("*.md"), *REPO_ROOT.glob("docs/**/*.md")]:
        text = path.read_text(encoding="utf-8")
        for placeholder in STAGING_CREDENTIAL_PLACEHOLDERS:
            if placeholder in text:
                offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()}: {placeholder}")

    assert offenders == []
