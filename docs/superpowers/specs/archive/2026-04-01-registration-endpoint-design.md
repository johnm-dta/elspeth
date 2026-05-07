# Registration Endpoint Design

**Date:** 2026-04-01
**Scope:** Add `POST /api/auth/register` to the local auth provider with configurable registration modes.

## Problem

`LocalAuthProvider.create_user()` exists but is not wired to any HTTP endpoint. Creating a user requires direct Python scripting against the auth database. There is no self-service registration flow.

## Design

### Configuration

New field in `WebSettings` (`src/elspeth/web/config.py`):

```python
registration_mode: Literal["open", "email_verified", "closed"] = "open"
```

- **`open`** — anyone can register without authentication. Default for development.
- **`email_verified`** — reserved for future email verification flow. Returns 501 Not Implemented until an email transport is configured.
- **`closed`** — registration disabled. Used alongside OIDC/Entra where user management is external.

### Route

New `POST /api/auth/register` in `src/elspeth/web/auth/routes.py`.

**Request model:**

```python
class RegisterRequest(BaseModel):
    username: str
    password: str
    display_name: str
    email: str | None = None
```

**Behavior by registration mode:**

| `registration_mode` | `auth_provider == "local"` | Response |
|---------------------|---------------------------|----------|
| `open`              | yes                       | Create user, auto-login, return `TokenResponse` |
| `email_verified`    | yes                       | 501 Not Implemented |
| `closed`            | yes                       | 404 Not Found |
| any                 | no                        | 404 Not Found |

**Error responses:**

| Condition | Status | Detail |
|-----------|--------|--------|
| Duplicate username | 409 Conflict | "User already exists: {username}" |
| Empty username or password | 422 Unprocessable Entity | Pydantic validation or explicit guard |
| Empty display_name | 409 Conflict | Surfaced from `create_user()` ValueError |

**Threading:** `create_user()` is synchronous (bcrypt hashing ~200ms). Wrap in `asyncio.to_thread()` to avoid blocking the event loop. `provider.login()` already handles this internally.

### Auth config disclosure

Add `registration_mode` to the existing `AuthConfigResponse` so the frontend can discover whether to show a registration form:

```python
class AuthConfigResponse(BaseModel):
    provider: str
    registration_mode: str
    oidc_issuer: str | None = None
    oidc_client_id: str | None = None
    authorization_endpoint: str | None = None
```

### Testing

- Registration succeeds in `open` mode — returns 200 with `TokenResponse`
- Registration returns 501 in `email_verified` mode
- Registration returns 404 in `closed` mode
- Registration returns 404 when `auth_provider != "local"` regardless of mode
- Duplicate username returns 409
- Empty username or password returns 422
- `GET /api/auth/config` includes `registration_mode`

## Files changed

| File | Change |
|------|--------|
| `src/elspeth/web/config.py` | Add `registration_mode` field |
| `src/elspeth/web/auth/routes.py` | Add `RegisterRequest` model, `/register` route, update `AuthConfigResponse` |
| `tests/` (TBD) | Unit tests for all modes and error cases |

## Out of scope

- Email verification flow (deferred; `email_verified` mode returns 501)
- Rate limiting on registration (existing composer rate limiter covers the pattern if needed later)
- CLI command for user creation (existing `create_user()` method is sufficient for scripting)
