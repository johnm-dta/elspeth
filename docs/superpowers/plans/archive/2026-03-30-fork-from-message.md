# Fork From Message Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users edit a prior message and branch the conversation from that point, creating a new session with inherited history and composition state — without mutating the original.

**Architecture:** Add `composition_state_id` to `chat_messages` for per-message state provenance. Add `forked_from_session_id` and `forked_from_message_id` to `sessions`. Fork creates a new session by copying messages up to the fork point, copying the composition state from that point, and copying referenced blobs (honoring quota). The original session is never mutated.

**Tech Stack:** SQLAlchemy Core, Alembic, FastAPI, Zustand, existing SessionServiceImpl/BlobServiceImpl patterns.

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `src/elspeth/web/sessions/migrations/versions/004_add_fork_support.py` | Migration: 3 new columns |
| `tests/unit/web/sessions/test_fork.py` | Fork endpoint + service tests |

### Modified Files

| File | Change |
|------|--------|
| `src/elspeth/web/sessions/models.py` | 3 new columns on 2 tables |
| `src/elspeth/web/sessions/protocol.py` | Add fields to SessionRecord, ChatMessageRecord; add fork_session to protocol |
| `src/elspeth/web/sessions/schemas.py` | Fork request/response models; extend SessionResponse, SendMessageRequest |
| `src/elspeth/web/sessions/service.py` | Modify create_session, add_message; add fork_session method |
| `src/elspeth/web/sessions/routes.py` | Add fork endpoint; update send_message to record state provenance |
| `src/elspeth/web/blobs/service.py` | Add copy_blobs_for_fork method |
| `src/elspeth/web/blobs/protocol.py` | Add copy_blobs_for_fork to protocol |
| `src/elspeth/web/frontend/src/types/index.ts` | Extend Session, ChatMessage types |
| `src/elspeth/web/frontend/src/api/client.ts` | Add forkFromMessage function |
| `src/elspeth/web/frontend/src/stores/sessionStore.ts` | Add forkFromMessage action |
| `src/elspeth/web/frontend/src/components/chat/MessageBubble.tsx` | Add fork button on user messages |
| `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx` | Wire fork handler |
| `src/elspeth/web/frontend/src/components/sessions/SessionSidebar.tsx` | Show fork provenance |

---

## Task 1: Schema migration — 3 new columns

**Files:**
- Modify: `src/elspeth/web/sessions/models.py`
- Create: `src/elspeth/web/sessions/migrations/versions/004_add_fork_support.py`

- [ ] **Step 1: Add columns to models.py**

Add to `sessions_table`:
```python
Column("forked_from_session_id", String, ForeignKey("sessions.id"), nullable=True),
Column("forked_from_message_id", String, nullable=True),
```

Add to `chat_messages_table`:
```python
Column("composition_state_id", String, ForeignKey("composition_states.id"), nullable=True),
```

- [ ] **Step 2: Create migration 004**

```python
revision: str = "004"
down_revision: str | Sequence[str] | None = "003"

def upgrade() -> None:
    with op.batch_alter_table("sessions") as batch_op:
        batch_op.add_column(sa.Column("forked_from_session_id", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("forked_from_message_id", sa.String(), nullable=True))
        batch_op.create_foreign_key(
            "fk_sessions_forked_from", "sessions", ["forked_from_session_id"], ["id"]
        )

    with op.batch_alter_table("chat_messages") as batch_op:
        batch_op.add_column(sa.Column("composition_state_id", sa.String(), nullable=True))
        batch_op.create_foreign_key(
            "fk_chat_messages_state", "composition_states",
            ["composition_state_id"], ["id"]
        )
```

Note: `batch_alter_table` is required for SQLite ALTER TABLE (render_as_batch=True in env.py).

- [ ] **Step 3: Verify migration runs**

```bash
python -c "
import tempfile, os
from sqlalchemy import create_engine, inspect
from elspeth.web.sessions.migrations import run_migrations
with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
    db_path = f.name
try:
    engine = create_engine(f'sqlite:///{db_path}')
    run_migrations(engine)
    cols = {c['name'] for c in inspect(engine).get_columns('chat_messages')}
    assert 'composition_state_id' in cols
    sess_cols = {c['name'] for c in inspect(engine).get_columns('sessions')}
    assert 'forked_from_session_id' in sess_cols
    print('PASS')
finally:
    os.unlink(db_path)
"
```

- [ ] **Step 4: Commit**

```
feat(web/sessions): add fork support columns — migration 004
```

---

## Task 2: Update protocol and service — SessionRecord, ChatMessageRecord, add_message

**Files:**
- Modify: `src/elspeth/web/sessions/protocol.py`
- Modify: `src/elspeth/web/sessions/service.py`
- Modify: `src/elspeth/web/sessions/schemas.py`

- [ ] **Step 1: Write test for add_message with composition_state_id**

```python
# In tests/unit/web/sessions/ (find existing test file)
@pytest.mark.asyncio
async def test_add_message_records_composition_state_id(session_service, session_id):
    state = await session_service.save_composition_state(session_id, CompositionStateData())
    msg = await session_service.add_message(
        session_id, "user", "hello",
        composition_state_id=state.id,
    )
    assert msg.composition_state_id == state.id

@pytest.mark.asyncio
async def test_add_message_without_state_id_is_none(session_service, session_id):
    msg = await session_service.add_message(session_id, "user", "hello")
    assert msg.composition_state_id is None
```

- [ ] **Step 2: Update ChatMessageRecord**

In `protocol.py`:
```python
@dataclass(frozen=True, slots=True)
class ChatMessageRecord:
    id: UUID
    session_id: UUID
    role: str
    content: str
    tool_calls: Mapping[str, Any] | None
    created_at: datetime
    composition_state_id: UUID | None = None  # NEW

    def __post_init__(self) -> None:
        if self.tool_calls is not None:
            freeze_fields(self, "tool_calls")
```

- [ ] **Step 3: Update SessionRecord**

```python
@dataclass(frozen=True, slots=True)
class SessionRecord:
    id: UUID
    user_id: str
    auth_provider_type: str
    title: str
    created_at: datetime
    updated_at: datetime
    forked_from_session_id: UUID | None = None  # NEW
    forked_from_message_id: UUID | None = None  # NEW
```

- [ ] **Step 4: Update add_message in SessionServiceImpl**

Add `composition_state_id: UUID | None = None` parameter. Pass it through to the INSERT. Update the row-to-record converter to read the new column.

- [ ] **Step 5: Update create_session in SessionServiceImpl**

Add `forked_from_session_id: UUID | None = None` and `forked_from_message_id: UUID | None = None` parameters.

- [ ] **Step 6: Update schemas**

```python
class SessionResponse(BaseModel):
    id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    forked_from_session_id: str | None = None
    forked_from_message_id: str | None = None

class ChatMessageResponse(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    tool_calls: Any | None = None
    created_at: datetime
    composition_state_id: str | None = None
```

- [ ] **Step 7: Run tests, verify pass**

- [ ] **Step 8: Commit**

```
feat(web/sessions): add composition_state_id to messages, fork fields to sessions
```

---

## Task 3: Message provenance — record state ID on send

**Files:**
- Modify: `src/elspeth/web/sessions/routes.py`
- Modify: `src/elspeth/web/sessions/schemas.py`

- [ ] **Step 1: Write test for state provenance**

```python
# Test that user message gets pre-send state_id
# Test that assistant message gets post-compose state_id
# Test that NULL composition_state_id on new messages is caught (AD-3)
```

- [ ] **Step 2: Update SendMessageRequest**

Add optional `state_version: int | None = None` field. The client sends the composition state version it was looking at when the user typed the message. The server uses this to find the corresponding state_id.

```python
class SendMessageRequest(BaseModel):
    content: str
    state_version: int | None = None  # client-observed state version
```

- [ ] **Step 3: Update send_message route handler**

After loading composition state (line ~192), capture the pre-send state_id:

```python
# Pre-send state: what the user was looking at
pre_send_state_id = state_record.id if state_record else None

# Persist user message with pre-send provenance
user_msg = await service.add_message(
    session.id, "user", body.content,
    composition_state_id=pre_send_state_id,
)
```

After the composer loop, persist the assistant message with post-turn provenance:

```python
# Post-compose state: what exists after the assistant's tool calls
post_compose_state_id = new_state_record.id if new_state_record else pre_send_state_id

assistant_msg = await service.add_message(
    session.id, "assistant", result.message,
    tool_calls=result.tool_calls,
    composition_state_id=post_compose_state_id,
)
```

If `body.state_version` is provided, verify it matches the current state version before proceeding. If mismatched, return 409 (stale state).

- [ ] **Step 4: Run tests, verify pass**

- [ ] **Step 5: Commit**

```
feat(web/sessions): record composition state provenance on every message
```

---

## Task 4: Blob copy-on-fork

**Files:**
- Modify: `src/elspeth/web/blobs/service.py`
- Modify: `src/elspeth/web/blobs/protocol.py`
- Test: `tests/unit/web/blobs/test_service.py`

- [ ] **Step 1: Write test for copy_blobs_for_fork**

```python
@pytest.mark.asyncio
async def test_copy_blobs_for_fork(blob_service, db_engine, session_id, tmp_path):
    # Create a blob in session A
    blob = await blob_service.create_blob(
        session_id=session_id, filename="data.csv",
        content=b"a,b,c\n1,2,3", mime_type="text/csv",
    )

    # Create session B
    s2_id = UUID(str(uuid4()))
    # ... insert session B row ...

    # Copy blobs
    copied = await blob_service.copy_blobs_for_fork(session_id, s2_id)
    assert len(copied) == 1
    assert copied[0].session_id == s2_id
    assert copied[0].filename == "data.csv"
    assert copied[0].id != blob.id  # new ID

    # Verify file exists at new path
    content = await blob_service.read_blob_content(copied[0].id)
    assert content == b"a,b,c\n1,2,3"

@pytest.mark.asyncio
async def test_copy_blobs_respects_quota(blob_service_small_quota, ...):
    # Create blobs that fill the quota
    # Fork should fail with BlobQuotaExceededError
```

- [ ] **Step 2: Implement copy_blobs_for_fork**

```python
async def copy_blobs_for_fork(
    self,
    source_session_id: UUID,
    target_session_id: UUID,
) -> list[BlobRecord]:
    """Copy all ready blobs from source session to target session.

    Creates new blob records with new IDs and new storage paths.
    Copies backing files to the new session's blob directory.
    Respects the per-session quota.
    """
```

Implementation: list source session's ready blobs, for each: read content, call create_blob on the target session. The create_blob method already handles quota checks, file writes, and hash computation.

- [ ] **Step 3: Add to protocol**

- [ ] **Step 4: Run tests, verify pass**

- [ ] **Step 5: Commit**

```
feat(web/blobs): add copy_blobs_for_fork for session forking
```

---

## Task 5: Fork endpoint

**Files:**
- Modify: `src/elspeth/web/sessions/service.py`
- Modify: `src/elspeth/web/sessions/routes.py`
- Modify: `src/elspeth/web/sessions/schemas.py`
- Create: `tests/unit/web/sessions/test_fork.py`

- [ ] **Step 1: Write fork tests**

Critical tests:
- `test_fork_creates_new_session_with_provenance` — new session has forked_from fields set
- `test_fork_copies_messages_up_to_fork_point` — messages after fork point excluded
- `test_fork_copies_composition_state_at_fork_point` — state matches the user message's pre-send state, not latest
- `test_fork_preserves_original_session` — original session unchanged after fork
- `test_fork_from_wrong_session_returns_404` — IDOR protection
- `test_fork_from_nonexistent_message_returns_404`
- `test_fork_copies_blobs` — blob files duplicated into new session
- `test_fork_respects_blob_quota` — fails gracefully if quota exceeded

- [ ] **Step 2: Add ForkMessageRequest and ForkMessageResponse to schemas**

```python
class ForkMessageRequest(BaseModel):
    new_message_content: str  # the edited message text

class ForkMessageResponse(BaseModel):
    session: SessionResponse
    messages: list[ChatMessageResponse]
    composition_state: CompositionStateResponse | None = None
```

- [ ] **Step 3: Add fork_session to SessionServiceImpl**

```python
async def fork_session(
    self,
    source_session_id: UUID,
    fork_message_id: UUID,
    new_message_content: str,
    user_id: str,
    auth_provider_type: str,
) -> tuple[SessionRecord, list[ChatMessageRecord], CompositionStateRecord | None]:
```

Implementation:
1. Load source session messages ordered by created_at
2. Find the fork message — must be a user message
3. Get the fork message's `composition_state_id` (the pre-send state)
4. Create new session with `forked_from_session_id` and `forked_from_message_id`
5. Copy all messages BEFORE the fork message into the new session
6. Add the new edited user message (with the copied state as its provenance)
7. If composition state exists at fork point, copy it as the new session's current state (new version 1, `derived_from_state_id` = original state ID)
8. Return the new session, its messages, and the copied state

- [ ] **Step 4: Add fork route handler**

```python
@router.post("/{session_id}/fork", status_code=201, response_model=ForkMessageResponse)
async def fork_from_message(
    session_id: UUID,
    body: ForkMessageRequest,
    request: Request,
    user: UserIdentity = Depends(get_current_user),
) -> ForkMessageResponse:
```

Wait — the sub-plan says the fork point is a message ID, but the request needs both the message ID and the new content. Options:

Option A: `POST /api/sessions/{session_id}/fork` with body `{from_message_id, new_message_content}`
Option B: `POST /api/sessions/{session_id}/messages/{message_id}/fork` with body `{new_message_content}`

Option A is cleaner — one endpoint, explicit body. Go with:

```python
class ForkSessionRequest(BaseModel):
    from_message_id: str
    new_message_content: str

@router.post("/{session_id}/fork", status_code=201, response_model=ForkMessageResponse)
```

- [ ] **Step 5: Wire blob copy into fork**

After creating the new session and copying messages/state, call `blob_service.copy_blobs_for_fork(source_session_id, new_session.id)`. Catch `BlobQuotaExceededError` — if quota is exceeded during fork, clean up (delete new session via cascade) and return 413.

- [ ] **Step 6: Run tests, verify pass**

- [ ] **Step 7: Commit**

```
feat(web/sessions): add fork endpoint — branch conversation from any user message
```

---

## Task 6: Frontend — fork flow

**Files:**
- Modify: `src/elspeth/web/frontend/src/types/index.ts`
- Modify: `src/elspeth/web/frontend/src/api/client.ts`
- Modify: `src/elspeth/web/frontend/src/stores/sessionStore.ts`
- Modify: `src/elspeth/web/frontend/src/components/chat/MessageBubble.tsx`
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx`
- Modify: `src/elspeth/web/frontend/src/components/sessions/SessionSidebar.tsx`

- [ ] **Step 1: Update types**

```typescript
export interface Session {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  forked_from_session_id?: string;
  forked_from_message_id?: string;
}

export interface ChatMessage {
  // ... existing fields ...
  composition_state_id?: string;
}
```

- [ ] **Step 2: Add API function**

```typescript
export async function forkFromMessage(
  sessionId: string,
  fromMessageId: string,
  newMessageContent: string,
): Promise<{ session: Session; messages: ChatMessage[]; composition_state: CompositionState | null }> {
  const response = await fetch(`/api/sessions/${sessionId}/fork`, {
    method: "POST",
    headers: authHeaders("application/json"),
    body: JSON.stringify({ from_message_id: fromMessageId, new_message_content: newMessageContent }),
  });
  return parseResponse<...>(response);
}
```

- [ ] **Step 3: Add forkFromMessage to sessionStore**

```typescript
forkFromMessage: async (messageId: string, newContent: string) => {
  const { activeSessionId } = get();
  if (!activeSessionId) return;

  set({ isComposing: true, error: null });
  try {
    const result = await api.forkFromMessage(activeSessionId, messageId, newContent);
    // Add the new session to the sidebar list
    set((state) => ({
      sessions: [result.session, ...state.sessions],
      activeSessionId: result.session.id,
      messages: result.messages,
      compositionState: result.composition_state,
      isComposing: false,
    }));
  } catch (err) {
    set({ isComposing: false, error: "Failed to fork conversation." });
  }
};
```

- [ ] **Step 4: Add edit/fork UI to MessageBubble**

For user messages only: add an edit icon (pencil) that, when clicked, replaces the bubble text with a textarea pre-filled with the original message content. A "Fork" submit button creates the fork. A "Cancel" button dismisses the edit mode.

Props needed: `onFork?: (messageId: string, newContent: string) => void`

- [ ] **Step 5: Wire fork handler in ChatPanel**

```typescript
const handleFork = useCallback(
  (messageId: string, newContent: string) => {
    useSessionStore.getState().forkFromMessage(messageId, newContent);
  },
  [],
);
```

Pass `onFork={handleFork}` to `MessageBubble` for user messages.

- [ ] **Step 6: Add fork provenance to SessionSidebar**

For sessions with `forked_from_session_id`, show a small "forked" badge or indicator next to the title.

- [ ] **Step 7: TypeScript check + frontend tests**

```bash
cd src/elspeth/web/frontend && npx tsc -p tsconfig.app.json --noEmit && npx vitest run
```

- [ ] **Step 8: Commit**

```
feat(web/frontend): add fork-from-message UX — edit bubble, fork action, provenance display
```

---

## Task 7: Integration tests and cleanup

- [ ] **Step 1: End-to-end fork flow test**

Test via TestClient:
1. Create session, send two messages (get assistant responses)
2. Fork from message 1 with edited content
3. Verify new session has correct message count (messages before fork + new edited message)
4. Verify original session is unchanged
5. Verify composition state in new session matches the pre-send state of the forked message

- [ ] **Step 2: Blob inheritance test**

1. Create session, upload a blob, send a message referencing it
2. Fork from that message
3. Verify blob is copied to new session with new ID
4. Verify blob content matches

- [ ] **Step 3: IDOR protection test**

1. Create session as User A, send messages
2. Attempt fork as User B → 404

- [ ] **Step 4: Commit**

```
test(web/sessions): add fork integration tests — provenance, blobs, IDOR
```

---

## Open Questions (resolve before implementation)

1. **State version in SendMessageRequest** — Should the client send `state_version` (int) or `state_id` (UUID)? Version is simpler for the frontend (already in compositionState). ID is more precise. Recommend: `state_version` since it's already available in the frontend store.

2. **Stale state handling** — If the client sends `state_version=3` but the server's current is `state_version=5` (another tab made changes), should send_message return 409 or proceed? Recommend: proceed with a warning in the response — the provenance records what was *actually* current, not what the client thought.

3. **Fork title** — `"[Original Title] (fork)"` or let the user name it? Recommend: auto-title with "(fork)" suffix, user can rename.

4. **System message in forked session** — Add a synthetic system message like "Forked from [original title] at message [N]"? Recommend: yes — it's visible context and appears in the audit trail.

5. **Maximum fork depth** — Should we limit fork-of-fork chains? Recommend: no limit for now, track via `forked_from_session_id` chain. Add a limit if abuse becomes a concern.

---

## Security Checklist

1. **IDOR:** Fork endpoint verifies session ownership (404 on mismatch)
2. **Audit:** Original session never mutated — fork is additive
3. **Blob quota:** Copy-on-fork respects per-session quota
4. **State provenance:** Server-authoritative, not client-trusted
5. **Cascade safety:** Deleting a forked session doesn't affect the original (FK is nullable, no CASCADE from child to parent)
