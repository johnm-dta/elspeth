import { describe, it, expect } from "vitest";
import type { ChatMessage, ToolCall } from "@/types/api";
import { groupIntoTurns } from "./turns";

function msg(overrides: Partial<ChatMessage> & { id: string; role: ChatMessage["role"] }): ChatMessage {
  return {
    session_id: "s1",
    content: "",
    tool_calls: null,
    created_at: "2026-05-19T00:00:00Z",
    ...overrides,
  } as ChatMessage;
}

function tc(name: string, id = name): ToolCall {
  return { id, type: "function", function: { name, arguments: "{}" } };
}

describe("groupIntoTurns", () => {
  it("returns no turns for an empty list", () => {
    expect(groupIntoTurns([])).toEqual([]);
  });

  it("emits a user turn for a single user message", () => {
    const turns = groupIntoTurns([msg({ id: "u1", role: "user", content: "Hi" })]);
    expect(turns).toHaveLength(1);
    expect(turns[0].kind).toBe("user");
    expect(turns[0].messages).toHaveLength(1);
    expect(turns[0].messages[0].id).toBe("u1");
  });

  it("coalesces the a8afd33e-shaped session into 2 turns: user + agent", () => {
    // Reproduces the live shape: 1 user prompt + 7 assistant rows + final answer.
    const messages: ChatMessage[] = [
      msg({ id: "u1", role: "user", content: "create a list..." }),
      msg({
        id: "a1",
        role: "assistant",
        tool_calls: [tc("list_secret_refs"), tc("list_models"), tc("get_plugin_schema", "g1"), tc("get_plugin_schema", "g2"), tc("get_plugin_schema", "g3"), tc("get_plugin_schema", "g4")],
      }),
      msg({ id: "a2", role: "assistant", tool_calls: [tc("create_blob")] }),
      msg({ id: "a3", role: "assistant", tool_calls: [tc("set_pipeline")] }),
      msg({ id: "a4", role: "assistant", tool_calls: [tc("patch_node_options")] }),
      msg({ id: "a5", role: "assistant", tool_calls: [tc("preview_pipeline", "p1")] }),
      msg({ id: "a6", role: "assistant" }), // orphan empty row — absorbed silently
      msg({ id: "a7", role: "assistant", tool_calls: [tc("preview_pipeline", "p2")] }),
      msg({ id: "a8", role: "assistant", content: "Built a workflow..." }),
    ];

    const turns = groupIntoTurns(messages);
    expect(turns).toHaveLength(2);
    expect(turns[0].kind).toBe("user");

    const agent = turns[1];
    expect(agent.kind).toBe("agent");
    expect(agent.messages.map((m) => m.id)).toEqual(["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8"]);
    expect(agent.aggregatedToolCalls.map((t) => t.function.name)).toEqual([
      "list_secret_refs",
      "list_models",
      "get_plugin_schema",
      "get_plugin_schema",
      "get_plugin_schema",
      "get_plugin_schema",
      "create_blob",
      "set_pipeline",
      "patch_node_options",
      "preview_pipeline",
      "preview_pipeline",
    ]);
    expect(agent.finalContent).toBe("Built a workflow...");
  });

  it("alternates: user, agent, user yields three turns", () => {
    const messages: ChatMessage[] = [
      msg({ id: "u1", role: "user", content: "first" }),
      msg({ id: "a1", role: "assistant", content: "answer one" }),
      msg({ id: "u2", role: "user", content: "second" }),
    ];
    const turns = groupIntoTurns(messages);
    expect(turns.map((t) => t.kind)).toEqual(["user", "agent", "user"]);
  });

  it("consecutive user messages each get their own turn", () => {
    const messages: ChatMessage[] = [
      msg({ id: "u1", role: "user", content: "a" }),
      msg({ id: "u2", role: "user", content: "b" }),
    ];
    const turns = groupIntoTurns(messages);
    expect(turns.map((t) => t.kind)).toEqual(["user", "user"]);
    expect(turns.map((t) => t.messages[0].id)).toEqual(["u1", "u2"]);
  });

  it("leading assistant rows (no preceding user) form an opening agent turn", () => {
    const messages: ChatMessage[] = [
      msg({ id: "a1", role: "assistant", content: "kickoff" }),
      msg({ id: "u1", role: "user", content: "hi" }),
    ];
    const turns = groupIntoTurns(messages);
    expect(turns.map((t) => t.kind)).toEqual(["agent", "user"]);
    expect(turns[0].finalContent).toBe("kickoff");
  });

  it("system messages stand alone, not absorbed into agent turn", () => {
    const messages: ChatMessage[] = [
      msg({ id: "u1", role: "user", content: "hi" }),
      msg({ id: "s1", role: "system", content: "Pipeline reverted to version 3." }),
      msg({ id: "a1", role: "assistant", content: "ok" }),
    ];
    const turns = groupIntoTurns(messages);
    expect(turns.map((t) => t.kind)).toEqual(["user", "system", "agent"]);
  });

  it("role=tool rows are absorbed into the agent turn defensively", () => {
    // role="tool" rows should normally be filtered server-side, but if any
    // leak through, they belong to the current agent turn — not their own bubble.
    const messages: ChatMessage[] = [
      msg({ id: "u1", role: "user", content: "hi" }),
      msg({ id: "a1", role: "assistant", tool_calls: [tc("set_pipeline")] }),
      msg({ id: "t1", role: "tool", content: '{"ok": true}', tool_call_id: "set_pipeline" }),
      msg({ id: "a2", role: "assistant", content: "Done." }),
    ];
    const turns = groupIntoTurns(messages);
    expect(turns).toHaveLength(2);
    expect(turns[1].kind).toBe("agent");
    expect(turns[1].messages.map((m) => m.id)).toEqual(["a1", "t1", "a2"]);
    expect(turns[1].finalContent).toBe("Done.");
  });

  it("finalContent is empty string when no assistant row in the turn has content", () => {
    const messages: ChatMessage[] = [
      msg({ id: "u1", role: "user", content: "hi" }),
      msg({ id: "a1", role: "assistant", tool_calls: [tc("list_models")] }),
      msg({ id: "a2", role: "assistant" }),
    ];
    const turns = groupIntoTurns(messages);
    expect(turns[1].kind).toBe("agent");
    expect(turns[1].finalContent).toBe("");
  });

  it("turn id is stable: the first message id in the turn", () => {
    const messages: ChatMessage[] = [
      msg({ id: "u1", role: "user", content: "hi" }),
      msg({ id: "a1", role: "assistant", tool_calls: [tc("x")] }),
      msg({ id: "a2", role: "assistant", content: "done" }),
    ];
    const turns = groupIntoTurns(messages);
    expect(turns[0].id).toBe("u1");
    expect(turns[1].id).toBe("a1");
  });

  it("aggregatedToolCalls excludes nulls and skips empty arrays cleanly", () => {
    const messages: ChatMessage[] = [
      msg({ id: "u1", role: "user", content: "hi" }),
      msg({ id: "a1", role: "assistant", tool_calls: null }),
      msg({ id: "a2", role: "assistant", tool_calls: [] }),
      msg({ id: "a3", role: "assistant", tool_calls: [tc("only_one")] }),
    ];
    const turns = groupIntoTurns(messages);
    expect(turns[1].aggregatedToolCalls.map((t) => t.function.name)).toEqual(["only_one"]);
  });

  it("primaryMessage is the last assistant message with non-empty content when one exists", () => {
    const messages: ChatMessage[] = [
      msg({ id: "u1", role: "user", content: "hi" }),
      msg({ id: "a1", role: "assistant", tool_calls: [tc("x")] }),
      msg({ id: "a2", role: "assistant", content: "the answer" }),
      msg({ id: "a3", role: "assistant", tool_calls: [tc("y")] }),
    ];
    const turns = groupIntoTurns(messages);
    expect(turns[1].primaryMessage.id).toBe("a2");
  });

  it("primaryMessage falls back to the last message in the turn when no content exists", () => {
    const messages: ChatMessage[] = [
      msg({ id: "u1", role: "user", content: "hi" }),
      msg({ id: "a1", role: "assistant", tool_calls: [tc("x")] }),
      msg({ id: "a2", role: "assistant", tool_calls: [tc("y")] }),
    ];
    const turns = groupIntoTurns(messages);
    expect(turns[1].primaryMessage.id).toBe("a2");
  });

  // ----- isComplete (atomic-reveal contract) -----
  //
  // The agent bubble is rendered atomically: it must remain hidden while the
  // turn is mid-flight (only tool-call rows visible so far, no LLM text reply
  // landed) and appear in one piece once the final content row arrives. This
  // reverses the prior "stream tool calls live" behaviour from commit ceb4d38a1.
  //
  // The contract is purely client-side and derived from already-present audit
  // rows — no backend signal is required. An agent turn is "complete" iff some
  // assistant row in the turn carries non-empty `content` (the LLM's text
  // reply). User and system turns are standalone and always complete.

  it("user turn isComplete is always true", () => {
    const turns = groupIntoTurns([msg({ id: "u1", role: "user", content: "hi" })]);
    expect(turns[0].isComplete).toBe(true);
  });

  it("system turn isComplete is always true", () => {
    const turns = groupIntoTurns([msg({ id: "s1", role: "system", content: "Pipeline reverted." })]);
    expect(turns[0].isComplete).toBe(true);
  });

  it("agent turn with only tool-call rows is isComplete=false (mid-flight)", () => {
    const messages: ChatMessage[] = [
      msg({ id: "u1", role: "user", content: "hi" }),
      msg({ id: "a1", role: "assistant", tool_calls: [tc("list_models")] }),
      msg({ id: "a2", role: "assistant", tool_calls: [tc("set_pipeline")] }),
    ];
    const turns = groupIntoTurns(messages);
    expect(turns[1].kind).toBe("agent");
    expect(turns[1].isComplete).toBe(false);
  });

  it("agent turn with a content-bearing assistant row is isComplete=true", () => {
    const messages: ChatMessage[] = [
      msg({ id: "u1", role: "user", content: "hi" }),
      msg({ id: "a1", role: "assistant", tool_calls: [tc("list_models")] }),
      msg({ id: "a2", role: "assistant", content: "Done." }),
    ];
    const turns = groupIntoTurns(messages);
    expect(turns[1].isComplete).toBe(true);
  });

  it("agent turn becomes complete as soon as ANY assistant row has content", () => {
    // Defensive case: content row appears before tool-call rows in the
    // sequence (shouldn't happen with current backend ordering, but the rule
    // is "content exists somewhere in the turn", not "content is last").
    const messages: ChatMessage[] = [
      msg({ id: "u1", role: "user", content: "hi" }),
      msg({ id: "a1", role: "assistant", content: "First take." }),
      msg({ id: "a2", role: "assistant", tool_calls: [tc("recheck")] }),
    ];
    const turns = groupIntoTurns(messages);
    expect(turns[1].isComplete).toBe(true);
  });

  it("empty content strings do not satisfy isComplete", () => {
    // An assistant row with content: "" (rather than null) must not be treated
    // as a content-bearing reply. This protects against backend writes that
    // pre-create the row with empty content before the LLM finishes streaming.
    const messages: ChatMessage[] = [
      msg({ id: "u1", role: "user", content: "hi" }),
      msg({ id: "a1", role: "assistant", content: "", tool_calls: [tc("only")] }),
    ];
    const turns = groupIntoTurns(messages);
    expect(turns[1].isComplete).toBe(false);
  });
});
