import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ComponentReviewTurn } from "./ComponentReviewTurn";
import type {
  ComponentReviewPayload,
  GuidedComponentAction,
  GuidedRespondAction,
} from "@/types/guided";

const SOURCE_A = "00000000-0000-4000-8000-000000000101";
const SOURCE_B = "00000000-0000-4000-8000-000000000102";

const SOURCE_REVIEW: ComponentReviewPayload = {
  component_kind: "source",
  items: [
    { stable_id: SOURCE_A, name: "customers", plugin: "csv", status: "reviewed" },
    { stable_id: SOURCE_B, name: "orders", plugin: "json", status: "reviewed" },
  ],
  allowed_actions: ["add", "edit", "remove", "reorder", "finish"],
};

function componentBody(component_action: GuidedComponentAction): GuidedRespondAction {
  return {
    chosen: null,
    edited_values: null,
    custom_inputs: null,
    proposal_id: null,
    draft_hash: null,
    edit_target: null,
    control_signal: null,
    component_action,
  };
}

describe("ComponentReviewTurn", () => {
  it("renders only the server-authored source collection and submits stable-id actions", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<ComponentReviewTurn payload={SOURCE_REVIEW} onSubmit={onSubmit} />);

    expect(screen.getByRole("heading", { name: "Review sources" })).toBeVisible();
    expect(screen.getByRole("listitem", { name: "customers, csv, reviewed" })).toBeVisible();
    expect(screen.getByRole("listitem", { name: "orders, json, reviewed" })).toBeVisible();

    await user.click(screen.getByRole("button", { name: "Edit orders" }));
    await user.click(screen.getByRole("button", { name: "Remove customers" }));
    await user.click(screen.getByRole("button", { name: "Add source" }));
    await user.click(screen.getByRole("button", { name: "Finish sources" }));

    expect(onSubmit).toHaveBeenNthCalledWith(
      1,
      componentBody({ action: "edit", target: { kind: "source", stable_id: SOURCE_B } }),
    );
    expect(onSubmit).toHaveBeenNthCalledWith(
      2,
      componentBody({ action: "remove", target: { kind: "source", stable_id: SOURCE_A } }),
    );
    expect(onSubmit).toHaveBeenNthCalledWith(
      3,
      componentBody({ action: "add", component_kind: "source" }),
    );
    expect(onSubmit).toHaveBeenNthCalledWith(
      4,
      componentBody({ action: "finish", component_kind: "source" }),
    );
  });

  it("submits an exact stable-id permutation and follows authoritative ordering after rerender", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    const { rerender } = render(
      <ComponentReviewTurn payload={SOURCE_REVIEW} onSubmit={onSubmit} />,
    );

    expect(screen.getByRole("button", { name: "Move customers up" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Move orders down" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Move orders up" }));

    expect(onSubmit).toHaveBeenCalledWith(
      componentBody({
        action: "reorder",
        component_kind: "source",
        stable_ids: [SOURCE_B, SOURCE_A],
      }),
    );
    // No optimistic reorder: the server response remains the rendered authority.
    expect(screen.getAllByRole("listitem").map((item) => item.getAttribute("aria-label"))).toEqual([
      "customers, csv, reviewed",
      "orders, json, reviewed",
    ]);

    rerender(
      <ComponentReviewTurn
        payload={{ ...SOURCE_REVIEW, items: [SOURCE_REVIEW.items[1], SOURCE_REVIEW.items[0]] }}
        onSubmit={onSubmit}
      />,
    );
    expect(screen.getAllByRole("listitem").map((item) => item.getAttribute("aria-label"))).toEqual([
      "orders, json, reviewed",
      "customers, csv, reviewed",
    ]);
    await user.click(screen.getByRole("button", { name: "Edit orders" }));
    expect(onSubmit).toHaveBeenLastCalledWith(
      componentBody({ action: "edit", target: { kind: "source", stable_id: SOURCE_B } }),
    );
  });

  it("renders output actions from the closed server allowlist and suppresses single-item removal", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <ComponentReviewTurn
        payload={{
          component_kind: "output",
          items: [
            {
              stable_id: SOURCE_A,
              name: "audit_log",
              plugin: "json",
              status: "reviewed",
            },
          ],
          allowed_actions: ["add", "edit", "reorder", "finish"],
        }}
        onSubmit={onSubmit}
      />,
    );

    expect(screen.queryByRole("button", { name: "Remove audit_log" })).toBeNull();
    expect(screen.queryByRole("button", { name: /Move audit_log/ })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Edit audit_log" }));
    expect(onSubmit).toHaveBeenCalledWith(
      componentBody({ action: "edit", target: { kind: "output", stable_id: SOURCE_A } }),
    );
  });

  it("hides closed actions and disables every rendered control while pending", () => {
    render(
      <ComponentReviewTurn
        payload={{ ...SOURCE_REVIEW, allowed_actions: ["finish"] }}
        onSubmit={vi.fn()}
        disabled
      />,
    );

    expect(screen.getByRole("button", { name: "Finish sources" })).toBeDisabled();
    expect(screen.queryByRole("button", { name: "Add source" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Edit customers" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Remove customers" })).toBeNull();
    expect(screen.queryByRole("button", { name: /Move customers/ })).toBeNull();
  });
});
