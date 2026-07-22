export interface WireReviewItem {
  id: string;
  from: string;
  to: string;
  summary: string;
  detail?: string | null;
  /**
   * Optional per-route contract status, rendered as a compact chip instead of
   * trailing "— …" prose (the operator-reported debug-dump read). Callers that
   * pass `status` MUST fold the status wording into `ariaLabel` too: the li's
   * aria-label overrides its text content as the accessible name, so a chip
   * outside the label is invisible to screen readers.
   */
  status?: "connected" | "warning" | "unchecked";
  ariaLabel?: string;
}

/** Chip copy reuses the wire stage's plain-language status register
 *  (edgeStatus) — no new vocabulary. */
const STATUS_LABELS: Record<NonNullable<WireReviewItem["status"]>, string> = {
  connected: "connected",
  warning: "not connected correctly",
  unchecked: "not yet checked",
};

interface WireReviewListProps {
  items: WireReviewItem[];
  ariaLabel: string;
  className?: string;
}

/** Shared presentation-only route list. Callers own topology construction. */
export function WireReviewList({
  items,
  ariaLabel,
  className,
}: WireReviewListProps): JSX.Element {
  return (
    <ul className={className ?? "guided-wire-review"} aria-label={ariaLabel}>
      {items.map((item) => (
        <li key={item.id} data-edge-id={item.id} aria-label={item.ariaLabel}>
          <span className="wire-review-route">
            <span>{item.from}</span>
            <span aria-hidden="true">{" → "}</span>
            <span>{item.to}</span>
          </span>
          <span aria-hidden="true">{" — "}</span>
          <span className="wire-review-flow">{item.summary}</span>
          {item.status !== undefined ? (
            <span className={`wire-review-status wire-review-status--${item.status}`}>
              {STATUS_LABELS[item.status]}
            </span>
          ) : null}
          {item.detail ? <span className="wire-review-detail"> {item.detail}</span> : null}
        </li>
      ))}
    </ul>
  );
}
