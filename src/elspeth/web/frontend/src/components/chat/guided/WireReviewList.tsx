export interface WireReviewItem {
  id: string;
  from: string;
  to: string;
  summary: string;
  detail?: string | null;
  status?: "connected" | "warning" | "unchecked";
  ariaLabel?: string;
}

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
          <span>{item.from}</span>
          <span aria-hidden="true">{" → "}</span>
          <span>{item.to}</span>
          <span aria-hidden="true">{" — "}</span>
          <span>{item.summary}</span>
          {item.detail ? <span> {item.detail}</span> : null}
        </li>
      ))}
    </ul>
  );
}
