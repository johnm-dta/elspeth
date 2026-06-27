import { useId } from "react";
import type { InputHTMLAttributes, ReactNode } from "react";

/**
 * Text input on the ELSPETH elevated surface with a strong border and visible
 * focus ring. Optional label + hint. `mono` switches to JetBrains Mono for
 * forensic-register values (secret names, paths, hashes). Pair with
 * `label`/`hint` or use the bare control inside your own field layout.
 *
 * The label is associated with the control via the passed `id`; when no `id`
 * is supplied one is generated with React `useId()` so the label/control pair
 * is always programmatically linked.
 */
export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  /** Field label rendered above the control. */
  label?: ReactNode;
  /** Helper text rendered below the control. */
  hint?: ReactNode;
  /** Render the value in JetBrains Mono. @default false */
  mono?: boolean;
}

export function Input({
  label,
  hint,
  mono = false,
  id,
  className = "",
  ...rest
}: InputProps) {
  const generatedId = useId();
  const inputId = id ?? generatedId;
  const cls = ["input", mono ? "input-mono" : "", className]
    .filter(Boolean)
    .join(" ");
  const control = <input id={inputId} className={cls} {...rest} />;
  if (!label && !hint) return control;
  return (
    <div>
      {label ? (
        <label className="field-label" htmlFor={inputId}>
          {label}
        </label>
      ) : null}
      {control}
      {hint ? <div className="field-hint">{hint}</div> : null}
    </div>
  );
}
