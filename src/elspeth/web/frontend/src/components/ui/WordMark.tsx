import type { ElementType, HTMLAttributes } from "react";

export interface WordMarkProps extends HTMLAttributes<HTMLElement> {
  /** Font size in px (number) or any CSS length (string). @default 13 */
  size?: number | string;
  /** Element to render. @default "span" */
  as?: keyof JSX.IntrinsicElements;
}

/**
 * The ELSPETH wordmark. Renders the brand as live text in JetBrains Mono 700,
 * uppercase, with the canonical wordmark tracking — never an image. `size` sets
 * the font size; `as` chooses the rendered element.
 */
export function WordMark({ size = 13, as = "span", className = "", style, ...rest }: WordMarkProps) {
  const Tag = as as ElementType;
  return (
    <Tag
      className={className}
      style={{
        fontFamily: "var(--font-mono)",
        fontWeight: 700,
        fontSize: typeof size === "number" ? `${size}px` : size,
        textTransform: "uppercase",
        letterSpacing: "var(--tracking-wordmark)",
        color: "var(--color-text)",
        ...style,
      }}
      {...rest}
    >
      ELSPETH
    </Tag>
  );
}
