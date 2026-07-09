import type { SVGProps } from "react";

export type IconName =
  | "assistant"
  | "download"
  | "eye"
  | "pipeline"
  | "play"
  | "status-error"
  | "status-pending"
  | "status-ready"
  | "status-unknown"
  | "trash"
  | "user";

export interface IconProps extends Omit<SVGProps<SVGSVGElement>, "name"> {
  name: IconName;
}

export function Icon({ name, className, ...props }: IconProps) {
  return (
    <svg
      aria-hidden="true"
      className={["ui-icon", className].filter(Boolean).join(" ")}
      data-icon={name}
      fill="none"
      focusable="false"
      height="16"
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="1.8"
      viewBox="0 0 24 24"
      width="16"
      {...props}
    >
      {renderIcon(name)}
    </svg>
  );
}

function renderIcon(name: IconName) {
  switch (name) {
    case "assistant":
      return (
        <>
          <rect x="5" y="8" width="14" height="10" rx="3" />
          <path d="M9 8V5h6v3" />
          <path d="M9 13h.01" />
          <path d="M15 13h.01" />
          <path d="M10 17h4" />
        </>
      );
    case "download":
      return (
        <>
          <path d="M12 4v11" />
          <path d="m7 10 5 5 5-5" />
          <path d="M5 20h14" />
        </>
      );
    case "eye":
      return (
        <>
          <path d="M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6Z" />
          <circle cx="12" cy="12" r="2.75" />
        </>
      );
    case "pipeline":
      return (
        <>
          <circle cx="6" cy="7" r="2" />
          <circle cx="18" cy="7" r="2" />
          <circle cx="12" cy="17" r="2" />
          <path d="M8 8.5 11 15" />
          <path d="M16 8.5 13 15" />
        </>
      );
    case "play":
      return <path d="m8 5 11 7-11 7Z" />;
    case "status-error":
      return (
        <>
          <path d="M12 4 3 20h18L12 4Z" />
          <path d="M12 9v4" />
          <path d="M12 17h.01" />
        </>
      );
    case "status-pending":
      return (
        <>
          <circle cx="12" cy="12" r="8" />
          <path d="M12 8v4l3 2" />
        </>
      );
    case "status-ready":
      return (
        <>
          <circle cx="12" cy="12" r="8" />
          <path d="m8.5 12.5 2.25 2.25 4.75-5" />
        </>
      );
    case "status-unknown":
      return (
        <>
          <circle cx="12" cy="12" r="8" />
          <path d="M9.75 9.75a2.5 2.5 0 1 1 3.6 2.25c-.8.42-1.35 1.05-1.35 2" />
          <path d="M12 17h.01" />
        </>
      );
    case "trash":
      return (
        <>
          <path d="M4 7h16" />
          <path d="M10 11v6" />
          <path d="M14 11v6" />
          <path d="M6 7l1 13h10l1-13" />
          <path d="M9 7V4h6v3" />
        </>
      );
    case "user":
      return (
        <>
          <circle cx="12" cy="8" r="3" />
          <path d="M5 20a7 7 0 0 1 14 0" />
        </>
      );
  }
}
