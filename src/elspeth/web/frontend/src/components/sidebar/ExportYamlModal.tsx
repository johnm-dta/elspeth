import { useEffect, useId, useRef, useState } from "react";
import { YamlView } from "@/components/inspector/YamlView";
import { useFocusTrap } from "@/hooks/useFocusTrap";
import { OPEN_YAML_MODAL_EVENT } from "@/lib/composer-events";

export function ExportYamlModal(): JSX.Element | null {
  const [isOpen, setIsOpen] = useState(false);
  const dialogRef = useRef<HTMLDivElement>(null);
  const titleId = useId();

  useFocusTrap(dialogRef, isOpen, ".yaml-modal-close");

  useEffect(() => {
    function handleOpen() {
      setIsOpen(true);
    }

    window.addEventListener(OPEN_YAML_MODAL_EVENT, handleOpen);
    return () => window.removeEventListener(OPEN_YAML_MODAL_EVENT, handleOpen);
  }, []);

  useEffect(() => {
    if (!isOpen) return;

    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.preventDefault();
        setIsOpen(false);
      }
    }

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <>
      <div
        className="yaml-modal-backdrop"
        data-testid="yaml-modal-backdrop"
        onClick={() => setIsOpen(false)}
        aria-hidden="true"
      />
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="yaml-modal"
      >
        <header className="yaml-modal-header">
          <h2 id={titleId}>Export YAML</h2>
          <button
            type="button"
            className="yaml-modal-close"
            onClick={() => setIsOpen(false)}
            aria-label="Close export YAML"
          >
            ×
          </button>
        </header>
        <div className="yaml-modal-body">
          <YamlView />
        </div>
      </div>
    </>
  );
}
