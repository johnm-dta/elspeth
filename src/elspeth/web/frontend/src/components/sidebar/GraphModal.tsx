import { useEffect, useId, useRef, useState } from "react";
import { GraphView } from "@/components/inspector/GraphView";
import { useFocusTrap } from "@/hooks/useFocusTrap";
import { OPEN_GRAPH_MODAL_EVENT } from "@/lib/composer-events";

export function GraphModal(): JSX.Element | null {
  const [isOpen, setIsOpen] = useState(false);
  const dialogRef = useRef<HTMLDivElement>(null);
  const titleId = useId();

  useFocusTrap(dialogRef, isOpen, ".graph-modal-close");

  useEffect(() => {
    function handleOpen() {
      setIsOpen(true);
    }

    window.addEventListener(OPEN_GRAPH_MODAL_EVENT, handleOpen);
    return () => window.removeEventListener(OPEN_GRAPH_MODAL_EVENT, handleOpen);
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
        className="graph-modal-backdrop"
        data-testid="graph-modal-backdrop"
        onClick={() => setIsOpen(false)}
        aria-hidden="true"
      />
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="graph-modal"
      >
        <header className="graph-modal-header">
          <h2 id={titleId}>Pipeline graph</h2>
          <button
            type="button"
            className="graph-modal-close"
            onClick={() => setIsOpen(false)}
            aria-label="Close graph"
          >
            x
          </button>
        </header>
        <div className="graph-modal-body">
          <GraphView />
        </div>
      </div>
    </>
  );
}
