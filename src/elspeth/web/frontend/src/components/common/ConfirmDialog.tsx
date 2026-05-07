import { useEffect, useRef } from "react";
import { useFocusTrap } from "@/hooks/useFocusTrap";

interface ConfirmDialogProps {
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  variant?: "default" | "danger";
  onConfirm: () => void;
  onCancel: () => void;
}

/**
 * Styled replacement for window.confirm().
 *
 * Focus-trapped modal with keyboard support (Escape to cancel,
 * Enter on confirm button). Focus returns to the trigger element
 * on close via useFocusTrap's restore-on-unmount behaviour.
 */
export function ConfirmDialog({
  title,
  message,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  variant = "default",
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  const dialogRef = useRef<HTMLDivElement>(null);

  // Focus trap with initial focus on the confirm button.
  // On unmount, focus restores to the element that was focused before the dialog.
  useFocusTrap(dialogRef, true, ".confirm-dialog-confirm-btn");

  // Escape key closes
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.preventDefault();
        onCancel();
      }
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onCancel]);

  const confirmBtnClass =
    variant === "danger" ? "btn btn-danger" : "btn btn-primary";

  return (
    <>
      <div
        className="confirm-dialog-backdrop"
        onClick={onCancel}
        role="presentation"
      />
      <div
        ref={dialogRef}
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-dialog-title"
        aria-describedby="confirm-dialog-message"
        className="confirm-dialog"
      >
        <h2 id="confirm-dialog-title" className="confirm-dialog-title">
          {title}
        </h2>
        <p id="confirm-dialog-message" className="confirm-dialog-message">
          {message}
        </p>
        <div className="confirm-dialog-actions">
          <button
            onClick={onCancel}
            className="btn confirm-dialog-btn"
          >
            {cancelLabel}
          </button>
          <button
            onClick={onConfirm}
            className={`${confirmBtnClass} confirm-dialog-btn confirm-dialog-confirm-btn`}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </>
  );
}
