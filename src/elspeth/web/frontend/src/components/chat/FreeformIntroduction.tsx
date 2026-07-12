import { Button } from "@/components/ui/Button";
import { usePreferencesStore } from "@/stores/preferencesStore";

export function FreeformIntroduction() {
  const loaded = usePreferencesStore((state) => state.loaded);
  const dismissedAt = usePreferencesStore(
    (state) => state.freeformIntroDismissedAt,
  );
  const writing = usePreferencesStore((state) => state.writing);
  const dismiss = usePreferencesStore((state) => state.dismissFreeformIntro);

  if (!loaded || dismissedAt !== null) {
    return null;
  }

  const handleDismiss = () => {
    void dismiss().catch(() => {
      // The store reports write failures through the app-level accessible
      // alert. Keep this card visible so the user can retry.
    });
  };

  return (
    <section
      className="freeform-introduction"
      aria-labelledby="freeform-introduction-title"
    >
      <h2 id="freeform-introduction-title">Build a pipeline</h2>
      <p>
        Describe what ELSPETH should read, how the data should change, and
        where the results should go. ELSPETH will propose an auditable pipeline
        for you to review before it runs.
      </p>
      <Button
        variant="ghost"
        compact
        disabled={writing}
        onClick={handleDismiss}
      >
        {writing ? "Hiding…" : "Don’t show this again"}
      </Button>
    </section>
  );
}
