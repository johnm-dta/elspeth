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
      <h2 id="freeform-introduction-title">How pipelines work</h2>
      <p>
        A pipeline is a controlled route for information. You choose what
        enters, what happens to it, and where the result goes. ELSPETH records
        each step so you can review how every output was produced.
      </p>
      <section className="freeform-introduction-section">
        <h3>The three building blocks</h3>
        <dl className="freeform-introduction-definitions">
          <div className="freeform-introduction-definition">
            <dt>Sources</dt>
            <dd>
              bring records into the pipeline from files, databases, APIs, or
              text. ELSPETH tracks each incoming record through the run.
            </dd>
          </div>
          <div className="freeform-introduction-definition">
            <dt>Transforms</dt>
            <dd>
              examine or change records. They can clean fields, enrich content,
              apply an LLM, or prepare data for the next step.
            </dd>
          </div>
          <div className="freeform-introduction-definition">
            <dt>Sinks</dt>
            <dd>
              receive records at the end of a route. They can write results to
              files, data stores, or other configured destinations; records
              requiring attention can follow a separate route.
            </dd>
          </div>
        </dl>
      </section>
      <section className="freeform-introduction-section">
        <h3>Wiring the flow</h3>
        <p>
          Wiring is the set of connections between these components. A simple
          pipeline runs from source to transforms to sink. For a more involved
          flow, think of each record as a case moving through a controlled
          workplace:
        </p>
        <dl className="freeform-introduction-definitions">
          <div className="freeform-introduction-definition">
            <dt>Gate</dt>
            <dd>
              is a sorting desk. It sends each case along the appropriate route
              according to a stated condition.
            </dd>
          </div>
          <div className="freeform-introduction-definition">
            <dt>Fork</dt>
            <dd>
              sends controlled copies of one case to several specialist teams.
              ELSPETH tracks each parallel path independently.
            </dd>
          </div>
          <div className="freeform-introduction-definition">
            <dt>Coalesce</dt>
            <dd>
              waits for the required specialist responses, then combines their
              findings into one case that can continue.
            </dd>
          </div>
          <div className="freeform-introduction-definition">
            <dt>Aggregate</dt>
            <dd>
              brings a group of cases together for batch work, such as
              producing totals, statistics, or a report.
            </dd>
          </div>
          <div className="freeform-introduction-definition">
            <dt>Queue</dt>
            <dd>
              is a shared inbox. It accepts cases from several upstream teams
              and feeds one next step while keeping every case separate.
            </dd>
          </div>
          <div className="freeform-introduction-definition">
            <dt>Expand</dt>
            <dd>
              opens a bundled case into several independently tracked cases.
            </dd>
          </div>
        </dl>
      </section>
      <p>
        Describe the outcome you need in ordinary language. ELSPETH will
        propose the components and their wiring; review the graph and details
        before you run it.
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
