import { AlertBanner } from "../ui";
import {
  TURN_1_PRIMARY_BUTTON,
  TURN_1_SKIP_BUTTON,
  TUTORIAL_RUN_PREAMBLE,
  WELCOME_LAYERS,
} from "./copy";

interface TutorialTurn1WelcomeProps {
  onStart: () => void;
  onSkip: () => void;
  startDisabled?: boolean;
  startDisabledReason?: string | null;
}

export function TutorialTurn1Welcome({
  onStart,
  onSkip,
  startDisabled = false,
  startDisabledReason = null,
}: TutorialTurn1WelcomeProps): JSX.Element {
  return (
    <section className="tutorial-turn" aria-labelledby="tutorial-welcome-title">
      <p className="tutorial-kicker">First run</p>
      <h2 id="tutorial-welcome-title">Welcome to ELSPETH.</h2>
      <p>
        In about 3 minutes we will build and run your first pipeline together.
        Then you will choose how you want to work going forward.
      </p>
      <ol className="tutorial-layer-grid" aria-label="Pipeline layers">
        {WELCOME_LAYERS.map((layer) => (
          <li className="tutorial-layer" key={layer.label}>
            <strong>{layer.label}</strong>
            <span>{layer.description}</span>
          </li>
        ))}
      </ol>
      <AlertBanner tone="info" className="tutorial-disclosure tutorial-preamble">
        {TUTORIAL_RUN_PREAMBLE}
      </AlertBanner>
      {startDisabledReason !== null && (
        <AlertBanner className="tutorial-disclosure">
          {startDisabledReason}
        </AlertBanner>
      )}
      <div className="tutorial-actions">
        <button
          type="button"
          className="btn btn-primary"
          onClick={onStart}
          disabled={startDisabled}
        >
          {TURN_1_PRIMARY_BUTTON}
        </button>
        <button
          type="button"
          className="tutorial-link-button"
          onClick={onSkip}
        >
          {TURN_1_SKIP_BUTTON}
        </button>
      </div>
    </section>
  );
}
