/**
 * ErrorBoundary — catches render-time errors in child components and
 * displays a recoverable fallback instead of white-screening the app.
 *
 * React error boundaries require class components; there is no hooks
 * equivalent for componentDidCatch / getDerivedStateFromError.
 */

import { Component, type ErrorInfo, type ReactNode } from "react";

interface ErrorBoundaryProps {
  /** Label shown in the fallback UI (e.g. "Graph view", "Chat panel") */
  label: string;
  children: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error(
      `[ErrorBoundary:${this.props.label}]`,
      error,
      info.componentStack,
    );
  }

  handleRetry = (): void => {
    this.setState({ hasError: false, error: null });
  };

  render(): ReactNode {
    if (this.state.hasError) {
      return (
        <div className="error-boundary-fallback" role="alert">
          <div className="error-boundary-icon" aria-hidden="true">
            &#x26A0;
          </div>
          <p className="error-boundary-title">
            {this.props.label} encountered an error
          </p>
          <p className="error-boundary-detail">
            {this.state.error?.message ?? "An unexpected error occurred."}
          </p>
          <button
            onClick={this.handleRetry}
            className="btn error-boundary-retry"
          >
            Retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
