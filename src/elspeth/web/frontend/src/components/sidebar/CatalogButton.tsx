export const OPEN_CATALOG_EVENT = "open-catalog";

export function CatalogButton(): JSX.Element {
  return (
    <button
      type="button"
      className="btn side-rail-catalog-btn"
      onClick={() => window.dispatchEvent(new CustomEvent(OPEN_CATALOG_EVENT))}
      aria-label="Catalog (reference)"
    >
      Catalog (reference)
    </button>
  );
}
