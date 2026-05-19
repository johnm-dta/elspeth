import { OPEN_CATALOG_EVENT } from "@/lib/composer-events";

export function CatalogButton(): JSX.Element {
  return (
    <button
      type="button"
      className="side-rail-catalog-btn"
      onClick={() => window.dispatchEvent(new CustomEvent(OPEN_CATALOG_EVENT))}
      aria-label="Catalog (reference)"
    >
      <span className="catalog-reference-label">Plugin catalog</span>
      <span className="catalog-reference-meta" aria-hidden="true">
        Reference
      </span>
    </button>
  );
}
