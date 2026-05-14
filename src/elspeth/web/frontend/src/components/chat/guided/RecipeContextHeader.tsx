import type { RecipeContext } from "@/types/guided";

export function RecipeContextHeader({ context }: { context: RecipeContext }) {
  return (
    <div className="recipe-context-header">
      <h3>{context.recipe_name}</h3>
      <p>{context.description}</p>
      {context.alternatives.length > 0 && (
        <div>Alternatives: {context.alternatives.join(", ")}</div>
      )}
    </div>
  );
}
