-- leadins.lua — detect run-in heading paragraphs and wrap them so
-- the Typst template can style the opener distinctly from ordinary
-- in-paragraph emphasis.
--
-- A "lead-in" is a paragraph whose first inline is a Strong span
-- that ends in a period or em-dash, e.g.:
--
--   **Headline result.** The composer's behaviour is disciplined…
--   **Failure mode 1 — Wrong model name.** All three runs failed…
--
-- These function as run-in headings: they label the paragraph's
-- topic without breaking the prose into a separate heading.  The
-- typographic treatment should reflect that role — not the same as
-- mid-paragraph emphasis like "**However**, none of the nine…".
--
-- We rewrite the Strong's content into:
--
--   #leadin[<the strong's children>]
--
-- followed by the rest of the paragraph.  The template defines
-- ``#let leadin(...) = ...`` which controls the styling (small caps,
-- navy, bold, tracked).  This keeps the visual decision in one place.
--
-- Why a Lua filter rather than a global ``show strong: ...`` rule:
-- the lead-in pattern is a structural anchor, distinct from ordinary
-- inline bold ("**However**, none of the…").  Restyling all bold
-- would lift the noise as well as the signal.

local function ends_with_terminator(text)
  if text == nil or #text == 0 then return false end
  -- Inspect the last UTF-8 codepoint by trimming trailing whitespace.
  -- Period, em-dash, en-dash all mark a finished label phrase.
  local trimmed = text:gsub("%s+$", "")
  if #trimmed == 0 then return false end
  -- Em-dash and en-dash are 3-byte UTF-8 sequences; check both
  -- their final byte and a single-character trailing slice.
  if trimmed:sub(-1) == "." then return true end
  -- Match the last UTF-8 codepoint by extracting up to 4 bytes.
  for _, dash in ipairs({"\xE2\x80\x94", "\xE2\x80\x93"}) do
    if trimmed:sub(-#dash) == dash then return true end
  end
  return false
end

function Para(p)
  if #p.content < 1 then return nil end
  local first = p.content[1]
  if first.t ~= "Strong" then return nil end

  local lead_text = pandoc.utils.stringify(first)
  if not ends_with_terminator(lead_text) then return nil end

  -- Build a new inline list: opener + Strong's children + closer + rest.
  local opener = pandoc.RawInline("typst", "#leadin[")
  local closer = pandoc.RawInline("typst", "]")

  local new_content = { opener }
  for _, child in ipairs(first.content) do
    new_content[#new_content + 1] = child
  end
  new_content[#new_content + 1] = closer
  for i = 2, #p.content do
    new_content[#new_content + 1] = p.content[i]
  end

  return pandoc.Para(new_content)
end
