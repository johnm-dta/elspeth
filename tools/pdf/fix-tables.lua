-- fix-tables.lua — Data-driven table column width overrides + figure
-- table-kind tagging for the ELSPETH PDF pipeline.
--
-- Loads table profiles from table-profiles.json and applies column widths
-- based on header text matching.  Profiles are matched by:
--   - col_N: prefix match (lowercase, trimmed) against header cell N
--   - col_N_contains: substring match against header cell N
--   - cols: exact column count match (required)
--
-- An empty profiles array is a valid configuration: the pipeline runs as
-- a no-op for column widths and Pandoc's auto-sizing handles every table.
-- This is the right starting point — only add a profile when a specific
-- table's auto-sized layout demonstrably reads poorly in the rendered PDF.

local script_dir = PANDOC_SCRIPT_FILE:match("(.*[/\\])")
local profiles_path = script_dir .. "table-profiles.json"

local function load_profiles()
  local file = io.open(profiles_path, "r")
  if not file then
    io.stderr:write("[warn] Could not open " .. profiles_path .. "\n")
    return {}
  end
  local content = file:read("*a")
  file:close()

  local ok, data = pcall(function()
    return pandoc.json.decode(content)
  end)

  if not ok or not data or not data.profiles then
    io.stderr:write("[warn] Could not parse " .. profiles_path .. "\n")
    return {}
  end

  return data.profiles
end

local PROFILES = load_profiles()

local function cell_text(cell)
  return pandoc.utils.stringify(cell.contents):lower():gsub("^%s+", ""):gsub("%s+$", "")
end

local function header_texts(tbl)
  if #tbl.head.rows == 0 then return {} end
  local texts = {}
  for _, cell in ipairs(tbl.head.rows[1].cells) do
    texts[#texts + 1] = cell_text(cell)
  end
  return texts
end

local function starts(s, prefix)
  return s:sub(1, #prefix) == prefix
end

local function has(s, sub)
  return s:find(sub, 1, true) ~= nil
end

local function matches_profile(profile, headers, col_count)
  if profile.cols ~= col_count then
    return false
  end

  local match = profile.match
  if not match then
    return false
  end

  for key, value in pairs(match) do
    local col_num, match_type = key:match("^col_(%d+)(.*)$")
    if col_num then
      col_num = tonumber(col_num)
      local header = headers[col_num]
      if not header then
        return false
      end

      if match_type == "" then
        if not starts(header, value) then
          return false
        end
      elseif match_type == "_contains" then
        if not has(header, value) then
          return false
        end
      else
        return false
      end
    end
  end

  return true
end

local function find_profile(headers, col_count)
  for _, profile in ipairs(PROFILES) do
    if matches_profile(profile, headers, col_count) then
      return profile
    end
  end
  return nil
end

function Table(tbl)
  local headers = header_texts(tbl)
  local col_count = #tbl.colspecs

  if #headers == 0 then
    return nil
  end

  local profile = find_profile(headers, col_count)
  if not profile then
    return nil
  end

  for i, width in ipairs(profile.widths) do
    if i <= col_count then
      tbl.colspecs[i] = {tbl.colspecs[i][1], width}
    end
  end

  return tbl
end

-- ─────────────────────────────────────────────────────────────
-- Figure filter: add kind="table" to figures containing tables
-- ─────────────────────────────────────────────────────────────
-- Pandoc wraps captioned tables in Figure elements.  The Typst writer
-- needs kind: table for proper table numbering and List of Tables
-- generation in the template.

local function contains_table(blocks)
  for _, block in ipairs(blocks) do
    if block.t == "Table" then
      return true
    end
    if block.content then
      for _, inner in ipairs(block.content) do
        if inner.t == "Table" then
          return true
        end
      end
    end
  end
  return false
end

function Figure(fig)
  if contains_table(fig.content) then
    fig.attr.attributes["kind"] = "table"
    return fig
  end
  return nil
end
