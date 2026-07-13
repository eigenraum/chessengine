// Parameter panel (DESIGN-VISU.md §4.3): every knob that influences the
// search, in two groups with different lifecycles. Search limits apply at
// the next search start; structural changes rebuild the engine and drop the
// search tree (the UI says so). Inputs are generated from the field lists —
// the server's /api/config is the source of truth for values.

const LIMIT_FIELDS = [
  ["max_time_ms", "ms per search", 100],
  ["max_simulations", "-1 = unlimited", 1],
  ["convergence_window", "simulations; 0 = off", 100],
  ["convergence_cp_threshold", "centipawns", 1],
  ["c_puct", "exploration constant", 0.1],
  ["virtual_loss", "", 1],
];
const STRUCTURAL_FIELDS = [
  ["workers", "search threads", 1],
  ["batch_size", "leaf evals per batch", 1],
  ["max_nodes", "tree arena capacity (nodes)", 1],
  ["seed", "", 1],
];

export class ParamsPanel {
  /** @param {HTMLElement} root
   *  @param {{put: (body: object) => Promise<object|null>}} callbacks */
  constructor(root, { put }) {
    this.put = put;
    this.inputs = {};
    root.appendChild(this._group("Search limits", LIMIT_FIELDS, "limits",
      "apply at the next search start"));
    root.appendChild(this._group("Structural", STRUCTURAL_FIELDS, "structural",
      "rebuilds the engine — the search tree is dropped"));
    this.pendingNote = document.createElement("p");
    this.pendingNote.className = "params-note";
    root.appendChild(this.pendingNote);
  }

  _group(title, fields, key, note) {
    const fieldset = document.createElement("fieldset");
    const legend = document.createElement("legend");
    legend.textContent = title;
    fieldset.appendChild(legend);
    for (const [name, hint, step] of fields) {
      const label = document.createElement("label");
      label.className = "param";
      label.title = hint;
      const span = document.createElement("span");
      span.textContent = name;
      const input = document.createElement("input");
      input.type = "number";
      input.step = String(step);
      this.inputs[name] = input;
      label.append(span, input);
      fieldset.appendChild(label);
    }
    const apply = document.createElement("button");
    apply.textContent = "Apply";
    apply.title = note;
    apply.addEventListener("click", () => this._apply(key, fields));
    fieldset.appendChild(apply);
    const small = document.createElement("small");
    small.textContent = note;
    fieldset.appendChild(small);
    return fieldset;
  }

  async _apply(key, fields) {
    const values = {};
    for (const [name] of fields) {
      const raw = this.inputs[name].value;
      if (raw === "") continue;
      values[name] = Number(raw);
    }
    await this.put({ [key]: values });
  }

  /** Refresh from a server config event; focused inputs are being edited
   * and are left alone. */
  render(config) {
    for (const [name, value] of Object.entries({ ...config.limits, ...config.structural })) {
      const input = this.inputs[name];
      if (input && document.activeElement !== input) input.value = String(value);
    }
    this.pendingNote.textContent = config.searching
      ? "search running — limit changes apply to the next search"
      : "";
  }
}
