import React, { useEffect, useState } from "react";
import "./PromptEditor.css";

const labels = {
  main_prompt: "Prompt principale",
  prompt_nazionale: "Normativa nazionale",
  prompt_comunale: "Regolamento comunale",
  prompt_aliquote: "Aliquote",
  prompt_file_rimanenti: "Altri documenti",
  prompt_vocab: "Vocabolario",
};

function PromptEditor({
  prompts: externalPrompts = {},
  selected: externalSelected = "",
  onSelectedChange,
  onPromptsUpdate,
}) {
  const [prompts, setPrompts] = useState(externalPrompts);
  const [selected, setSelected] = useState(externalSelected);

  useEffect(() => {
    if (Object.keys(externalPrompts).length > 0) {
      setPrompts(externalPrompts);
      setSelected(externalSelected || Object.keys(externalPrompts)[0] || "");
    }
  }, [externalPrompts, externalSelected]);

  useEffect(() => {
    if (Object.keys(externalPrompts).length === 0) {
      fetch("http://localhost:8000/prompts")
        .then((res) => res.json())
        .then((data) => {
          setPrompts(data);
          setSelected(Object.keys(data)[0] || "");
          onPromptsUpdate?.(data);
          onSelectedChange?.(Object.keys(data)[0] || "");
        });
    }
  }, [externalPrompts, onPromptsUpdate, onSelectedChange]);

  const selectPrompt = (value) => {
    setSelected(value);
    onSelectedChange?.(value);
  };

  const handleChange = (value) => {
    setPrompts({
      ...prompts,
      [selected]: value,
    });
  };

  const changeAll = async () => {
    await fetch("http://localhost:8000/prompts", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(prompts),
    });

    onPromptsUpdate?.(prompts);
    alert("Prompt modificati!");
  };

  const saveAll = async () => {
    await fetch("http://localhost:8000/prompts/save", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(prompts),
    });

    onPromptsUpdate?.(prompts);
    alert("Prompt salvati!");
  };

  const reloadPrompts = async () => {
    const res = await fetch("http://localhost:8000/prompts");
    const data = await res.json();
    setPrompts(data);
    onPromptsUpdate?.(data);
    return data;
  };

  const resetSingle = async () => {
    await fetch(`http://localhost:8000/prompts/reset/${selected}`, {
      method: "POST",
    });

    const data = await reloadPrompts();
    if (!selected) {
      selectPrompt(Object.keys(data)[0] || "");
    }
  };

  const resetAll = async () => {
    if (!window.confirm("Sei sicuro di voler resettare tutto?")) return;
    await fetch("http://localhost:8000/prompts/reset", {
      method: "POST",
    });

    const data = await reloadPrompts();
    selectPrompt(Object.keys(data)[0] || "");
  };

  const showSelect = !onSelectedChange;

  return (
    <div className="prompt-container">
      <h2 className="prompt-title">Editor Prompt</h2>

      {showSelect && (
        <select
          className="prompt-select"
          value={selected}
          onChange={(e) => selectPrompt(e.target.value)}
        >
          {Object.keys(prompts).map((k) => (
            <option key={k} value={k}>
              {labels[k] || k}
            </option>
          ))}
        </select>
      )}

      <textarea
        className="prompt-textarea"
        value={prompts[selected] || ""}
        onChange={(e) => handleChange(e.target.value)}
      />

      <div className="prompt-actions">
        <button className="prompt-button modify" onClick={changeAll}>
          Modifica
        </button>

        <button className="prompt-button save" onClick={saveAll}>
          💾 Salva
        </button>

        <button className="prompt-button reset-single" onClick={resetSingle}>
          Reset
        </button>

        <button className="prompt-button reset-all" onClick={resetAll}>
          Reset tutto
        </button>
      </div>
    </div>
  );
}

export default PromptEditor;
