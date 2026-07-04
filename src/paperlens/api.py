"""FastAPI service for PaperLens."""

from __future__ import annotations

from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse

from paperlens.io import DocumentLoadError, load_document
from paperlens.pipeline import Pipeline
from paperlens.tensorflow_features import TensorFlowUnavailableError

app = FastAPI(title="PaperLens Classifier", version="0.1.0")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PaperLens Classifier</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #17202a;
      --muted: #5f6b7a;
      --line: #d9e0e8;
      --panel: #ffffff;
      --bg: #f4f7fb;
      --accent: #2563eb;
      --accent-dark: #1d4ed8;
      --ok: #0f766e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    main {
      width: min(920px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 56px 0;
    }
    h1 {
      margin: 0 0 10px;
      font-size: clamp(34px, 6vw, 58px);
      line-height: 1;
      letter-spacing: 0;
    }
    p {
      margin: 0;
      color: var(--muted);
      font-size: 17px;
      line-height: 1.55;
    }
    .workspace {
      margin-top: 32px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(280px, 360px);
      gap: 20px;
      align-items: stretch;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 18px 45px rgba(23, 32, 42, 0.08);
    }
    .upload {
      min-height: 360px;
      padding: 28px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 20px;
    }
    .dropzone {
      border: 2px dashed #aebbd0;
      border-radius: 8px;
      min-height: 210px;
      padding: 26px;
      display: grid;
      place-items: center;
      text-align: center;
      background: #f9fbfe;
      transition: border-color 160ms ease, background 160ms ease;
    }
    .dropzone.dragover {
      border-color: var(--accent);
      background: #eef5ff;
    }
    input[type="file"] {
      width: 100%;
      max-width: 360px;
      color: var(--muted);
    }
    button {
      appearance: none;
      border: 0;
      border-radius: 7px;
      background: var(--accent);
      color: white;
      font-weight: 700;
      font-size: 16px;
      min-height: 46px;
      padding: 0 18px;
      cursor: pointer;
    }
    button:hover { background: var(--accent-dark); }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.55;
    }
    .result {
      padding: 24px;
      min-height: 360px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }
    .label {
      display: inline-flex;
      align-items: center;
      width: fit-content;
      min-height: 34px;
      border-radius: 999px;
      padding: 0 12px;
      background: #e7f7f4;
      color: var(--ok);
      font-size: 14px;
      font-weight: 800;
      text-transform: uppercase;
    }
    .answer {
      font-size: clamp(30px, 5vw, 46px);
      line-height: 1.04;
      font-weight: 800;
      letter-spacing: 0;
      overflow-wrap: anywhere;
    }
    .confidence {
      color: var(--muted);
      font-size: 15px;
    }
    .scores {
      display: grid;
      gap: 10px;
      margin-top: auto;
    }
    .score-row {
      display: grid;
      grid-template-columns: 128px 1fr 46px;
      gap: 10px;
      align-items: center;
      font-size: 13px;
      color: var(--muted);
    }
    .bar {
      height: 8px;
      background: #edf1f6;
      border-radius: 999px;
      overflow: hidden;
    }
    .bar span {
      display: block;
      height: 100%;
      background: var(--accent);
    }
    .error {
      color: #b42318;
      font-weight: 700;
    }
    @media (max-width: 760px) {
      main { padding: 32px 0; }
      .workspace { grid-template-columns: 1fr; }
      .score-row { grid-template-columns: 104px 1fr 40px; }
    }
  </style>
</head>
<body>
  <main>
    <h1>PaperLens Classifier</h1>
    <p>
      Upload a document and get its type: invoice, resume, contract, receipt,
      report, scan, or unknown.
    </p>
    <section class="workspace">
      <form class="panel upload" id="uploadForm">
        <div class="dropzone" id="dropzone">
          <div>
            <p><strong>Drop a document here</strong></p>
            <p>PDF, TXT, image, or DOCX if installed with document support</p>
            <br>
            <input id="fileInput" name="file" type="file" required>
          </div>
        </div>
        <button id="submitButton" type="submit">Classify Document</button>
      </form>
      <aside class="panel result" id="result">
        <span class="label">Waiting</span>
        <div class="answer">Upload a document</div>
        <p class="confidence">The answer will appear here after classification.</p>
      </aside>
    </section>
  </main>
  <script>
    const form = document.getElementById("uploadForm");
    const input = document.getElementById("fileInput");
    const dropzone = document.getElementById("dropzone");
    const result = document.getElementById("result");
    const button = document.getElementById("submitButton");

    function prettyLabel(value) {
      return String(value || "unknown").replaceAll("_", " ");
    }

    function renderLoading() {
      result.innerHTML = `
        <span class="label">Reading</span>
        <div class="answer">Classifying...</div>
        <p class="confidence">Extracting text and scoring document type.</p>
      `;
    }

    function renderError(message) {
      result.innerHTML = `
        <span class="label">Error</span>
        <div class="answer">Could not classify</div>
        <p class="error">${message}</p>
      `;
    }

    function renderResult(payload) {
      const classification = payload.classification || {};
      const scores = classification.scores || {};
      const label = prettyLabel(payload.document_type || classification.label);
      const confidence = Math.round((classification.confidence || 0) * 100);
      const rows = Object.entries(scores)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(([name, score]) => {
          const pct = Math.round(score * 100);
          return `
            <div class="score-row">
              <span>${prettyLabel(name)}</span>
              <span class="bar"><span style="width: ${pct}%"></span></span>
              <span>${pct}%</span>
            </div>
          `;
        })
        .join("");

      result.innerHTML = `
        <span class="label">Document Type</span>
        <div class="answer">${label}</div>
        <p class="confidence">
          ${confidence}% confidence for ${payload.filename || "uploaded file"}
        </p>
        <div class="scores">${rows}</div>
      `;
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!input.files.length) {
        renderError("Choose a file first.");
        return;
      }
      const data = new FormData();
      data.append("file", input.files[0]);
      button.disabled = true;
      renderLoading();
      try {
        const response = await fetch("/classify", { method: "POST", body: data });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "The server rejected this file.");
        }
        renderResult(payload);
      } catch (error) {
        renderError(error.message);
      } finally {
        button.disabled = false;
      }
    });

    ["dragenter", "dragover"].forEach((name) => {
      dropzone.addEventListener(name, (event) => {
        event.preventDefault();
        dropzone.classList.add("dragover");
      });
    });
    ["dragleave", "drop"].forEach((name) => {
      dropzone.addEventListener(name, (event) => {
        event.preventDefault();
        dropzone.classList.remove("dragover");
      });
    });
    dropzone.addEventListener("drop", (event) => {
      input.files = event.dataTransfer.files;
    });
  </script>
</body>
</html>
"""


async def _run_classification(file: UploadFile, tensorflow: bool) -> dict[str, object]:
    suffix = ""
    if file.filename and "." in file.filename:
        suffix = "." + file.filename.rsplit(".", 1)[-1]

    with NamedTemporaryFile(delete=True, suffix=suffix) as temp:
        temp.write(await file.read())
        temp.flush()
        try:
            document = load_document(temp.name)
            analyzed = Pipeline(use_tensorflow=tensorflow).run(document)
        except DocumentLoadError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except TensorFlowUnavailableError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = analyzed.model_dump(mode="json")
    result["metadata"]["uploaded_filename"] = file.filename
    return result


@app.post("/classify")
async def classify(
    file: UploadFile = File(...),
    tensorflow: bool = Query(False, description="Enable TensorFlow text features."),
) -> dict[str, object]:
    analyzed = await _run_classification(file=file, tensorflow=tensorflow)
    classification = analyzed["classification"]
    return {
        "filename": file.filename,
        "document_type": classification["label"],
        "answer": f"This looks like a {classification['label'].replace('_', ' ')}.",
        "classification": classification,
        "metadata": analyzed["metadata"],
    }
