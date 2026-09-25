const form = document.querySelector("#run-form");
const uploadField = document.querySelector("#upload-field");
const archiveInput = document.querySelector("#archive");
const runButton = document.querySelector("#run-button");
const statusPanel = document.querySelector("#status-panel");
const statusMessage = document.querySelector("#status-message");
const runMetrics = document.querySelector("#run-metrics");
const downloadLink = document.querySelector("#download-link");

function selectedSource() {
  return form.querySelector('input[name="source"]:checked').value;
}

function updateSourceUI() {
  const source = selectedSource();
  uploadField.hidden = source !== "upload";
  archiveInput.required = source === "upload";
  document.querySelectorAll("[data-source-card]").forEach((card) => {
    card.classList.toggle("selected", card.querySelector("input").checked);
  });
}

form.querySelectorAll('input[name="source"]').forEach((input) => {
  input.addEventListener("change", updateSourceUI);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const source = selectedSource();

  if (source === "upload" && !archiveInput.files.length) {
    archiveInput.reportValidity();
    return;
  }

  const payload = new FormData();
  payload.append("source", source);
  if (source === "upload") {
    payload.append("archive", archiveInput.files[0]);
  }

  statusPanel.hidden = false;
  statusMessage.textContent = "Segmenting documentation…";
  runMetrics.hidden = true;
  downloadLink.hidden = true;
  runButton.disabled = true;

  try {
    const response = await fetch("/api/v1/runs", { method: "POST", body: payload });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.detail || "The segmentation run failed.");
    }

    statusMessage.textContent = body.status === "completed"
      ? "Segmentation completed."
      : "Segmentation completed with errors; see errors.json in the result.";
    document.querySelector("#document-count").textContent = body.document_count;
    document.querySelector("#fragment-count").textContent = body.fragment_count;
    document.querySelector("#cache-hits").textContent = body.cache_hits;
    runMetrics.hidden = false;
    downloadLink.href = body.result_url;
    downloadLink.hidden = false;
  } catch (error) {
    statusMessage.textContent = error.message;
  } finally {
    runButton.disabled = false;
  }
});

updateSourceUI();
