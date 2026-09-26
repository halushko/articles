const form = document.querySelector("#run-form");
const uploadField = document.querySelector("#upload-field");
const archiveInput = document.querySelector("#archive");
const runButton = document.querySelector("#run-button");
const statusPanel = document.querySelector("#status-panel");
const statusMessage = document.querySelector("#status-message");
const runMetrics = document.querySelector("#run-metrics");
const downloadLink = document.querySelector("#download-link");

const documentationButton = document.querySelector("#documentation-button");
const documentationPanel = document.querySelector("#documentation-panel");
const documentationMessage = document.querySelector("#documentation-message");
const documentationMetrics = document.querySelector("#documentation-metrics");
const documentCards = document.querySelector("#document-cards");
const referenceList = document.querySelector("#reference-list");
const documentSelect = document.querySelector("#document-select");
const documentTree = document.querySelector("#document-tree");

const processButton = document.querySelector("#process-button");
const processPanel = document.querySelector("#process-panel");
const processMessage = document.querySelector("#process-message");
const processMetrics = document.querySelector("#process-metrics");
const derivationNote = document.querySelector("#derivation-note");
const processCanvas = document.querySelector("#process-canvas");
const processEdges = document.querySelector("#process-edges");
const processNodes = document.querySelector("#process-nodes");
const processDetails = document.querySelector("#process-details");
const transitionDetails = document.querySelector("#transition-details");
const processWarnings = document.querySelector("#process-warnings");
const levelSummary = document.querySelector("#level-summary");
const aggregationTableBody = document.querySelector("#aggregation-table-body");

let documentationGraphUrl = null;
let processModelUrl = null;
let documentationGraph = null;
let processModel = null;
let processIndexCache = null;
let selectedProcessNodeId = null;
let visibleProcessNodeIds = new Set();

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

function setText(selector, value) {
  document.querySelector(selector).textContent = value;
}

function lineLabel(source) {
  const start = source?.line_start;
  const end = source?.line_end;
  if (!start) return "line not recorded";
  return end && end !== start ? `lines ${start}–${end}` : `line ${start}`;
}

function emptyState(message) {
  const paragraph = document.createElement("p");
  paragraph.className = "context-note";
  paragraph.textContent = message;
  return paragraph;
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
  if (source === "upload") payload.append("archive", archiveInput.files[0]);

  statusPanel.hidden = false;
  statusMessage.textContent = "Segmenting documentation…";
  runMetrics.hidden = true;
  downloadLink.hidden = true;
  documentationButton.hidden = true;
  processButton.hidden = true;
  documentationButton.textContent = "Documentation";
  processButton.textContent = "Process model";
  documentationButton.classList.remove("active");
  processButton.classList.remove("active");
  documentationPanel.hidden = true;
  processPanel.hidden = true;
  documentationGraphUrl = null;
  processModelUrl = null;
  documentationGraph = null;
  processModel = null;
  runButton.disabled = true;

  try {
    const response = await fetch("/api/v1/runs", { method: "POST", body: payload });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "The segmentation run failed.");

    statusMessage.textContent = body.status === "completed"
      ? "Segmentation completed. Inspect the source structure before building the process hierarchy."
      : "Segmentation completed with errors; see errors.json in the result.";
    setText("#document-count", body.document_count);
    setText("#fragment-count", body.fragment_count);
    setText("#cache-hits", body.cache_hits);
    runMetrics.hidden = false;
    downloadLink.href = body.result_url;
    downloadLink.hidden = false;
    documentationGraphUrl = body.documentation_graph_url;
    processModelUrl = body.process_model_url;
    documentationButton.hidden = !documentationGraphUrl;
    processButton.hidden = !processModelUrl;
    processButton.disabled = !body.process_model_available;
    processButton.textContent = body.llm_status === "Без LLM"
      ? "Process model · Без LLM"
      : "Process model";
    if (body.llm_status === "Без LLM") {
      statusMessage.textContent += " Без LLM: process extraction will use deterministic evidence rules.";
    }
    if (!body.process_model_available && body.process_model_unavailable_reason) {
      statusMessage.textContent += ` Process extraction is unavailable: ${body.process_model_unavailable_reason}`;
    }
  } catch (error) {
    statusMessage.textContent = error.message;
  } finally {
    runButton.disabled = false;
  }
});

documentationButton.addEventListener("click", async () => {
  if (!documentationGraphUrl) return;
  documentationPanel.hidden = false;
  processPanel.hidden = true;
  documentationButton.classList.add("active");
  documentationButton.setAttribute("aria-selected", "true");
  processButton.classList.remove("active");
  processButton.setAttribute("aria-selected", "false");
  documentationMessage.textContent = "Reading document structure and explicit references…";
  documentationMetrics.hidden = true;
  documentationButton.disabled = true;
  try {
    const response = await fetch(documentationGraphUrl, { method: "POST" });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "The documentation view could not be built.");
    documentationGraph = body.graph;
    renderDocumentationExplorer();
    documentationButton.textContent = "Documentation";
    documentationPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    documentationMessage.textContent = error.message;
  } finally {
    documentationButton.disabled = false;
  }
});

function documentationIndex() {
  const nodeById = new Map(documentationGraph.nodes.map((node) => [node.id, node]));
  const childrenById = new Map();
  documentationGraph.nodes.forEach((node) => {
    if (!node.parent_id) return;
    if (!childrenById.has(node.parent_id)) childrenById.set(node.parent_id, []);
    childrenById.get(node.parent_id).push(node);
  });
  childrenById.forEach((children) => children.sort((left, right) => {
    const leftLine = Number(left.source?.line_start || left.metadata?.position || 0);
    const rightLine = Number(right.source?.line_start || right.metadata?.position || 0);
    return leftLine - rightLine || left.label.localeCompare(right.label);
  }));
  const documents = documentationGraph.nodes
    .filter((node) => node.type === "document")
    .sort((left, right) => left.metadata.position - right.metadata.position);
  return { nodeById, childrenById, documents };
}

function renderDocumentationExplorer() {
  const index = documentationIndex();
  const references = documentationGraph.edges.filter((edge) => edge.type === "references");
  setText("#documentation-count", index.documents.length);
  setText("#reference-count", references.length);
  documentationMetrics.hidden = false;
  documentationMessage.textContent = documentationGraph.semantic_analysis.message;

  documentCards.replaceChildren();
  documentSelect.replaceChildren();
  index.documents.forEach((doc) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "document-card";
    const title = document.createElement("strong");
    title.textContent = doc.label;
    const meta = document.createElement("small");
    meta.textContent = `${doc.metadata.leaf_fragment_count} leaf fragments · ${doc.source.path}`;
    card.append(title, meta);
    card.addEventListener("click", () => openDocumentStructure(doc.id));
    documentCards.append(card);

    const option = document.createElement("option");
    option.value = doc.id;
    option.textContent = doc.label;
    documentSelect.append(option);
  });

  referenceList.replaceChildren();
  if (!references.length) {
    referenceList.append(emptyState("No explicit references between document titles were found."));
  } else {
    references.forEach((edge) => {
      const source = index.nodeById.get(edge.source);
      const target = index.nodeById.get(edge.target);
      const item = document.createElement("details");
      item.className = "reference-item";
      const summary = document.createElement("summary");
      const sourceName = document.createElement("strong");
      sourceName.textContent = source?.label || edge.source;
      const arrow = document.createElement("span");
      arrow.className = "reference-arrow";
      arrow.textContent = "→";
      const targetName = document.createElement("strong");
      targetName.textContent = target?.label || edge.target;
      const badge = document.createElement("span");
      badge.className = "reference-badge";
      badge.textContent = `${edge.evidence.length} source ${edge.evidence.length === 1 ? "quote" : "quotes"}`;
      summary.append(sourceName, arrow, targetName, badge);
      item.append(summary);

      const evidenceBox = document.createElement("div");
      evidenceBox.className = "reference-evidence";
      edge.evidence.forEach((evidence) => {
        const quote = document.createElement("p");
        quote.className = "evidence-quote";
        quote.textContent = `“${evidence.quote}”`;
        const meta = document.createElement("div");
        meta.className = "source-meta";
        const sourceNode = index.nodeById.get(evidence.node_id);
        meta.textContent = `${source?.source.path || "source document"} · ${lineLabel(sourceNode?.source)}`;
        evidenceBox.append(quote, meta);
      });
      item.append(evidenceBox);
      referenceList.append(item);
    });
  }

  if (index.documents.length) {
    documentSelect.value = index.documents[0].id;
    renderDocumentTree(index.documents[0].id);
  }
}

document.querySelectorAll("[data-documentation-view]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-documentation-view]").forEach((candidate) => {
      candidate.classList.toggle("active", candidate === button);
    });
    const structure = button.dataset.documentationView === "structure";
    document.querySelector("#documentation-map-view").hidden = structure;
    document.querySelector("#documentation-structure-view").hidden = !structure;
  });
});

documentSelect.addEventListener("change", () => renderDocumentTree(documentSelect.value));

function openDocumentStructure(documentId) {
  documentSelect.value = documentId;
  renderDocumentTree(documentId);
  document.querySelector('[data-documentation-view="structure"]').click();
}

function fragmentCard(node) {
  const card = document.createElement("article");
  card.className = "fragment-card";
  const text = document.createElement("p");
  text.textContent = node.label;
  const meta = document.createElement("div");
  meta.className = "fragment-meta";
  const type = document.createElement("span");
  type.textContent = node.metadata.fragment_type || "fragment";
  const lines = document.createElement("span");
  lines.textContent = lineLabel(node.source);
  const method = document.createElement("span");
  method.textContent = node.metadata.segmentation_method || "structural";
  meta.append(type, lines, method);
  card.append(text, meta);
  return card;
}

function sectionBranch(node, index) {
  const branch = document.createElement("details");
  branch.className = "section-branch";
  branch.open = node.depth <= 1;
  const summary = document.createElement("summary");
  summary.textContent = node.label;
  branch.append(summary);
  const fragments = document.createElement("div");
  fragments.className = "fragment-list";
  (index.childrenById.get(node.id) || []).forEach((child) => {
    fragments.append(child.type === "section" ? sectionBranch(child, index) : fragmentCard(child));
  });
  branch.append(fragments);
  return branch;
}

function renderDocumentTree(documentId) {
  if (!documentationGraph) return;
  const index = documentationIndex();
  const doc = index.nodeById.get(documentId);
  documentTree.replaceChildren();
  if (!doc) return;
  const children = index.childrenById.get(documentId) || [];
  const rootFragments = children.filter((node) => node.type === "fragment");
  if (rootFragments.length) {
    const rootBox = document.createElement("div");
    rootBox.className = "document-root-fragments fragment-list";
    rootFragments.forEach((fragment) => rootBox.append(fragmentCard(fragment)));
    documentTree.append(rootBox);
  }
  children
    .filter((node) => node.type === "section")
    .forEach((section) => documentTree.append(sectionBranch(section, index)));
  if (!children.length) documentTree.append(emptyState("This document contains no active leaf fragments."));
}

processButton.addEventListener("click", async () => {
  if (!processModelUrl) return;
  processPanel.hidden = false;
  documentationPanel.hidden = true;
  processButton.classList.add("active");
  processButton.setAttribute("aria-selected", "true");
  documentationButton.classList.remove("active");
  documentationButton.setAttribute("aria-selected", "false");
  processMessage.textContent = "Loading or building a grounded process hierarchy…";
  processMetrics.hidden = true;
  processButton.disabled = true;
  try {
    let response = await fetch(processModelUrl);
    let body = await response.json();
    if (response.status === 404) {
      response = await fetch(processModelUrl, { method: "POST" });
      body = await response.json();
    }
    if (!response.ok) throw new Error(body.detail || "The process hierarchy could not be built.");
    showProcessModel(body);
    processPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    processMessage.textContent = error.message;
  } finally {
    processButton.disabled = false;
  }
});

function showProcessModel(body) {
  processModel = body.process_model;
  derivationNote.textContent = processModel.derivation.message;
  setText(
    "#process-title",
    processModel.process_title || "Hierarchical process model",
  );
  processMessage.textContent = body.llm_status === "Без LLM"
    ? "Без LLM · deterministic model. Inspect warnings and source evidence before accepting inferred stages."
    : "Choose a level, expand stages, or select an arrow to inspect why the transition exists.";
  renderProcessWarnings();
  renderLevelControls();
  const availableLevels = [...processGraphs().keys()];
  setExactProcessLevel(Math.max(...availableLevels));
  renderAggregationTable();
  processButton.textContent = body.llm_status === "Без LLM"
    ? "Process model · Без LLM"
    : "Process model";
}

function processGraphs() {
  const graphs = new Map([[0, processModel.hierarchy.base_graph]]);
  processModel.hierarchy.levels.forEach((level) => graphs.set(level.target_level, level.graph));
  return graphs;
}

function renderProcessWarnings() {
  processWarnings.replaceChildren();
  const warnings = processModel.warnings || [];
  if (!warnings.length) {
    processWarnings.hidden = true;
    return;
  }
  const strong = document.createElement("strong");
  strong.textContent = "Extraction warnings";
  const list = document.createElement("ul");
  warnings.forEach((warning) => {
    const item = document.createElement("li");
    item.textContent = warning;
    list.append(item);
  });
  processWarnings.append(strong, list);
  processWarnings.hidden = false;
}

function renderLevelControls() {
  const summaryByLevel = new Map(
    (processModel.summary || []).map((item) => [item.level, item]),
  );
  levelSummary.replaceChildren();
  [...summaryByLevel.keys()].sort((left, right) => left - right).forEach((level) => {
    const summary = summaryByLevel.get(level);
    const item = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = `L${level}`;
    const counts = document.createElement("small");
    counts.textContent = `${summary.node_count} nodes · ${summary.edge_count} transitions`;
    item.append(title, counts);
    levelSummary.append(item);
  });
  document.querySelectorAll("[data-process-level]").forEach((button) => {
    button.hidden = !summaryByLevel.has(Number(button.dataset.processLevel));
  });
}

function atomicNumber(nodeId) {
  const match = /^v(\d+)$/.exec(nodeId);
  return match ? Number(match[1]) : Number.MAX_SAFE_INTEGER;
}

function processNodeOrder(leftId, rightId) {
  const left = processIndexCache?.nodeById.get(leftId);
  const right = processIndexCache?.nodeById.get(rightId);
  const leftMember = Math.min(...(left?.member_ids || [leftId]).map(atomicNumber));
  const rightMember = Math.min(...(right?.member_ids || [rightId]).map(atomicNumber));
  return leftMember - rightMember || leftId.localeCompare(rightId);
}

function processIndex() {
  const graphs = processGraphs();
  const nodeById = new Map();
  graphs.forEach((graph) => graph.nodes.forEach((node) => nodeById.set(node.id, node)));
  const childrenByParent = new Map();
  const parentByChild = new Map();
  processModel.hierarchy.levels.forEach((level) => {
    Object.entries(level.mapping).forEach(([child, parent]) => {
      if (!childrenByParent.has(parent)) childrenByParent.set(parent, []);
      childrenByParent.get(parent).push(child);
      parentByChild.set(child, parent);
    });
  });
  const scoreByNode = new Map();
  processModel.hierarchy.levels.forEach((level) => {
    level.accepted_candidates.forEach((candidate) => {
      const target = level.mapping[candidate.visible_node_ids[0]];
      if (target) scoreByNode.set(target, candidate);
    });
  });
  return { graphs, nodeById, childrenByParent, parentByChild, scoreByNode };
}

function processNodeKind(nodeId, children) {
  if (children.length) return "stage";
  return processModel.node_metadata?.[nodeId]?.node_type === "gateway"
    ? "gateway"
    : "action";
}

document.querySelectorAll("[data-process-level]").forEach((button) => {
  button.addEventListener("click", () => setExactProcessLevel(Number(button.dataset.processLevel)));
});

function setExactProcessLevel(level) {
  if (!processModel) return;
  processIndexCache = processIndex();
  processIndexCache.childrenByParent.forEach((children) => children.sort(processNodeOrder));
  const graph = processIndexCache.graphs.get(level);
  if (!graph) return;
  visibleProcessNodeIds = new Set(graph.nodes.map((node) => node.id));
  selectedProcessNodeId = null;
  processDetails.hidden = true;
  transitionDetails.hidden = true;
  document.querySelectorAll("[data-process-level]").forEach((button) => {
    button.classList.toggle("active", Number(button.dataset.processLevel) === level);
  });
  renderProcessGraph();
}

function expandProcessNode(nodeId) {
  const children = processIndexCache.childrenByParent.get(nodeId) || [];
  if (!children.length) return;
  visibleProcessNodeIds.delete(nodeId);
  children.forEach((child) => visibleProcessNodeIds.add(child));
  selectedProcessNodeId = null;
  processDetails.hidden = true;
  transitionDetails.hidden = true;
  document.querySelectorAll("[data-process-level]").forEach((button) => button.classList.remove("active"));
  renderProcessGraph();
}

function collapseToProcessParent(parentId) {
  const parent = processIndexCache.nodeById.get(parentId);
  if (!parent) return;
  const members = new Set(parent.member_ids);
  [...visibleProcessNodeIds].forEach((visibleId) => {
    const visible = processIndexCache.nodeById.get(visibleId);
    if (visible && visible.member_ids.every((member) => members.has(member))) {
      visibleProcessNodeIds.delete(visibleId);
    }
  });
  visibleProcessNodeIds.add(parentId);
  selectedProcessNodeId = parentId;
  document.querySelectorAll("[data-process-level]").forEach((button) => button.classList.remove("active"));
  renderProcessGraph();
  showProcessDetails(parentId);
}

function visibleRepresentative(atomicId) {
  for (const visibleId of visibleProcessNodeIds) {
    const visible = processIndexCache.nodeById.get(visibleId);
    if (visible?.member_ids.includes(atomicId)) return visibleId;
  }
  return null;
}

function visibleProcessEdges() {
  const groups = new Map();
  processModel.hierarchy.base_graph.edges.forEach((edge) => {
    const source = visibleRepresentative(edge.source);
    const target = visibleRepresentative(edge.target);
    if (!source || !target || source === target) return;
    const key = `${source}\u001f${target}`;
    if (!groups.has(key)) {
      groups.set(key, { source, target, types: new Set(), conditions: new Set(), originalIds: [] });
    }
    const group = groups.get(key);
    group.types.add(edge.edge_type);
    if (edge.condition) group.conditions.add(edge.condition);
    group.originalIds.push(...edge.original_edge_ids);
  });
  return [...groups.values()].map((group, index) => {
    const originalIds = [...new Set(group.originalIds)];
    const confidences = originalIds
      .map((edgeId) => processModel.transition_provenance?.[edgeId]?.confidence)
      .filter((value) => typeof value === "number");
    return {
      id: `visible-edge-${index + 1}`,
      source: group.source,
      target: group.target,
      edgeType: group.types.size === 1 ? [...group.types][0] : "mixed",
      condition: [...group.conditions].join(" | "),
      originalIds,
      confidence: confidences.length ? Math.min(...confidences) : null,
    };
  });
}

function processLayout(nodeIds, edges) {
  const outgoing = new Map(nodeIds.map((id) => [id, []]));
  const indegree = new Map(nodeIds.map((id) => [id, 0]));
  edges.forEach((edge) => {
    outgoing.get(edge.source).push(edge.target);
    indegree.set(edge.target, indegree.get(edge.target) + 1);
  });
  const queue = nodeIds.filter((id) => indegree.get(id) === 0).sort(processNodeOrder);
  const layers = new Map(nodeIds.map((id) => [id, 0]));
  const visited = new Set();
  while (queue.length) {
    const current = queue.shift();
    visited.add(current);
    (outgoing.get(current) || []).forEach((target) => {
      layers.set(target, Math.max(layers.get(target), layers.get(current) + 1));
      indegree.set(target, indegree.get(target) - 1);
      if (indegree.get(target) === 0) {
        queue.push(target);
        queue.sort(processNodeOrder);
      }
    });
  }
  nodeIds.filter((id) => !visited.has(id)).sort(processNodeOrder).forEach((id, index) => {
    layers.set(id, Math.max(layers.get(id), index));
  });

  const byLayer = new Map();
  nodeIds.forEach((id) => {
    const layer = layers.get(id);
    if (!byLayer.has(layer)) byLayer.set(layer, []);
    byLayer.get(layer).push(id);
  });
  byLayer.forEach((ids) => ids.sort(processNodeOrder));

  const width = 230;
  const height = 94;
  const horizontalGap = 150;
  const verticalGap = 42;
  const margin = 34;
  const maxRows = Math.max(...[...byLayer.values()].map((ids) => ids.length));
  const positions = new Map();
  byLayer.forEach((ids, layer) => {
    const columnHeight = ids.length * height + Math.max(0, ids.length - 1) * verticalGap;
    const totalHeight = maxRows * height + Math.max(0, maxRows - 1) * verticalGap;
    const startY = margin + (totalHeight - columnHeight) / 2;
    ids.forEach((id, row) => {
      positions.set(id, {
        x: margin + layer * (width + horizontalGap),
        y: startY + row * (height + verticalGap),
        width,
        height,
      });
    });
  });
  const maxLayer = Math.max(...layers.values());
  return {
    positions,
    width: Math.max(820, margin * 2 + (maxLayer + 1) * width + maxLayer * horizontalGap),
    height: Math.max(360, margin * 2 + maxRows * height + Math.max(0, maxRows - 1) * verticalGap),
  };
}

function svgElement(name, attributes = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  return element;
}

function addArrowMarkers() {
  const defs = svgElement("defs");
  [["arrow-sequence", "#8491a7"], ["arrow-conditional", "#c47c17"]].forEach(([id, color]) => {
    const marker = svgElement("marker", {
      id,
      viewBox: "0 0 10 10",
      refX: "9",
      refY: "5",
      markerWidth: "7",
      markerHeight: "7",
      orient: "auto-start-reverse",
    });
    marker.append(svgElement("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: color }));
    defs.append(marker);
  });
  processEdges.append(defs);
}

function renderProcessEdge(edge, layout) {
  const source = layout.positions.get(edge.source);
  const target = layout.positions.get(edge.target);
  if (!source || !target) return;
  const sourceX = source.x + source.width;
  const sourceY = source.y + source.height / 2;
  const targetX = target.x;
  const targetY = target.y + target.height / 2;
  const middleX = sourceX + (targetX - sourceX) / 2;
  const isConditional = edge.edgeType !== "sequence";
  const confidenceClass = edge.confidence === null || edge.confidence >= 0.8
    ? "process-edge-confirmed"
    : edge.confidence >= 0.5
      ? "process-edge-inferred"
      : "process-edge-very-low";
  const pathDefinition = `M ${sourceX} ${sourceY} C ${middleX} ${sourceY}, ${middleX} ${targetY}, ${targetX} ${targetY}`;
  const selectTransition = (event) => {
    event.stopPropagation();
    showTransitionDetails(edge);
  };
  const hitPath = svgElement("path", {
    class: "process-edge-hit",
    d: pathDefinition,
    tabindex: "0",
    role: "button",
    "aria-label": `Inspect transition ${edge.source} to ${edge.target}; confidence ${edge.confidence ?? "not recorded"}`,
  });
  hitPath.addEventListener("click", selectTransition);
  hitPath.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") selectTransition(event);
  });
  const path = svgElement("path", {
    class: `process-edge ${confidenceClass}${isConditional ? " process-edge-conditional" : ""}`,
    d: pathDefinition,
  });
  const title = svgElement("title");
  title.textContent = `${edge.source} → ${edge.target}; ${edge.edgeType}; confidence ${edge.confidence?.toFixed(2) ?? "not recorded"}; ${edge.condition || "no condition"}; source transitions: ${edge.originalIds.join(", ")}`;
  path.append(title);
  path.addEventListener("click", selectTransition);
  processEdges.append(hitPath, path);

  if (edge.condition) {
    const labelX = middleX;
    const labelY = (sourceY + targetY) / 2;
    const visibleLabel = edge.condition.length > 30
      ? `${edge.condition.slice(0, 29)}…`
      : edge.condition;
    const labelWidth = Math.max(54, visibleLabel.length * 6.2 + 16);
    processEdges.append(svgElement("rect", {
      class: "process-edge-label-bg",
      x: String(labelX - labelWidth / 2),
      y: String(labelY - 11),
      width: String(labelWidth),
      height: "22",
      rx: "6",
    }));
    const label = svgElement("text", {
      class: "process-edge-label",
      x: String(labelX),
      y: String(labelY),
    });
    label.textContent = visibleLabel;
    processEdges.append(label);
  }
}

function renderProcessGraph() {
  if (!processModel) return;
  if (!processIndexCache) processIndexCache = processIndex();
  const nodeIds = [...visibleProcessNodeIds].sort(processNodeOrder);
  const edges = visibleProcessEdges();
  const layout = processLayout(nodeIds, edges);

  processCanvas.style.width = `${layout.width}px`;
  processCanvas.style.height = `${layout.height}px`;
  processEdges.setAttribute("width", layout.width);
  processEdges.setAttribute("height", layout.height);
  processEdges.setAttribute("viewBox", `0 0 ${layout.width} ${layout.height}`);
  processEdges.replaceChildren();
  processNodes.replaceChildren();
  addArrowMarkers();
  edges.forEach((edge) => renderProcessEdge(edge, layout));

  nodeIds.forEach((nodeId) => {
    const node = processIndexCache.nodeById.get(nodeId);
    const position = layout.positions.get(nodeId);
    const children = processIndexCache.childrenByParent.get(nodeId) || [];
    const parent = processIndexCache.parentByChild.get(nodeId);
    const kind = processNodeKind(nodeId, children);
    const box = document.createElement("div");
    box.className = `process-node process-node-${kind}`;
    if (nodeId === selectedProcessNodeId) box.classList.add("selected");
    box.style.left = `${position.x}px`;
    box.style.top = `${position.y}px`;
    box.dataset.nodeId = nodeId;
    box.tabIndex = 0;
    box.setAttribute("role", "button");
    box.title = children.length ? "Double-click to expand this stage" : "Select to inspect source evidence";

    const type = document.createElement("span");
    type.className = "process-node-kind";
    type.textContent = `${kind} · ${nodeId}`;
    const label = document.createElement("span");
    label.className = "process-node-label";
    label.textContent = node.operation;
    const members = document.createElement("span");
    members.className = "process-node-members";
    members.textContent = node.member_ids.length === 1 ? "1 atomic action" : `${node.member_ids.length} atomic actions`;
    box.append(type, label, members);

    if (children.length) {
      const expand = document.createElement("button");
      expand.type = "button";
      expand.className = "process-expand";
      expand.textContent = "+";
      expand.title = "Expand one level";
      expand.addEventListener("click", (event) => {
        event.stopPropagation();
        expandProcessNode(nodeId);
      });
      box.append(expand);
    }
    if (parent && !visibleProcessNodeIds.has(parent)) {
      const collapse = document.createElement("button");
      collapse.type = "button";
      collapse.className = "process-collapse";
      collapse.textContent = "↑";
      collapse.title = "Collapse parent stage";
      collapse.addEventListener("click", (event) => {
        event.stopPropagation();
        collapseToProcessParent(parent);
      });
      box.append(collapse);
    }
    box.addEventListener("click", () => {
      selectedProcessNodeId = nodeId;
      showProcessDetails(nodeId);
      processNodes.querySelectorAll(".selected").forEach((selected) => selected.classList.remove("selected"));
      box.classList.add("selected");
    });
    box.addEventListener("dblclick", (event) => {
      event.preventDefault();
      if (children.length) expandProcessNode(nodeId);
      else if (parent) collapseToProcessParent(parent);
    });
    box.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectedProcessNodeId = nodeId;
        showProcessDetails(nodeId);
      }
    });
    processNodes.append(box);
  });

  setText("#visible-process-nodes", nodeIds.length);
  setText("#visible-process-edges", edges.length);
  processMetrics.hidden = false;
}

function showProcessDetails(nodeId) {
  const node = processIndexCache.nodeById.get(nodeId);
  if (!node) return;
  const children = processIndexCache.childrenByParent.get(nodeId) || [];
  const kind = processNodeKind(nodeId, children);
  const type = kind === "stage"
    ? "Aggregated stage"
    : (kind === "gateway" ? "Decision gateway" : "Atomic action");
  setText("#process-detail-type", `${type} · ${nodeId}`);
  setText("#process-detail-label", node.operation);
  setText("#process-detail-role", node.role);
  setText("#process-detail-system", node.system);
  setText("#process-detail-members", node.member_ids.join(", "));
  const score = processIndexCache.scoreByNode.get(nodeId);
  setText("#process-detail-score", score ? `Q = ${score.q.toFixed(3)}` : "not aggregated");

  const evidenceBox = document.querySelector("#process-evidence");
  evidenceBox.replaceChildren();
  node.member_ids.forEach((atomicId) => {
    const atomicNode = processIndexCache.nodeById.get(atomicId);
    (processModel.provenance[atomicId] || []).forEach((evidence) => {
      const article = document.createElement("article");
      const quote = document.createElement("p");
      quote.textContent = `“${evidence.quote}”`;
      const meta = document.createElement("div");
      meta.className = "source-meta";
      meta.textContent = `${atomicId} ${atomicNode?.operation || ""} · ${evidence.document_path} · ${lineLabel(evidence)}`;
      article.append(quote, meta);
      evidenceBox.append(article);
    });
  });
  if (!evidenceBox.children.length) evidenceBox.append(emptyState("No source evidence is recorded for this node."));
  transitionDetails.hidden = true;
  processDetails.hidden = false;
}

function showTransitionDetails(edge) {
  const source = processIndexCache.nodeById.get(edge.source);
  const target = processIndexCache.nodeById.get(edge.target);
  setText(
    "#transition-detail-label",
    `${source?.operation || edge.source} → ${target?.operation || edge.target}`,
  );
  setText("#transition-detail-type", edge.edgeType);
  setText("#transition-detail-condition", edge.condition || "none");

  const records = edge.originalIds
    .map((edgeId) => processModel.transition_provenance?.[edgeId])
    .filter(Boolean);
  const confidences = records
    .map((record) => record.confidence)
    .filter((value) => typeof value === "number");
  setText(
    "#transition-detail-confidence",
    confidences.length
      ? `${Math.min(...confidences).toFixed(2)}–${Math.max(...confidences).toFixed(2)}`
      : "not recorded",
  );

  const reasons = document.querySelector("#transition-reasons");
  reasons.replaceChildren();
  const uniqueReasons = [...new Set(records.map((record) => record.reason).filter(Boolean))];
  if (!uniqueReasons.length) {
    uniqueReasons.push("This transition is defined by the selected process graph.");
  }
  uniqueReasons.forEach((reason) => {
    const paragraph = document.createElement("p");
    paragraph.textContent = reason;
    reasons.append(paragraph);
  });

  const evidenceBox = document.querySelector("#transition-evidence");
  evidenceBox.replaceChildren();
  const evidenceItems = records.flatMap((record) => record.evidence || []);
  const evidenceKeys = new Set();
  evidenceItems.forEach((evidence) => {
    const key = `${evidence.document_path}\u001f${evidence.fragment_id}`;
    if (evidenceKeys.has(key)) return;
    evidenceKeys.add(key);
    const article = document.createElement("article");
    const quote = document.createElement("p");
    quote.textContent = `“${evidence.quote}”`;
    const meta = document.createElement("div");
    meta.className = "source-meta";
    meta.textContent = `${evidence.document_path} · ${lineLabel(evidence)}`;
    article.append(quote, meta);
    evidenceBox.append(article);
  });
  if (!evidenceBox.children.length) {
    evidenceBox.append(emptyState("No transition-specific source evidence is recorded."));
  }
  selectedProcessNodeId = null;
  processNodes.querySelectorAll(".selected").forEach((node) => node.classList.remove("selected"));
  processDetails.hidden = true;
  transitionDetails.hidden = false;
}

function renderAggregationTable() {
  aggregationTableBody.replaceChildren();
  processModel.hierarchy.levels.forEach((level) => {
    const rows = [
      ...level.accepted_candidates.map((candidate) => ({ candidate, accepted: true })),
      ...level.rejected_candidates
        .filter((candidate) => candidate.candidate_id || candidate.rejection_reason === "branch_integrity")
        .map((candidate) => ({ candidate, accepted: false })),
    ];
    rows.forEach(({ candidate, accepted }) => {
      const row = document.createElement("tr");
      const values = [
        `L${level.source_level} → L${level.target_level}`,
        candidate.candidate_id || candidate.candidate_name || "generated",
        candidate.atomic_node_ids.join(", "),
        candidate.s_txt.toFixed(3),
        candidate.s_ctx.toFixed(3),
        candidate.s_flow.toFixed(3),
        candidate.q.toFixed(3),
      ];
      values.forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      });
      const decision = document.createElement("td");
      decision.className = accepted ? "decision-accepted" : "decision-rejected";
      decision.textContent = accepted
        ? `aggregated at L${level.target_level}`
        : `rejected: ${(candidate.rejection_reason || "not selected").replaceAll("_", " ")}`;
      row.append(decision);
      aggregationTableBody.append(row);
    });
  });
}

updateSourceUI();
