const form = document.querySelector("#run-form");
const uploadField = document.querySelector("#upload-field");
const archiveInput = document.querySelector("#archive");
const runButton = document.querySelector("#run-button");
const statusPanel = document.querySelector("#status-panel");
const statusMessage = document.querySelector("#status-message");
const runMetrics = document.querySelector("#run-metrics");
const downloadLink = document.querySelector("#download-link");
const graphButton = document.querySelector("#graph-button");
const graphPanel = document.querySelector("#graph-panel");
const graphMessage = document.querySelector("#graph-message");
const graphMetrics = document.querySelector("#graph-metrics");
const graphCanvas = document.querySelector("#graph-canvas");
const graphEdges = document.querySelector("#graph-edges");
const graphNodes = document.querySelector("#graph-nodes");
const nodeDetails = document.querySelector("#node-details");

let documentationGraphUrl = null;
let graphData = null;
let expandedNodes = new Set();
let selectedNodeId = null;

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
  graphButton.hidden = true;
  graphPanel.hidden = true;
  documentationGraphUrl = null;
  runButton.disabled = true;

  try {
    const response = await fetch("/api/v1/runs", { method: "POST", body: payload });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.detail || "The segmentation run failed.");
    }

    statusMessage.textContent = body.status === "completed"
      ? "Segmentation completed. You can now inspect its documentation graph."
      : "Segmentation completed with errors; see errors.json in the result.";
    document.querySelector("#document-count").textContent = body.document_count;
    document.querySelector("#fragment-count").textContent = body.fragment_count;
    document.querySelector("#cache-hits").textContent = body.cache_hits;
    runMetrics.hidden = false;
    downloadLink.href = body.result_url;
    downloadLink.hidden = false;
    documentationGraphUrl = body.documentation_graph_url;
    graphButton.hidden = !documentationGraphUrl;
  } catch (error) {
    statusMessage.textContent = error.message;
  } finally {
    runButton.disabled = false;
  }
});

graphButton.addEventListener("click", async () => {
  if (!documentationGraphUrl) return;

  graphPanel.hidden = false;
  graphMessage.textContent = "Building the documentation graph…";
  graphMetrics.hidden = true;
  graphButton.disabled = true;

  try {
    const response = await fetch(documentationGraphUrl, { method: "POST" });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.detail || "The documentation graph could not be built.");
    }

    graphData = body.graph;
    expandedNodes = new Set();
    selectedNodeId = null;
    document.querySelector("#graph-node-count").textContent = body.node_count;
    document.querySelector("#graph-edge-count").textContent = body.edge_count;
    graphMetrics.hidden = false;
    graphMessage.textContent = graphData.semantic_analysis.message;
    graphButton.textContent = "Reopen documentation graph";
    renderGraph();
    graphPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    graphMessage.textContent = error.message;
  } finally {
    graphButton.disabled = false;
  }
});

function graphIndex() {
  const nodeById = new Map(graphData.nodes.map((node) => [node.id, node]));
  const childrenById = new Map();
  graphData.nodes.forEach((node) => {
    if (!node.parent_id) return;
    if (!childrenById.has(node.parent_id)) childrenById.set(node.parent_id, []);
    childrenById.get(node.parent_id).push(node);
  });
  childrenById.forEach((children) => children.sort((left, right) => {
    const leftLine = left.source.line_start || left.metadata.position || 0;
    const rightLine = right.source.line_start || right.metadata.position || 0;
    return leftLine - rightLine || left.label.localeCompare(right.label);
  }));
  return { nodeById, childrenById };
}

function visibleTree(index) {
  const visible = [];
  const visit = (node) => {
    visible.push(node);
    if (!expandedNodes.has(node.id)) return;
    (index.childrenById.get(node.id) || []).forEach(visit);
  };
  graphData.nodes
    .filter((node) => node.parent_id === null)
    .sort((left, right) => left.metadata.position - right.metadata.position)
    .forEach(visit);
  return visible;
}

function layoutVisibleNodes(index, visibleNodes) {
  const visibleIds = new Set(visibleNodes.map((node) => node.id));
  const positions = new Map();
  const nodeWidth = 260;
  const nodeHeight = 70;
  const xGap = 90;
  const yGap = 22;
  const rootGap = 34;
  let cursorY = 36;
  let maxDepth = 0;

  const place = (node, visibleDepth) => {
    maxDepth = Math.max(maxDepth, visibleDepth);
    const children = (index.childrenById.get(node.id) || [])
      .filter((child) => visibleIds.has(child.id));
    let centerY;
    if (children.length) {
      const childCenters = children.map((child) => place(child, visibleDepth + 1));
      centerY = (childCenters[0] + childCenters[childCenters.length - 1]) / 2;
    } else {
      centerY = cursorY + nodeHeight / 2;
      cursorY += nodeHeight + yGap;
    }
    positions.set(node.id, {
      x: 28 + visibleDepth * (nodeWidth + xGap),
      y: centerY - nodeHeight / 2,
      width: nodeWidth,
      height: nodeHeight,
      centerY,
    });
    return centerY;
  };

  visibleNodes
    .filter((node) => node.parent_id === null)
    .forEach((root) => {
      place(root, 0);
      cursorY += rootGap;
    });

  return {
    positions,
    width: Math.max(760, 56 + (maxDepth + 1) * nodeWidth + maxDepth * xGap),
    height: Math.max(278, cursorY + 16),
  };
}

function edgePath(source, target, type) {
  if (type === "references" && source.x === target.x) {
    const sourceX = source.x + source.width;
    const targetX = target.x + target.width;
    const loopX = sourceX + 42;
    return `M ${sourceX} ${source.centerY} C ${loopX} ${source.centerY}, ${loopX} ${target.centerY}, ${targetX} ${target.centerY}`;
  }
  const sourceX = source.x + source.width;
  const targetX = target.x;
  const middle = (sourceX + targetX) / 2;
  return `M ${sourceX} ${source.centerY} C ${middle} ${source.centerY}, ${middle} ${target.centerY}, ${targetX} ${target.centerY}`;
}

function renderGraph() {
  if (!graphData) return;
  const index = graphIndex();
  const visibleNodes = visibleTree(index);
  const visibleIds = new Set(visibleNodes.map((node) => node.id));
  const layout = layoutVisibleNodes(index, visibleNodes);

  graphCanvas.style.width = `${layout.width}px`;
  graphCanvas.style.height = `${layout.height}px`;
  graphEdges.setAttribute("width", layout.width);
  graphEdges.setAttribute("height", layout.height);
  graphEdges.setAttribute("viewBox", `0 0 ${layout.width} ${layout.height}`);
  graphEdges.replaceChildren();
  graphNodes.replaceChildren();

  graphData.edges.forEach((edge) => {
    if (!visibleIds.has(edge.source) || !visibleIds.has(edge.target)) return;
    const source = layout.positions.get(edge.source);
    const target = layout.positions.get(edge.target);
    if (!source || !target) return;
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    const classNames = ["graph-edge"];
    if (edge.type === "references") classNames.push("graph-edge-reference");
    if (edge.type === "document_order") classNames.push("graph-edge-order");
    path.setAttribute("class", classNames.join(" "));
    path.setAttribute("d", edgePath(source, target, edge.type));
    graphEdges.append(path);
  });

  visibleNodes.forEach((node) => {
    const position = layout.positions.get(node.id);
    const children = index.childrenById.get(node.id) || [];
    const button = document.createElement("button");
    button.type = "button";
    button.className = `graph-node graph-node-${node.type}`;
    if (node.id === selectedNodeId) button.classList.add("selected");
    button.style.left = `${position.x}px`;
    button.style.top = `${position.y}px`;
    button.dataset.nodeId = node.id;
    button.title = children.length
      ? "Double-click to expand or collapse"
      : "Select to inspect source";

    const kind = document.createElement("span");
    kind.className = "node-kind";
    kind.textContent = node.type;
    const label = document.createElement("span");
    label.className = "node-label";
    label.textContent = node.label;
    button.append(kind, label);

    if (children.length) {
      const toggle = document.createElement("span");
      toggle.className = "node-toggle";
      toggle.textContent = expandedNodes.has(node.id) ? "−" : "+";
      button.append(toggle);
    }

    button.addEventListener("click", () => {
      selectedNodeId = node.id;
      showNodeDetails(node);
      graphNodes.querySelectorAll(".graph-node.selected").forEach((selected) => {
        selected.classList.remove("selected");
      });
      button.classList.add("selected");
    });
    button.addEventListener("dblclick", (event) => {
      event.preventDefault();
      if (!children.length) return;
      if (expandedNodes.has(node.id)) {
        collapseNode(node.id, index.childrenById);
      } else {
        expandedNodes.add(node.id);
      }
      renderGraph();
    });
    graphNodes.append(button);
  });
}

function collapseNode(nodeId, childrenById) {
  expandedNodes.delete(nodeId);
  (childrenById.get(nodeId) || []).forEach((child) => collapseNode(child.id, childrenById));
}

function showNodeDetails(node) {
  nodeDetails.hidden = false;
  document.querySelector("#detail-type").textContent = node.type;
  document.querySelector("#detail-label").textContent = node.label;
  document.querySelector("#detail-source").textContent = node.source.path || "—";
  const start = node.source.line_start;
  const end = node.source.line_end;
  document.querySelector("#detail-lines").textContent = start
    ? (end && end !== start ? `${start}–${end}` : String(start))
    : "—";
  document.querySelector("#detail-method").textContent = node.metadata.segmentation_method || "structural";
}

updateSourceUI();
