# Process learning laboratory — experiment 01

A small LLM investigates an unknown deterministic finite-state system through interventions and observations. This is a baseline for external-memory learning, not training of model weights.

## Run

Requires Docker with Compose v2. CPU-only works but can be slow. The default model is Qwen3:4B; Ollama downloads its weights on the first pull.

    docker compose up -d --build

Open http://localhost:8765 after model-init finishes. The initializer waits for Ollama and automatically downloads the configured model if missing. Set MODEL in .env to change it. The UI binds to loopback only. For remote use, forward port 8765 over SSH rather than publicly exposing it.

Select w01, profile default, 12 steps, seed 42, event probability 0.25, clean start. Start the experiment. The dashboard shows the actual model predictions, observations, evolving hypotheses, baseline and final test scores. Download a full JSON report. Repeat with reuse enabled and the same profile/world to test persistence. A clean start ignores previously saved knowledge but replaces the saved snapshot on successful completion; use different profiles to keep independent replicates.

## Worlds

The agent sees only possible output values, legal action names and possible environmental event names. At each step the environment independently decides whether an event occurs. If so, the agent sees its name before predicting the effect; otherwise the agent chooses an action and predicts its effect. Each turn has one input, not simultaneous events. WAIT is a normal action if the world exposes it. The agent does not receive hidden transition tables.

- w01: one binary observable, A toggles, WAIT leaves unchanged, E resets to zero.
- w02: three output values, two different actions plus WAIT, and two external event types.

Add a JSON file to worlds/ with a unique id, an initial state, outputs mapping state names to unique integers, and complete actions/events transition tables. Restart world to load changes. Use new world IDs for changed laws to avoid stale knowledge. v1 intentionally supports deterministic systems with uniquely observable states, no hidden-state ambiguity, time dynamics or simultaneous inputs. The world API also provides private audit/evaluation data to the laboratory.

## Measurement

Baseline and final evaluation independently query every one-step state/input pair, with no answer feedback provided to the model. The fixed test suite includes transitions possibly observed during exploration: it measures one-step prediction coverage, NOT guaranteed generalization to unseen worlds. The final test uses only the consolidated hypothesis, not the raw journal. Baseline uses either an empty hypothesis or the selected saved hypothesis. Accuracy is number of exact predictions divided by case count. Online predictions are recorded before actions are executed. Responses use strict JSON Schema and independent validation. Up to three attempts are allowed for invalid responses, with format-only feedback and no correct answers. Every attempt records its request, raw response and validation result; exhausted retries mark the run as an error. No fabricated predictions or automatic substitution. Compare multiple seeds/profiles; one successful run does not establish learning.

The seed controls the external schedule and is passed to Ollama; exact reproducibility across hardware/runtime versions is not guaranteed. Pin the Ollama image and model digest before formal experiments. The report records the world law, config, hypotheses, observations and model token counts. Only the researcher sees the audit fields.

## Architecture and persistence

Four runtime services plus model-init: Ollama (inference), agent (JSON-only LLM interface), world (private transition engine), laboratory (orchestration, evaluation, UI and SQLite). Separate Docker networks prevent the agent from directly reaching the world; no Docker socket or host filesystem is mounted into the agent. Model and experiment data use named volumes. This is application-level separation, not a hardened adversarial sandbox.

    docker compose down
    docker compose up -d

Data survives ordinary down/up. Do not use down -v unless you want to delete both experience and downloaded models. The run journal is saved after every step. A restart marks unfinished runs interrupted; it does not resume them automatically. Knowledge snapshots are saved on successful completion and keyed by profile, world and configured model name. Recent six observations plus a compact hypothesis are supplied during exploration. No fine-tuning, LoRA, or weight updates are implemented in v1.

## Tests and troubleshooting

    python -m unittest discover -s tests -v
    docker compose logs -f laboratory agent world
    docker compose exec ollama ollama list

Tests cover transition rules, public interface, fake-agent experiment persistence and absence of expected values in evaluation prompts. Real LLM inference must be tested on your machine. If a model exhausts format retries, inspect llm_calls in the JSON report and agent logs. Use a new run after resolving it. One experiment runs at a time.
