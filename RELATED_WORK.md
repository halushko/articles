# Preliminary related work

- Qiu et al., ICLR 2024, https://arxiv.org/abs/2310.08559 — hypothesis induction vs application gap; symbolic interpretation.
- Tang et al., NeurIPS 2024, https://arxiv.org/abs/2402.12275 — executable world models from interaction and transfer.
- Jansen et al., NeurIPS 2024, https://allenai.github.io/discoveryworld/ — scientific discovery benchmark, separate knowledge and action metrics.
- Shinn et al., NeurIPS 2023, https://arxiv.org/abs/2303.11366 — verbal feedback and episodic memory without weight updates.
- Wang et al., ICLR 2024, https://proceedings.iclr.cc/paper_files/paper/2024/hash/a934282496e7b907f5d48d49bb4c9d7d-Abstract-Conference.html — candidate hypotheses, program execution and validation.
- Angluin, 1987, https://homepages.math.uic.edu/~lreyzin/papers/angluin87.pdf — active automata learning with membership and equivalence queries; different oracle assumptions.
- SHELDON, preprint 2026, https://sheldon.genie.stanford.edu/ — separate hypothesis and observation search.

Novelty not established. Investigate process documentation grounding, small fixed-weight models, prediction vs hypothesis consistency, transfer to unseen dynamics. Compare random and observation-table baselines, V1, thinking, separated calls, structured memory, multiple hypotheses and executable rules. Control interaction and token budgets. This is a preliminary abstract-based review, not a systematic literature review.

## Batch operation

Commit 5a7aea093268be63589ccb6e23957b45fba126ad adds server-side sequential batches. Fill the existing form, click + Experiment, repeat, then Run batch. Queue supports moving, duplicating and removing entries. Completed batches export manifest.json and numbered full run JSONs in a ZIP. Closing the browser does not stop the server worker. Server restart marks unfinished runs interrupted; no auto-resume. Model is configured globally via Compose. Draft queue is browser-local until submission. Docker and tests were not executed in the editing session.
