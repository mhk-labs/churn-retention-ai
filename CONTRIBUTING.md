# Working together

1. Pick one task from docs/BACKLOG.md and create a GitHub issue with an owner.
2. Pull main and create a branch, e.g. `git switch -c feat/listening-features`.
3. Make one focused change, add meaningful tests, and update affected documentation.
4. Run `python -m unittest discover -s tests -v` and `python -m churn_retention.demo`.
5. Open a PR explaining the problem, change, validation and remaining limitations.
6. Get a teammate's review before merging. Each contributor should understand their code.

A repository admin should configure main to require a PR, one approval and the
offline-checks status. This skeleton does not configure or enforce branch protection.

## Using Codex or Claude Code

Open your local clone in your editor. Point the assistant at AGENTS.md and one issue.
Ask it to explain the relevant code and propose a small implementation before editing.
Review its diff and test results, then ask it to walk through an example.
Do not ask two agents to edit the same working tree at once; use separate branches
and separate worktrees/clones if working concurrently.

Example task:
"Read AGENTS.md and docs/CONTRACTS.md. Implement a listening-days feature using only
events strictly before the prediction date. Add a test showing future events are
excluded. Explain the function with a tiny synthetic example."

AI can help write code; the human author owns the modelling assumptions and review.
Keep decisions in the repo so a new chat and every teammate have the same context.
