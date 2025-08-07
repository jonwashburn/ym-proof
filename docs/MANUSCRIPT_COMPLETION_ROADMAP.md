Manuscript Completion Roadmap
=================================

Scope
-
This roadmap focuses on completing the written manuscript first, using `Yang-Mills-July-7.txt` as the current draft and `docs/YANG_MILLS_MANUSCRIPT.tex` as the archival/canonical location. No code builds are required for this plan.

Outcomes
-
- A single, polished LaTeX manuscript in `docs/YANG_MILLS_MANUSCRIPT.tex` ready for public release and submission.
- Consistent notation, numerics, and claims across sections.
- Clear linkage between conceptual statements and formalization status (without requiring builds).

Milestones (author-facing)
-
M0. Canonicalize the source
- [ ] Choose canonical file: copy the contents of `Yang-Mills-July-7.txt` over `docs/YANG_MILLS_MANUSCRIPT.tex` (or vice‑versa) and remove duplication.
- [ ] Normalize preamble: packages, macros (`\phi` vs `\varphi`, math fonts, theorem styles).
- [ ] Introduce shared macros for constants (`\varphi`, `E_{\text{coh}}`, `\Delta`), operators, and common environments.

M1. Claims alignment and framing
- [ ] Decide framing: “complete proof” vs “program with partial formalization.”
  - Option A (conservative): soften absolute claims (zero axioms/zero sorries) and clearly label the formalization plan.
  - Option B (assertive): keep strong claims and add a precise “Verification Checklist” section enumerating what must be finished in code; mark them as “to be released.”
- [ ] Update Abstract and Theorem statements to match chosen framing.

M2. Notation and constants consistency
- [ ] Use `\varphi` uniformly; avoid ad‑hoc `phi` numerics in prose.
- [ ] Fix a single definition: \( \massGap = E_{\text{coh}}\,\varphi \).
- [ ] Provide one authoritative list of parameters and units (move to a dedicated “Parameters” subsection; cross‑refer throughout).
- [ ] Ensure all numerical values (e.g., 1.775–1.78 GeV) derive from the same definitions and approximations.

M3. Section by section content work
- Introduction
  - [ ] Clarify motivation, contrast to standard approaches (path integrals/renormalization), and contributions.
  - [ ] Remove/condense marketing language; emphasize mathematical structure and verifiability plan.
- Recognition Science framework
  - [ ] Present the eight foundations succinctly with non‑circular definitions.
  - [ ] Move long philosophical exposition to an appendix.
- Gauge embedding and lattice theory
  - [ ] Replace informal metaphors with precise mathematical mappings (objects, morphisms, invariants).
  - [ ] Give the exact lattice objects and Wilson loop definitions used later.
- OS reconstruction and reflection positivity
  - [ ] State hypotheses precisely; isolate what is proven vs assumed.
  - [ ] Provide the structure of the reconstruction without code references.
- Wilson correspondence / Continuum limit
  - [ ] Clarify limits, topologies, and error terms. Include a statement of what “converges” means.
- BRST/cohomology
  - [ ] Define complexes, degrees, and the physical space quotient precisely.
- Main results
  - [ ] Collect theorems in one place with consistent numbering and hypotheses.
  - [ ] If keeping assertive framing, add an explicit “Verification Checklist” linking each theorem to evidence (see M5).
- Numerical section
  - [ ] Present a single computation pipeline from constants to \(\Delta\) with error bars and unit consistency.
  - [ ] Remove duplicate or inconsistent numbers.
- Related work / comparison
  - [ ] Positioning with lattice QCD and rigorous RG; cite accurately.
- Conclusion
  - [ ] State limitations and near‑term verification tasks honestly (one short paragraph, not a list).

M4. Figures, tables, and reproducibility
- [ ] Include 1–2 figures: (i) \(\varphi\)-cascade energy ladder; (ii) lattice-to-continuum mapping schematic.
- [ ] Add a parameters table (symbols, definition, numeric value, units, provenance).
- [ ] Reference `docs/REPRODUCIBILITY_GUIDE.md` for environment + artifact layout; summarize in one paragraph in the manuscript.

M5. Claim–evidence concordance (documentation only; no builds)
- [ ] Add an appendix “Formalization Status & Evidence” with a two‑column checklist:
  - Column 1: Manuscript claims (theorems/lemmas).
  - Column 2: Evidence type (conceptual proof sketch, prior literature, or planned Lean module) and file path(s) to source code where applicable.
- [ ] For statements that are not yet fully formalized in Lean, include a forward reference to the planned Lean modules rather than asserting completion.

M6. References and style
- [ ] Normalize bibliography entries and years; check arXiv versions.
- [ ] Enforce consistent theorem/lemma environments; ensure all definitions are first usage bold/italics per journal style.

M7. Final editorial pass
- [ ] Copy edit for clarity and concision; remove redundancies.
- [ ] Run LaTeX compilation locally (optional) and fix warnings (undefined refs, overfull boxes).
- [ ] Freeze the PDF in `ym-proof/Yang_Mills_Lean.pdf` and tag the commit.

Risk/decision log (keep short)
-
- R1: Strength of claims vs present formalization. Decision: ______________________ (A/B).
- R2: Numerical value consistency (1.78 GeV, constants). Owner to confirm: ________.
- R3: Scope of OS/BRST sections: full statements vs high‑level summary. Decision: ___.

Proposed timeline (adjust as needed)
-
Week 1
- M0 Canonicalize source; M1 framing; M2 notation/parameters.

Week 2
- M3 rewrite of OS, Wilson correspondence, and BRST sections; M4 figures/tables.

Week 3
- M5 concordance appendix; references/style normalization; M7 editorial pass; PDF freeze.

Acceptance criteria
-
- Single canonical manuscript builds cleanly with no TODOs/Placeholders.
- Abstract and theorems are consistent with chosen framing (A or B).
- All symbols and numbers are consistent across the manuscript.
- Concordance appendix lists each major claim with its evidence path.

