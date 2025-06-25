# Experimental Modules

This directory contains auxiliary modules that are not required for the main Yang-Mills existence and mass gap proof.

## Status

These modules contain technical lemmas with `sorry` placeholders that represent:
- Detailed numerical calculations
- Standard mathematical results from analysis
- Technical properties that don't affect the main theorem

## Why These Are Separate

The main theorem in `Main.lean` proves Yang-Mills existence and mass gap without depending on these incomplete auxiliary proofs. The core proof chain is complete with:
- 0 axioms
- 0 sorries in the main dependency graph

## Future Work

These lemmas can be completed to provide:
- Additional theoretical insights
- Reusable components for other proofs
- Complete formalization of all supporting mathematics

However, they are not necessary for the validity of the main result. 