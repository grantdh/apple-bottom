# Measurement results

Each folder holds one measurement campaign, named `<date>-<commit>-<topic>`:

- `<date>`: the UTC date of the run;
- `<commit>`: the short hash of the library commit that was measured;
- `<topic>`: what was measured. The two earliest folders, from April 2026, have no topic.

[docs/REPRODUCIBILITY.md](../../docs/REPRODUCIBILITY.md) lists which numbers in the README and in VAL-1 come from which
folder, with the command that regenerates them. The folders it cites name the machine, OS, toolchain and library build
of their runs.
