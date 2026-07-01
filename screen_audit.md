# Screening audit — top Materials Project Li–S hits

After applying the OBELiX CatBoost model to Materials Project (`04_screen_mp.py`),
the **top 15 hits of an initial, deliberately loose run** (filter: contains Li+S,
`e_above_hull ≤ 0.05`, `band_gap ≥ 0.5`) were each **adversarially audited** by an
independent check against the literature + physical plausibility. The goal was
not to trust the ranking but to *break* it — find which top hits are real, which
are artifacts, and turn that into concrete upstream filters.

**A ranking is only as good as what you let into it.** The audit found the top of
the list was genuinely right (LGPS family) but the tail was polluted by three
failure modes a naive Li+S query cannot see.

## Verdicts (15 hits, each audited independently)

| # | material_id | formula (true identity) | model σ (S/cm) | verdict | reality | filtered now? |
|---|---|---|---|---|---|---|
| 1 | mp-696128 | Li₁₀GeP₂S₁₂ (**LGPS**) | 6.7e-3 | ✅ true positive | superionic ~1.2e-2 (Kanno 2011) | kept |
| 2 | mp-696129 | Li₁₀SiP₂S₁₂ | 3.2e-3 | ✅ true positive | superionic ~2.3e-3 | kept |
| 3 | mp-696123 | Li₁₀SnP₂S₁₂ | 2.1e-3 | ✅ true positive | superionic ~4e-3 (Bron 2013) | kept |
| 4 | mp-696138 | Li₁₀GeP₂S₁₂ (P1 cell) | 1.8e-5 | ✅ true positive* | superionic ~1e-2 | kept |
| 5 | mp-720509 | Li₁₀SiP₂S₁₂ (P1 cell) | 1.4e-5 | ✅ true positive* | superionic ~2e-3 | kept |
| 6 | mp-1040451 | Li₂₀Si₃P₃S₂₃Cl | 2.8e-5 | 🔬 **novel candidate** | Ge-free LGPS/argyrodite hybrid, no data | **kept** |
| 7 | mp-753546 | Li₈TiS₆ | 1.7e-5 | 🔬 **novel candidate** | Li-rich antifluorite (Ti⁴⁺ d⁰), no data | **kept** |
| 8 | mp-756490 | Li₆MnS₄ | 1.6e-5 | 🔬 novel (⚠ Mn redox) | antifluorite, electronic-leakage risk | dropped (redox-TM) |
| 9 | mp-1153 | Li₂S | 3.1e-5 | ❌ false positive | ~1e-13, antifluorite has no Li path | dropped (binary) |
| 10 | mp-769032 | Li₃NbS₄ | 2.2e-5 | ❌ false positive | disordered-rocksalt **cathode** (~386 mAh/g) | dropped (gap<1.5) |
| 11 | mp-753974 | Li₈CrS₆ | 1.9e-5 | ❌ false positive | narrow-gap (0.77 eV) MIEC | dropped (redox-TM) |
| 12 | mp-766506 | Li₃CuS₂ | 1.1e-4 | ❌ false positive | Cu-redox MIEC **cathode** (Doi 2021) | dropped (redox-TM) |
| 13 | mp-777963 | Li₃SbS₄·9H₂O | 3.3e-5 | 🚫 artifact | Schlippe-salt **nonahydrate** | dropped (exclude H) |
| 14 | mp-740717 | Li(NH₄)SO₄ | 1.3e-5 | 🚫 artifact | ammonium **sulfate** (S is SO₄, not S²⁻) | dropped (exclude H) |
| 15 | mp-760046 | LiPH₂₁S₃N₇ | 5.5e-6 | 🚫 artifact | ammine/ammonium **molecular salt** | dropped (exclude H) |

\* identity correct (it *is* LGPS) but σ under-predicted by ~2–3 orders — a useful
calibration-failure case for the regressor on the distorted P1 DFT cells.

## The three failure modes — and the fix

1. **Hydrate / oxysalt / molecular artifacts** (#13–15). A Li+S filter catches the
   *sulfur of a sulfate* (Li(NH₄)SO₄) or *water of crystallisation* (Li₃SbS₄·9H₂O) —
   chemically irrelevant to sulfide electrolytes. All three are H-rich.
   → **`exclude_elements=["H"]`** removes every one.
2. **Mixed ionic–electronic conductors / cathodes** (#8, #10–12). Real chalcogenide
   crystals, but redox-active open-shell TMs (Cr, Mn, Cu) or narrow gaps make them
   *electrodes*, not electrolytes — an electrolyte must be a redox-inert electronic
   insulator. The model only sees Li mobility, not electronic leakage.
   → **drop redox-active TM sulfides** {V,Cr,Mn,Fe,Co,Ni,Cu,Mo,W} + **`band_gap ≥ 1.5`**.
3. **Antifluorite binary precursors** (#9, Li₂S). A real, stable sulfide but no Li
   transport pathway (~1e-13 S/cm); it is a *precursor*, not an electrolyte.
   → **`num_elements ≥ 3`** drops binaries.

These are now baked into `run_live()` in `04_screen_mp.py`. The re-run dropped the
pool from 328 → 184 clean candidates (64 redox-TM sulfides removed), and the top of
the list is now LGPS-family true positives followed by two clean novel leads.

## Leads for Project 2 (MLIP-MD)

After filtering, the two genuine, literature-blank candidates worth quantitative MD:
- **Li₂₀Si₃P₃S₂₃Cl** (mp-1040451) — a Cl-substituted Ge-free LGPS/argyrodite hybrid.
- **Li₈TiS₆** (mp-753546) — a Li-rich antifluorite; the closest analog (Li₈GeS₆)
  suggests *moderate*, not superionic — temper expectations and let MD decide.

## Honest takeaways

- The screen's **strongest signal is real**: a blind MP query put the LGPS family at
  ranks 1–3, matching the SHAP family ranking — the model learned chemistry.
- The screen's **weakness is also real and now documented**: composition+structure
  features are blind to electronic conductivity and electrochemical role, so a pre-ML
  filter on chemistry/gap is mandatory, not optional.
- This is exactly the "honest diagnosis" stance of the portfolio: the funnel *ranks*
  candidates; it does not certify them. The head of the list goes to MD, not to a press release.

*Method: each of the 15 top hits was audited one at a time — given its Materials Project
metadata and the model's prediction, then challenged to refute that prediction against the
materials literature and physical plausibility (redox activity, band gap, known conductivity,
structural role).*
