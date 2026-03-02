# Evaluation Sweep: Baseline (MFCC) vs Contrastive (CNN)

**Setup:** 5 references (blues.00000–00004), 10 transformed queries each = 50 total queries  
**Transforms:** crop (1s, 2s), noise (snr10, snr20), orig, pitch (±2, ±4 semitones), tempo (0.85×)

---

## Hit Rates (Top-1 / Top-k)

| Metric | Baseline | Contrastive |
|--------|----------|-------------|
| **Overall** | 50/50 (100%) | 50/50 (100%) |
| **By transform** | All 100% | All 100% |

Both models achieve **perfect retrieval** on all 50 queries across all transform types.

---

## Average Match Scores (top_score) by Transform

| Transform | Baseline (avg) | Contrastive (avg) | Δ (Contrastive − Baseline) |
|-----------|----------------|-------------------|----------------------------|
| **crop** | 0.991 | 0.982 | −0.009 |
| **noise** | 0.929 | **0.962** | **+0.033** |
| **orig** | 0.993 | 0.992 | −0.001 |
| **pitch** | 0.967 | 0.977 | **+0.010** |
| **tempo** | 0.989 | 0.990 | +0.001 |

---

## Key Findings

1. **Noise robustness:** Contrastive model scores **+3.3 pp** higher on average for noise transforms (0.962 vs 0.929). This aligns with training on noise-augmented pairs.

2. **Pitch robustness:** Contrastive scores **+1.0 pp** higher on pitch-shifted queries (0.977 vs 0.967).

3. **Crop:** Baseline scores slightly higher on crop (0.991 vs 0.982), likely because MFCCs are less sensitive to truncation.

4. **Overall:** Both models perform well; contrastive shows stronger robustness to noise and pitch, which are common real-world degradations.

---

## Output Files

- **Baseline:** `eval_sweep_baseline.csv`, `eval_sweep_summary_baseline.csv`, `eval_sweep_report_baseline.md`
- **Contrastive:** `eval_sweep_contrastive.csv`, `eval_sweep_summary_contrastive.csv`, `eval_sweep_report_contrastive.md`
