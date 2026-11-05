# Lab D vs. Lab B: Autoencoders on MNIST

**Lab D** = `lab6-2.pdf` (group11-section1)
**Lab B** = `06-autoencoders.pdf`

Both are submissions of the same assignment: build a 784→128→64→2→64→128→784 autoencoder in PyTorch on MNIST, train it, visualize the 2-D latent space, compare it with 2-component PCA, and write a short collaboration reflection (100 points total).

**Scope and limits.** I read both notebooks end to end. I could not check anything that cites the Michelucci (2022) paper, since the paper was not provided. I read the reconstruction images from the rendered PDFs, so my digit-by-digit descriptions are approximate.

---

## 1. Bottom line

- **Lab D is clearly the stronger submission.** It is complete, reproducible, and backs most of its claims with numbers (loss floor, per-class MSE, PCA reconstruction error).
- **Lab B is incomplete.** Names are blank, Section 6 is a placeholder, template text and `TODO` comments remain, the written answers are thin, and nothing is measured on the reporting set.
- **Lab B has two real strengths D lacks:** it ties the overlap in the latent plot to specific reconstruction errors, and it proposes a remedy (a larger bottleneck).
- **Lab D's weaknesses are small:** a few overstated claims, a reflection one sentence too long, and per-class MSE over-read as evidence of lost digit identity.

---

## 2. What the two labs share

| Area | Both labs |
|---|---|
| Model | Identical architecture; hidden blocks are Linear → BatchNorm1d → ReLU → Dropout(0.2); bottleneck is linear only; output is Linear → Sigmoid |
| Optimization | Adam, lr 1e-3, weight decay 1e-5, batch size 128, same EarlyStopping and training loop, up to 50 epochs |
| Data | Per-image min-max normalization; 70/15/15 split (49,000 / 10,500 / 10,500) |
| PCA | Fit on fitting + tuning data, applied to the reporting set (no leakage) |
| Conclusions | Reconstructions are blurry; digit identity is often lost (e.g., 4→9); the autoencoder's latent space shows more structure than PCA's; autoencoder axes have no fixed order |

The core code is essentially the same, so the difference between the labs is in **completeness, evidence, and rigor of the write-up**, not in the model.

---

## 3. Section-by-section comparison

| Section (points) | Lab D | Lab B | Edge |
|---|---|---|---|
| Header | Group name filled in | `NAME(s) = ""` left blank | D |
| 1. Theory (10) | Clear answers to all four questions; cites a k-NN speed/accuracy example; adds the linear-autoencoder-equals-PCA point | Very short. Q2 is incomplete; Q3's first bullet is unclear ("relates to reducing noise?"); Q4 says the AE "separates classes" although it is unsupervised; typo "adn"; stray `USE AS MANY MARKDOWN CELLS AS NEEDED` | D |
| 2. Data (10) | Seeds set, stratified split, zero-division guard, numeric `assert` (prints [0.0, 1.0]), explicit fixed-vs-fitted explanation, 10 distinct digits plotted | No seeds, no stratify, no zero guard, no assert; fixed-vs-fitted only in a code comment; 5 digits via `randint` (can repeat) | D |
| 3.1 Model (30) | Correct | Correct | Tie |
| 3.2 Training and loss reasoning (15) | BCE with a fuller justification; later computes the BCE floor from the data (18.1% non-binary pixels, floor 0.0581, reporting BCE 0.1866, excess 0.1285); explains why tuning loss is below fitting loss; GPU-aware | MSE with two brief sentences, typos, and "the overall shape of each pixel"; `TODO` comments left in code; CPU only | D |
| 3.3 Reconstruction (5) | One digit per class; overall and per-class MSE; explicit verdict ("only partly good… identity often lost") | First 8 digits of one batch; accurate, specific description of what went wrong; explicit verdict ("only partly good") | D (slightly) |
| 4. Latent space (15) | Plot plus analysis, plus a note that axes are unordered and non-orthogonal | Plot only; no analysis under it (discussion sits in 3.3) | D |
| 5. PCA (10) | Side-by-side plots, explained variance (16.89%), fit-set sensitivity check (16.87%), reconstruction MSE (AE 0.0418 vs. PCA 0.0558) | Single PCA plot; no variance, no reconstruction comparison; does note AE axes are unordered | D |
| 6. Reflection (5) | Written (but 6 sentences; limit is 3–5) | `YOUR TEXT HERE` | D |

---

## 4. Three themes worth a closer look

### 4.1 Loss function (BCE vs. MSE)

- **D chose BCE.** BCE with a sigmoid output gives the gradient `σ(z) − y`, so the sigmoid derivative cancels and learning doesn't stall when outputs saturate. D also shows that BCE has a non-zero floor on soft targets and that its loss value (~0.19) isn't "19% error."
- **B chose MSE.** It is valid and directly interpretable, but B's reasoning is brief and never compares it with BCE or connects it to the sigmoid.
- **Neither is wrong.** The skeleton hint visible in B ("reconstructing continuous pixel values in [0,1]") can be read either way, so check which loss your rubric expects.
- **Rough scale check:** D's BCE-trained model reaches a reporting-set MSE of 0.0418, while B's MSE-trained model ends near 0.0435 on the tuning set. The sets and splits differ, so this is only indicative, but the loss choice evidently didn't change reconstruction error much.

### 4.2 Quality of evidence

D's claims are mostly checkable from its own cells. B's are mostly assertions. B never evaluates the trained model on the reporting set numerically, never reports PCA's variance or reconstruction error, and the PCA-vs-AE comparison rests on eyeballing scatterplots. B's qualitative reading is, however, accurate to its own sample (3 sevens and a 2 decoded as 9s; the two 4s as a 9 and a 6).

### 4.3 AE vs. PCA

D quantifies the gap: AE reconstruction MSE 0.0418 vs. PCA 0.0558 on the reporting set. Since PCA's two components explain about 17% of the variance, those MSEs imply the autoencoder removes roughly 38% of the baseline variance versus about 17% for PCA (my back-of-envelope derivation, approximate). B gives only the qualitative contrast.

---

## 5. Which is more "right"?

**Lab D.** The model code is equivalent, so correctness of the implementation is a tie. D wins on everything surrounding it: completeness, reproducibility, evidence, and clarity of reasoning. B would lose most or all of the points tied to the missing Section 6 and to the thin or missing explanations in Sections 1, 3.2, 4, and 5.

---

## 6. Problems to fix in Lab D

1. **Section 6 has 6 sentences** (the lab asks for 3–5). Merge the two CPU/GPU sentences.
2. **"Entirely" is overclaimed (Section 1 Q4 and Section 5).**
   - Bourlard & Kamp's result is a real, well-known one, but it concerns a *linear* autoencoder with MSE: its optimum spans the same subspace as PCA (axes may differ by a rotation or mix).
   - D's autoencoder also differs from PCA in loss (BCE), depth, BatchNorm, and dropout, and the notebook has no linear-autoencoder control. "Stems entirely from non-linear activations" is plausible but unshown.
   - Fix: say "mainly," or run the control (same architecture, no ReLUs/BN/dropout/sigmoid, MSE loss; it should land near PCA's 0.0558).
3. **Per-class MSE is over-read as evidence of lost identity.** The text says per-class MSE "corroborates" severe failures for 4 and 7, yet digit 7 has the *second-lowest* MSE (0.0343) and digit 4 is mid-range (0.0447), even though D describes their reconstructions as a 9-like loop and an 8-like blob. MSE reflects pixel error (thin digits score low), not identity. To quantify identity loss, classify the reconstructions:
   ```python
   from sklearn.linear_model import LogisticRegression
   clf = LogisticRegression(max_iter=300).fit(x_fit, y_fit)
   model.eval()
   with torch.no_grad():
       recon = model(torch.from_numpy(x_report).to(dev)).cpu().numpy()
   print("accuracy on originals:       ", clf.score(x_report, y_report))
   print("accuracy on reconstructions: ", clf.score(recon, y_report))
   ```
4. **"…rather than a failure to reconstruct"** (end of the loss-floor discussion) conflicts with D's own verdict that identity is often lost. The excess BCE of 0.1285 is substantial. Say the floor explains *part* of the ~0.19.
5. **k-NN example wording and verification.** k-NN has no training step, so "training… took 1000 seconds" should say fitting/evaluating. The numbers (1000 s vs. 1.1 s, 96.4% → 89.0%, 909×, 7.4 percentage points) are arithmetically consistent, but I couldn't check them against the paper. Verify against Table 1.
6. **Make eval mode explicit.** The loss-floor and reconstruction cells call `model(...)` without `model.eval()`. They work only because training leaves the model in eval mode. Add `model.eval()` so BatchNorm and dropout can't silently run in training mode if cells are re-run out of order.
7. **Blank execution counters (`In [ ]`).** Outputs are present, but I can't tell they came from one clean run. Use "Restart & run all" before submitting.
8. **Latent plot and reconstruction errors aren't connected.** D notes that 4 and 9 crowd together in the latent space and separately that 4 reconstructs as a 9, but never links them. Add one sentence, and optionally B's remedy (a larger bottleneck, e.g., 8–16 dims).
9. **Single run.** All conclusions (MSE gap, cluster shapes) come from one seed. A sentence acknowledging this, or a second seed, would help.
10. **Minor:** the per-class loop re-converts `_recon_all.cpu().numpy()` ten times (compute once). Training ran the full 50 epochs with tuning loss still creeping down, so the epoch cap, not early stopping, ended training.

## 7. Problems to fix in Lab B

1. **Finish the submission:** fill in names and Section 6; delete `USE AS MANY MARKDOWN CELLS AS NEEDED`, `YOUR TEXT HERE`, and the `TODO` comments.
2. **Theory answers:**
   - Q2: the bottleneck forces compression and blocks an identity copy.
   - Q3: replace the unclear first bullet and fix "adn."
   - Q4: the AE captures nonlinear structure; it doesn't "separate classes" (it is unsupervised).
3. **Loss justification (3.2):** compare MSE with BCE, tie the choice to the sigmoid, and fix the typos ("penallizes," "devitions," "reconustrction") and the "shape of each pixel" sentence.
4. **Blur attribution (3.3):** "MSE also makes the outputs blurry" is not MSE-specific. With a 2-D bottleneck, any proper loss (BCE included) is minimized by the average of plausible digits. The bottleneck is the main cause.
5. **Add the missing evidence:** a reporting-set metric, PCA explained variance and reconstruction error, side-by-side plots, and a Section 4 analysis. Also explain why tuning loss sits below fitting loss (dropout is active only in training; fitting loss is averaged over the epoch while weights improve).
6. **Robustness:** set seeds; stratify the split; guard the min-max division (`torch.clamp(X_max - X_min, min=1e-8)`); add an `assert` on the normalized range; use a device variable and `.cpu()` in `extract_latent_features`.
7. **Presentation:** title the loss plot and start the epoch axis at 1.
