# Diffusions-LLM-DLLM-

Interactive **text diffusion** demo and notes on how diffusion models work—in **images, audio, video, molecules**, and **language**—and how **diffusion LLMs (dLLMs)** differ from ordinary left-to-right (autoregressive) large language models.

---

## What is a diffusion model?

A **diffusion model** is a generative model that learns to **undo a gradual corruption process**. You define a **forward** process that destroys structure in data (by adding noise or, for discrete data, masking), and train a network to run the **reverse** process: starting from “noise,” it refines samples step by step into realistic outputs.

That framing—**learn to reverse destruction**—appears across modalities. Surveys and tutorials often describe the image case first (Gaussian noise on pixels), then generalize ideas to other domains. For a careful walkthrough of the forward/reverse picture and training objectives, see tutorials such as [*Step-by-Step Diffusion: An Elementary Tutorial*](https://arxiv.org/html/2406.08929v2) and handbooks like [*Generative Diffusion Modeling: A Practical Handbook*](https://arxiv.org/abs/2412.17162).

---

## Continuous diffusion (typical for images)

**Forward process:** Starting from a real image \(x_0\), noise is added over timesteps until the signal looks like **pure Gaussian noise** \(x_T\). This Markov chain is fixed (no learning); it only needs a **noise schedule**.

**Reverse process:** A neural network (often a **U-Net** in classic image setups) learns **denoising**: at each step it predicts how to move from \(x_t\) toward a slightly cleaner \(x_{t-1}\). Generation = sample noise, then iterate denoising until an image appears.

**Training intuition:** The model is trained so its predictions match the noise (or score) implied by the forward process—related to **denoising score matching** and variational objectives. A landmark formulation is **DDPM** (*Denoising Diffusion Probabilistic Models*, [Ho et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)).

**Why it works well:** The network only has to make **small local improvements** each step instead of inventing a whole image in one shot, which can be an easier optimization problem than single-step generation.

---

## Text and discrete data: masked / discrete diffusion

Text is **discrete** (tokens), so you cannot add Gaussian noise to token IDs in the same way as pixels. Common approaches include:

- **Masking:** Replace tokens with a special mask token; the model learns to **predict masked positions** given context.
- **Iterative refinement:** Start from (almost) all masks and **unmask or resample** over steps—**coarse-to-fine** text generation.
- **Parallel prediction:** At each step, the model may **score many positions at once**, unlike strict left-to-right autoregressive decoding.

This is the family of ideas behind **diffusion / masked diffusion language models** discussed in research (e.g. **LLaDA**—Large Language Diffusion with masking—and commercial **Mercury**-style dLLMs). Research demos and writeups: [LLaDA project page](https://ml-gsai.github.io/LLaDA-demo/), paper [*Large Language Diffusion Models*](https://arxiv.org/abs/2502.09992) (mask predictor, bidirectional context, no causal attention at train time). Scaling lines include follow-ups such as [LLaDA 2.0](https://arxiv.org/abs/2512.15745). For a diffusion LLM positioned for speed and parallel generation, see [Mercury (Inception Labs)](https://www.inceptionlabs.ai/introducing-mercury) and the technical writeup [*Mercury: Ultra-Fast Language Models Based on Diffusion*](https://arxiv.org/html/2506.17298v1).

### Autoregressive LLM vs diffusion-style text model (conceptual)

| Aspect | Autoregressive (e.g. GPT-style) | Diffusion / masked text (dLLM-style) |
|--------|----------------------------------|--------------------------------------|
| Generation order | Usually **strictly sequential** (token 1, then 2, …) | Often **many positions updated in parallel** over steps |
| Starting point | Empty context, grow prefix | Can start from **full noise / full mask** and **denoise** |
| Editing / refinement | Most often regenerate suffix | Natural metaphor for **iterative correction** (research-dependent) |
| Throughput | One new token per forward pass (without speculative tricks) | Parallel steps can yield **very high tokens/sec** in some systems (vendor/research claims vary by hardware and model) |

Exact training recipes and inference algorithms differ by paper and product; the table captures the **design axis** most people mean when they say “diffusion LLM.”

---

## Applications (beyond “pretty pictures”)

Diffusion-style generative models are used broadly:

- **Images & video:** Synthesis, editing, **inpainting**, super-resolution, deblurring, enhancement, video-to-video and temporally consistent generation (see e.g. surveys and application papers in computer vision venues).
- **Audio:** Speech and music generation, enhancement, and similar denoising-from-noise stories in the audio domain.
- **Science & engineering:** **Molecule and protein** generation, medical and microscopy imaging, remote sensing, and other domains where high-dimensional structured data benefits from iterative refinement (see survey literature on diffusion for scientific applications).

The pattern is the same: define corruption → learn reversal → sample or condition on inputs for a task.

---

## This repository

A small **educational React (Vite)** demo lives in `src/TextDiffusionDemo.jsx`. It is **not** a neural network; it uses a **topic-conditioned toy “vocabulary”** to simulate:

1. Starting from **all `[MASK]` tokens** (pure noise in this discrete story).
2. Each step: **predict every masked position**, assign a **confidence**, **sort**, and **unmask the most confident** tokens first.
3. Repeat until the sequence is fully unmasked.

That illustrates the **non-left-to-right** flavor of masked diffusion text generation. For real dLLMs, a **transformer** replaces the toy predictor and is trained on massive text (see references above).

### Run locally

```bash
npm install
npm run dev
```

### Build for GitHub Pages (project URL under `/<repo-name>/`)

```bash
npm run build:pages
```

Upload the contents of `dist/` to your hosting branch or static host. Adjust `BASE_PATH` in `package.json` → `build:pages` if you rename the repository.

---

## Further reading

- **Tutorial:** [Step-by-Step Diffusion: An Elementary Tutorial](https://arxiv.org/html/2406.08929v2)  
- **Handbook:** [Generative Diffusion Modeling: A Practical Handbook](https://arxiv.org/abs/2412.17162)  
- **Classic image DDPM:** [Ho et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)  
- **Text diffusion (research):** [Large Language Diffusion Models (LLaDA)](https://arxiv.org/abs/2502.09992) · [LLaDA demo](https://ml-gsai.github.io/LLaDA-demo/) · [LLaDA 2.0 (scaling)](https://arxiv.org/abs/2512.15745)  
- **Diffusion LLM (product + paper):** [Mercury announcement](https://www.inceptionlabs.ai/introducing-mercury) · [Mercury technical report](https://arxiv.org/html/2506.17298v1)  

---

## Disclaimer

This repo’s UI is for **intuition only**. Production systems use learned distributions, careful schedules, and evaluation harnesses; benchmarks and speeds depend on hardware and implementation. Check primary sources for claims about accuracy and throughput.
