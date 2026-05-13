# Project: Multilingual Keyword Spotting with TinyML Deployment

## Dataset
Multilingual Spoken Words Corpus (MSWC) — 23.4M one-second audio clips, 340,000 keywords, 50 languages, CC-BY 4.0. Chosen over Google Speech Commands V2 for novelty and scale.

## Architecture
Two-stage pipeline:
- **Language Identification head**
- **Keyword Detection head** — conditioned on language prediction

Both stages share a common backbone (multi-task learning), though the exact backbone architecture is intentionally left open. The shared representation is what enables graceful degradation: acoustically similar keywords across languages (e.g. Turkish "merhaba" / Arabic "marhaba") cluster in the same embedding space, so misrouted audio still tends to produce correct keyword predictions. This will be quantified experimentally by deliberately misrouting test samples and measuring accuracy drop.

## Training
Quantization-Aware Training (QAT) in PyTorch (torch.ao.quantization). Post-training the model will be exported via ONNX → TFLite for embedded deployment.

## Deployment Target
ESP32. Audio capture via I2S microphone (e.g. INMP441) — the built-in ADC is too noisy for reliable audio at 16kHz. Deployment will be simulated using QEMU-based ESP32 environment, with optional validation on a physical board.

## Key Design Rationale
(For report motivation section) Current production wake word systems (e.g. Apple's "Hey Siri") use fixed DSP pipelines hardcoded to a single phrase per language. Commercial multilingual support exists but is handled via separate per-language models, not a unified system. A shared multilingual model is more scalable and practical for resource-constrained devices where storing multiple separate models is infeasible.

## Implementation
PyTorch for all training and QAT. Colab or equivalent for experimentation. Report must include dataset justification, modeling procedure, implementation plan, annotated code, and a reflection section including prompts used (AI assistance is explicitly expected by the professor).

## Scope Decision Pending
Which languages and how many keywords to include — recommended to pick 3-5 languages including English and potentially Turkish given your location.