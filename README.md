## Hi, I'm Saksham 👋

I'm a final-year B.Tech CSE (AI/ML) student at UPES, Dehradun. I like taking an idea all the way to something people can use: training the model, writing the backend, wiring up the LLM, and shipping the app.

Most of what I build sits where machine learning meets ordinary software: FastAPI services, deep-learning models I design and benchmark myself, RAG pipelines that keep LLM answers grounded, and the frontends that make them usable.

🌐 [sakshamagrawal.me](https://sakshamagrawal.me) · 💼 [LinkedIn](https://www.linkedin.com/in/sakshamhere/) · 🧩 [LeetCode](https://leetcode.com/u/Sakshamm_codes/) (500+ solved) · 📄 [Résumé](https://drive.google.com/file/d/1hk_MAIPaLUcRsX-Tg9IukpXzCxVgiDun/view?usp=sharing) · ✉️ reach.saksham.agrawal@gmail.com

### What I've been up to

- **IBM, intern (Jun–Jul 2026):** trained a custom CNN in PyTorch for crop-disease classification that beat EfficientNet by about 10% macro-F1, and shipped it as a Streamlit app that gives treatment advice in Hindi and English.
- **Cloud Security Alliance, UPES:** President since May 2026, leading a 50-member chapter. Before that I was Registrations Head and ran onboarding for 400+ attendees at AWS Community Day Dehradun.
- **CSA Hackathon 3.0:** 5th place with PhoniX, a real-time detector for AI-generated voices on phone calls.
- **Research:** Kolmogorov-Arnold Networks (KANs) for session-based recommendation and medical imaging.

### Things I've built

| Project | What it is | Stack |
|---|---|---|
| **JobGenie** · [live](https://jobgenie.tech) | AI job-search copilot: one search across job boards, résumé parsing and health scores, ATS match, skill-gap analysis, and an application tracker with reminders | FastAPI, React, Gemini, Qdrant, Supabase, Celery |
| [DealerOS](https://github.com/devSaksham/DealerOS) | Dashboard for a Honda dealership. Forecasts spare-parts demand (TSB + festive seasonality), builds the weekly order list, and logs lost sales by voice in Hindi/Hinglish | Python, Streamlit, pandas |
| [billing-core](https://github.com/devSaksham/billing-core) | Invoice API that stays correct under retries: idempotency keys, an append-only ledger, and concurrency tests with real threads | FastAPI, SQLite, pytest |
| [causalfunnel-analytics](https://github.com/devSaksham/causalfunnel-analytics) | Session tracking and click heatmaps: an embeddable tracker, an Express API and a React dashboard | Node, Express, MongoDB, React |
| [crop-disease-detection](https://github.com/devSaksham/crop-disease-detection) | Leaf-photo diagnosis across 38 classes with severity scoring, weather-aware LLM treatment plans and Hindi text-to-speech | PyTorch, Streamlit, Groq |
| [LeafLens](https://github.com/devSaksham/LeafLens) | The mobile version: on-device TFLite diagnosis, treatment plans and a Gemini assistant, in English and Hindi | Flutter, TFLite, Supabase |
| [session-based-recommendation](https://github.com/devSaksham/session-based-recommendation-system-latest) | GRU vs KAN-GRU for next-item recommendation on YooChoose, with KAN layers written from scratch | PyTorch |
| [PhoniX](https://github.com/devSaksham/PhoniX) | Hackathon team project: flags deepfake voices during live calls | Flutter, TensorFlow, Firebase |
| [hogwarts-portfolio](https://github.com/devSaksham/hogwarts-portfolio) | My portfolio as a walk through Hogwarts, with a command-line "Spell Class" mini-game | HTML, CSS, JS |
| [Launch-Cadence](https://github.com/devSaksham/Launch-Cadence) | Terminal app for booking rocket launch slots, with admin and customer roles | Python |

### Tools I use

- **Languages:** Python, Java, SQL, C, TypeScript/JavaScript, Dart
- **ML / AI:** PyTorch, TensorFlow, scikit-learn, OpenCV, CNNs, GRU/LSTM, KANs, Gemini API, RAG, vector search
- **Backend & data:** FastAPI, REST, PostgreSQL, MySQL, MongoDB, Redis, Qdrant, Supabase, Celery
- **Shipping:** Docker, Git, GitHub Actions, Streamlit, Netlify, Flutter

### Notebooks in this repo

KAN experiments that don't have their own repo yet:

- `KAN_RNN_Session_Based_Recommendation_YOOCHOOSE.ipynb`: KAN-RNN vs RNN/GRU for session-based recommendation
- `kan_transformer_chest_xray.ipynb`, `kan_xray_pneumonia.ipynb`: KAN-Transformer hybrids for pneumonia detection on chest X-rays
- `kan_transformer_mnist.py`: KAN-Transformer hybrid on MNIST

---

Open to full-time roles and freelance work. If you're building something interesting, say hi.
