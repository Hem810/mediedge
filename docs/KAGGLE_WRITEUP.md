# MediEdge , Bringing Frontier AI to India's Last Health Mile

**Track:** Health & Sciences · Ollama Special Tech
**Team:** MediEdge
**Code:** [your-github-url]
**Video:** [https://youtu.be/BhsvpntCbnE]

---

## The 10:30 PM Problem

It is 10:30 PM in Bissau, a village in Jhunjhunu district, Rajasthan. A one year old boy named Rahul has had a fever for three days. His mother walks 400 metres to the small concrete house where Sushma , the village's ASHA (Accredited Social Health Activist) worker, lives.

Sushma has a Class 12 education, three months of community health training, and a paper booklet from 2018. She has no doctor on call. The nearest Community Health Centre is 14.5 km away. The cell tower three villages over has been out since the morning rain.

She has to decide, by herself, in the next thirty seconds, whether Rahul needs to travel that 14.5 km tonight.

This decision plays out **roughly 13 million times every week** across rural India. Get it wrong by under referring and a child dies of pneumonia. Get it wrong by over referring and a daily wage family loses an entire week's income.

**Until now.**

---

## What MediEdge Does

MediEdge is a web based clinical decision support assistant powered by Gemma 4. The ASHA worker opens it on any browser on any laptop, and walks through a four step assessment:

1. **Patient details**- name, age in months (critical for IMCI age banding), sex, village.
2. **Voice symptom recording**- she taps a microphone and describes the patient's symptoms in Hindi. Whisper transcribes. The English translation is generated automatically for the model.
3. **Vitals + optional clinical photo**- temperature, heart rate, respiratory rate, SpO₂. Abnormal values are flagged in red in real time. She can photograph a wound or a medication packet.
4. **Gemma 4 analyses**- the system retrieves the relevant WHO IMCI protocols for the patient's age band via BM25 retrieval, builds a clinical prompt, and asks Gemma 4 to produce a structured JSON assessment.

In 60–70 seconds, Sushma sees:

- A ranked **differential diagnosis** with confidence scores, ICD-10 codes, supporting and against findings, and the corresponding WHO IMCI category.
- A list of **medications** drawn exclusively from the NHM India Essential Medicines list, with weight based paediatric dosing in both English and Hindi.
- A **referral decision** with the nearest facility, phone number, urgency level, and a one line reason in Hindi.
- A **plain language Hindi summary** that she can read aloud to Rahul's mother , three sentences in Devanagari script that explain what is wrong, what to do tonight, and when to come back.

She makes the call. Rahul either gets the right amoxicillin tonight, or he gets in the 108 ambulance.

---

## Architecture

The system has three architectural layers, each chosen for the deployment context.

**The retrieval layer** uses BM25 over a SQLite backed knowledge base. This was a deliberate choice over vector embeddings: the WHO IMCI corpus is small (a dozen protocols), age banding matters more than semantic similarity (a 7 month old's diarrhoea is a categorically different decision tree from a 4 year old's), and BM25 needs zero ML infrastructure. The retrieval is age filtered before scoring, then top 5 results are passed to Gemma as authoritative context.

**The reasoning layer** is Gemma 4 running locally via Ollama. The choice of local over cloud is not aesthetic , it is structural. The user has unreliable internet, the data is very sensitive, and the per query economics of any cloud API do not scale to 13 million weekly consultations. The prompt is structured: a tightly scoped system message that constrains the model to NHM formulary medicines and WHO IMCI protocols, the patient context, the retrieved protocols, and a strict JSON schema. The response is generated with Ollama's `format: "json"` flag and `temperature: 0.1` to maximise output reliability.

**The presentation layer** is a single page app served by FastAPI. The visual design is intentional: forest greens and warm cream tones, DM Serif Display headings for clinical gravitas, Noto Sans Devanagari for genuine Hindi rendering, and a four step linear flow that mirrors how a non technical health worker actually thinks through a patient encounter. Urgency is communicated through colour at every level , the differential cards, the referral box, the dashboard badges , so a quick glance conveys clinical priority before the worker reads a single word.

The complete state of every assessment, voice transcript, vitals, image path, all model outputs, inference latency , is persisted to SQLite. This serves three purposes: a clinical audit trail, a historical record per patient (every visit feeds context for the next), and the offline first foundation that allows the same architecture to work without connectivity once Gemma is moved on device.

---

## Why Gemma 4 Specifically , Running Entirely On-Device

Three properties of Gemma are doing real work here, and the deployment is the story.

**Local first via Ollama.** MediEdge does not call any external API. There is no cloud LLM, no API key, no network dependency for inference. We use [Ollama](https://ollama.com) to run Gemma 4 entirely on the worker's machine. This is not a graceful fallback , it is the architecture. The `services/gemma_service.py` module talks to a local Ollama HTTP server on `localhost:11434`, and the only thing the user installs is a single binary plus the model weights , `gemma4:e4b`. After that, the entire system runs offline forever.

This matters because the alternative , calling a cloud LLM , is structurally incompatible with the user. A village in Jhunjhunu district can lose connectivity for days. ASHA workers do not have data plans that survive a busy clinic afternoon. And the per query economics of cloud APIs do not scale to 13 million weekly consultations across 900,000 workers , even at $0.001 per call, that is over $13,000 per week with no funding source and then there is the issue of privacy of the patient data.

**Multimodal input.** When the worker photographs a wound, the model sees both the image and the symptom narrative simultaneously. Gemma 4 supports multimodal inputs through Ollama natively , we send the base64-encoded image alongside the text prompt and the model reasons over both. A wound described as "small cut, healing well" plus a photograph showing pus and red streaking produces a different assessment than either input alone.

**JSON mode for reliable outputs.** Ollama exposes a `format: "json"` parameter that constrains Gemma to emit valid JSON, which lets us define a strict output schema and trust the model to fill it. When the schema is enforced, our parsing layer never has to deal with markdown fences, hallucinated commentary, or partial responses. The schema is the contract, and Gemma honours it.

**Open weights.** This is the deployment story. The model file lives on the user's disk. They own it. There is no vendor that can deprecate it, no terms of service change that can disrupt the village clinic, no rate limit that can fail at the worst moment. Common Service Centres , small Linux servers already deployed in 250,000 panchayats , can host a single Ollama instance that serves an entire taluka of ASHA workers, with zero per query API cost, zero cloud dependency, and full data sovereignty for sensitive health records.

---

## Engineering Decisions

We made several non obvious choices that materially improve the product.

**Age in months, not years.** WHO IMCI protocols have hard age boundaries (a 7 month old vs an 8 month old can fall into different decision trees for fast breathing thresholds). Storing age in months and passing exact months to the model prevents the kind of off by one error that would invalidate a referral.

**Whisper at "small" size via faster-whisper.** The Whisper small model gives accurate Hindi transcription on CPU in 5–10 seconds per recording. The larger models add latency without meaningfully improving Hindi accuracy at this domain. We use [faster-whisper](https://github.com/SYSTRAN/faster-whisper) , a CTranslate2 reimplementation , instead of the original OpenAI library, which makes inference roughly 4× faster on CPU and removes the PyTorch dependency. The first call downloads the model (~500 MB) and caches it locally; every subsequent call reuses the in memory model, and the cache persists offline.

**Hindi summary always shown, English summary collapsible.** The primary user is a Hindi speaker. Forcing them to scroll past English to find their language is a small daily indignity that adds up. The English (and the SOAP note) are available but not in the way.

**Confidence visualised as ring gauges, not numbers.** A 67% confidence written as "67%" looks like a weather forecast. As a partial ring around the diagnosis name, it communicates immediately and pre verbally that this is probable but not certain.

**Drugs tagged "NHM" inline.** The National Health Mission essential medicines are what's actually in Sushma's drug box. The tag tells her at a glance which of Gemma's recommendations are things she physically has tonight versus things that require a referral.

---

## What This Is Really About

There is a tendency in AI to chase the frontier, bigger models, harder benchmarks, more compute. MediEdge points the other direction. It takes Gemma 4 and asks: who needs this most, and what's stopping them from having it?

What's stopping them is that frontier AI is built for English speakers with $1,000 phones and gigabit fibre. It is not built for Sushma. But Gemma 4 is open. The Hindi tokens are in the vocabulary. The IMCI protocols are public domain. The API is free at the tier we need. Every barrier is removable.

What MediEdge demonstrates is that the work to bring frontier intelligence to the last health mile is not a research problem. It is a deployment problem, and a UX problem, and a respect problem. The tools exist. We just have to ship them.

---

