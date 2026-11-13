# Plant pest & disease diagnosis agent

One vision model with tools, over OpenRouter. Given a photo, the agent searches an insect field guide, lists the
conditions known for the host plant, compares the photo side by side with labelled reference photos, asks the user
when an answer would change the diagnosis, and returns a structured assessment (cause, confidence, differentials,
what to check next, what to do). It may answer "healthy" or "uncertain".

## Files

| File | What it is |
|---|---|
| `plant_agent.py` | Knowledge layer, tools, agent loop, CLI |
| `eval_agent.py` | Evaluation against labelled images, next to a no-tools baseline |
| `streamlit_app.py` | Persian chat UI |

## Setup

```
pip install -r requirements.txt
```

Put the key in a `.env` file next to the scripts (it is git-ignored):

```
OPENROUTER_API_KEY=...
```

The data is not in this repository. Paths are set at the top of `plant_agent.py` and can be overridden with
environment variables:

| Variable | Content |
|---|---|
| `PLANT_CHUNKS_DIR` | `upgraded_chunks_*.json` of the insect field guide |
| `PLANT_BOOK_FILES` | folder holding `OEBPS/images` |
| `PLANT_INAT_ROOT` | folder holding `inaturalist_images/` |
| `PLANT_PAPERS_XLSX` | label spreadsheet of the merged public datasets |
| `PLANT_PAPERS_IMG_DIRS` | folders with the hash-named dataset images |
| `PLANT_AGENT_MODEL` | OpenRouter model id |

Any folder-per-class dataset placed in `datasets/<name>/` is picked up as a verified reference set.

## Use

```
python plant_agent.py selftest                      # offline check, no key needed
python plant_agent.py diagnose photo.jpg -q "What is wrong with this plant?"
streamlit run streamlit_app.py
```

## Evaluation

```
python eval_agent.py pest --n 40 --baseline
python eval_agent.py disease --set PlantDoc --n 40 --baseline
python eval_agent.py healthy --dir path/to/healthy_photos --baseline
```

Query photos come from a held-out pool and are never shown back as references. The baseline is the same model
without tools; on the labelled sets it is given the label list so that wording differences do not count against it.

## Limits

- Retrieval over the field guide is BM25 only.
- The field guide covers North American garden insects and mites; diseases rely on the reference photo sets and
  on the model's own knowledge.
- PlantVillage and TomatoLeaves contain near-duplicate photos across splits, so accuracy on them is optimistic.
