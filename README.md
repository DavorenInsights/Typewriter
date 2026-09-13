# Field Notes Typewriter

A Streamlit app that renders editable text as imperfect A5 typewriter pages.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What it does

- A5 page output
- character-by-character jitter
- uneven ribbon darkness
- occasional double strikes
- ink dropout
- baseline wander
- paper texture
- fixed random seed for reproducible pages
- optional TTF/OTF font upload
- optional reference-photo texture influence
- PNG downloads and a ZIP of all pages

The app does not require an AI image model. Your words remain exact and selectable in the input, while the page image is generated deterministically.


## v2 refinements

This version reduces vertical wobble heavily. The intended look is now:
- straight mechanical baselines within a sentence/line
- slightly different line starts / paragraph indents
- occasional double spacing between words
- uneven ribbon darkness
- rare double strikes and dropouts
- much less character rotation
