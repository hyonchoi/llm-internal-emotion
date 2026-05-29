# How to Read the Emotion Plots

## The Heatmap

The heatmap shows the full emotion picture across every token in the conversation.

- **X-axis** — each column is one token (word or punctuation mark) in the conversation
- **Y-axis** — each row is one emotion
- **Color** — red means that emotion is strongly active; blue means it is suppressed; white is neutral

**What to look for:**
- A bright red patch in the "sad" row around the word "died" tells you: the model internally registered sadness at that point
- Yellow/cream background = Human turn; light blue background = Assistant turn

## The Timeline

The timeline zooms in on the top-5 most active emotions and shows how each one rises and falls across the conversation.

- Each line is one emotion
- Peaks = the model is strongly activating that emotion at that token
- Valleys = that emotion is suppressed

The token just before "Assistant:" (the colon token) is especially interesting — research shows that the model's emotion at *that exact point* predicts the emotional tone of the response it's about to generate.

## The Numbers (JSON / CSV)

Each row in the CSV corresponds to one token. The numeric value in each emotion column is the "dot product" — how strongly that token's hidden state points in the emotion's direction. Positive = active; negative = suppressed; near zero = neutral.
