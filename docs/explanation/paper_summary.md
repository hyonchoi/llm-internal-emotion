# Paper Summary: Emotions Inside a Language Model

**Source:** "Emotion Concepts and their Function in a Large Language Model"  
Sofroniew et al., Anthropic (arXiv:2604.07729v1)

## What Did They Discover?

1. **Emotions have geometry.** Inside Claude, emotions correspond to specific directions in the model's internal space — just like "up" and "down" are directions in physical space.

2. **These directions are causal.** It's not just that the model *represents* emotions — they actually influence what the model says next. When researchers artificially amplified the "fear" direction, the model started generating fearful text. When they suppressed it, the fear disappeared.

3. **Different layers tell different stories.** Early layers encode the emotional meaning of the *current* token. Later layers (around 2/3 depth) encode the emotion that will *drive* the next generated token.

4. **The colon predicts the response.** In Claude's conversation format, the ":" token before the Assistant's response carries the strongest signal about what emotional tone the response will have.

## Why Does This Matter?

This research suggests that "emotion" in language models isn't just a metaphor — it's a measurable, functional internal state that can be studied, amplified, suppressed, or redirected.

This tool applies these findings to open-source Mistral models, making the same kind of analysis accessible without needing access to Claude.
