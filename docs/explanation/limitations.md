# Limitations of This Approach

## What This Tool Cannot Tell You

### 1. Whether the model "feels" anything
The numbers this tool produces are geometric measurements — how much an internal vector points in the direction we labeled "happy." They do not tell us whether the model has any subjective experience. "Activation" ≠ "feeling."

### 2. Whether the labeling is correct
We call a direction "sadness" because it activates when we show the model sad stories. But the same direction might be capturing something adjacent — loss, reflection, downward valence. The label is ours, not the model's.

### 3. Persistent emotional states
Transformer models have no memory between tokens except what fits in the context window. Each token's hidden state is freshly computed from the full context. The model doesn't "carry" an emotion the way a person does — it re-reads the entire conversation and re-computes everything at each step.

### 4. Generalization across models
Emotion vectors computed from Mistral 7B may not transfer to Mixtral 8x7B or a fine-tuned variant. Each model has its own internal geometry.

### 5. External validation
We have no ground truth for what the model's "real" internal state is. We can only measure structure in the activations. Whether that structure corresponds to anything meaningful is a scientific question, not an engineering one.

## Bottom Line

Use this tool as an **exploratory instrument**, not a measurement device. It can surface interesting patterns and generate hypotheses. It cannot prove or disprove that a model has emotions.
