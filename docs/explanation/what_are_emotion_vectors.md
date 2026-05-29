# What Are Emotion Vectors?

## The Basic Idea

Imagine a large language model as a vast, high-dimensional space — thousands of dimensions, each representing some abstract concept. When the model processes text, it moves through this space, creating a unique point (called a "hidden state") for each word it reads.

Researchers have discovered that emotions correspond to specific *directions* in this space. "Happy" is roughly one direction; "sad" is roughly another. These directions are called **emotion vectors**.

## How We Find Them

1. We generate dozens of short stories where a character explicitly feels a specific emotion (e.g., "She beamed with joy as she opened the gift").
2. We generate equally many "neutral" versions — same structure, but no emotion ("She opened the gift").
3. We run both sets through the model and record the internal activations.
4. We subtract the average neutral activation from the average emotional activation.
5. The result, after normalizing to unit length, is the **emotion vector** for that emotion.

Think of it like tuning a compass: the emotional stories pull the needle in one direction; neutral stories stay put; the difference tells us which way "happy" points.

## Why Does It Work?

Because the model learned from human text, and human text is soaked in emotion. The model implicitly learned to represent emotions in its internal geometry — not because we told it to, but because it had to in order to predict text accurately.
