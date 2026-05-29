import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from emotion_tracer.config import EmotionTracerConfig

class ModelWrapper:
    """Wraps a HuggingFace model for hidden state extraction.

    Loads models in bfloat16 with automatic device placement (D9),
    and casts tensors to float32 before NumPy conversion (D11).
    """

    def __init__(self, config: EmotionTracerConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load model and tokenizer from HuggingFace.

        Uses bfloat16 + device_map='auto' (D9) for efficient GPU utilization.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            output_hidden_states=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

    @property
    def num_layers(self) -> int:
        """Total number of transformer layers in the loaded model."""
        return self.model.config.num_hidden_layers

    def get_tokens(self, text: str) -> list[str]:
        """Tokenize text and return the list of token strings."""
        inputs = self.tokenizer(text, return_tensors="pt")
        ids = inputs["input_ids"][0].tolist()
        return self.tokenizer.convert_ids_to_tokens(ids)

    def get_hidden_states(self, text: str, layer_indices: list[int]) -> dict[int, np.ndarray]:
        """Extract hidden states for specified transformer layers.

        Returns a dict mapping layer index to a NumPy array of shape
        [seq_len, hidden_dim].

        Notes:
            - Uses ``torch.no_grad()`` to avoid building the computation graph.
            - Casts bfloat16 tensors to float32 via ``.float()`` before
              NumPy conversion (D11), since NumPy does not support bfloat16.
            - ``outputs.hidden_states`` is a tuple of (num_layers+1) tensors;
              index 0 is the embedding output, index i is layer i's output.
            - We add 1 to the requested layer index to skip the embedding layer.
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        result = {}
        for idx in layer_indices:
            hs = outputs.hidden_states[idx + 1]  # [1, seq_len, hidden_dim]
            result[idx] = hs[0].float().cpu().numpy()  # D11: .float() for bfloat16 -> NumPy
        return result

    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Run autoregressive generation and return only the newly generated text."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_ids = output_ids[0][input_len:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)
