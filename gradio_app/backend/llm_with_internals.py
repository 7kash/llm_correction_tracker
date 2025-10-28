"""
Local LLM with Internal State Extraction

Supports extracting:
- Attention weights per layer
- Hidden states per layer
- Intermediate logits (logit lens)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class LLMWithInternals:
    """
    Wrapper for local LLM that extracts internal states during generation.

    Designed for educational purposes - tracks how the model processes
    corrections and changes its responses.
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: str = "auto"
    ):
        """
        Initialize model with internal state tracking.

        Parameters
        ----------
        model_name : str
            HuggingFace model identifier
        device : str
            'auto', 'cpu', 'cuda', or 'mps'
        """
        print(f"🔧 Loading model: {model_name}")

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        print(f"📍 Using device: {device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with output flags
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_attentions=True,
            output_hidden_states=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            attn_implementation="eager"  # Use eager attention to support output_attentions
        ).to(device)

        self.model.eval()  # Inference mode

        print(f"✅ Model loaded: {self.model.config.num_hidden_layers} layers")

    def format_chat_prompt(
        self,
        question: str,
        history: Optional[List[Tuple[str, str]]] = None
    ) -> str:
        """
        Format question + history as chat prompt using TinyLlama's chat template.

        Parameters
        ----------
        question : str
            Current question
        history : list of (correction, response) tuples
            Previous turns
        """
        # Build messages in proper format
        messages = []

        if history:
            for correction, response in history:
                messages.append({"role": "user", "content": correction})
                messages.append({"role": "assistant", "content": response})

        messages.append({"role": "user", "content": question})

        # Use tokenizer's chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt

    def generate_with_internals(
        self,
        question: str,
        history: Optional[List[Tuple[str, str]]] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> Dict:
        """
        Generate response with full internal state extraction.

        Returns
        -------
        dict with keys:
            - response: str
            - tokens: list[str]
            - attentions: list of np.ndarray (layers x heads x seq x seq)
            - hidden_states: list of np.ndarray (layers x seq x hidden_dim)
            - logits_per_layer: list of np.ndarray (layers x vocab_size)
            - input_tokens: list[str] (prompt tokens)
        """
        prompt = self.format_chat_prompt(question, history)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]

        # Step 1: Generate response (without internals to avoid dimension issues)
        # Add stop tokens to prevent generating fake conversation
        stop_token_ids = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),  # TinyLlama chat end token
        ]
        # Remove None values
        stop_token_ids = [tid for tid in stop_token_ids if tid is not None]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=stop_token_ids
            )

        # Extract response text
        full_sequence = outputs[0]
        response_ids = full_sequence[input_length:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Extract tokens
        all_tokens = self.tokenizer.convert_ids_to_tokens(full_sequence)
        input_tokens = all_tokens[:input_length]
        output_tokens = all_tokens[input_length:]

        # Step 2: Forward pass on FULL sequence to extract internals
        # This gives us consistent shapes for attention/hidden states
        full_inputs = {"input_ids": full_sequence.unsqueeze(0)}

        with torch.no_grad():
            forward_outputs = self.model(
                **full_inputs,
                output_attentions=True,
                output_hidden_states=True
            )

        # Process attention weights (now consistent shape!)
        attentions = self._process_attentions_from_forward(forward_outputs.attentions)

        # Process hidden states
        hidden_states = self._process_hidden_states_from_forward(forward_outputs.hidden_states)

        # Extract logits per layer (for logit lens)
        logits_per_layer = self._extract_logit_lens(hidden_states)

        return {
            "response": response_text.strip(),
            "tokens": output_tokens,
            "input_tokens": input_tokens,
            "attentions": attentions,  # (layers, seq, seq)
            "hidden_states": hidden_states,  # (layers, hidden_dim)
            "logits_per_layer": logits_per_layer,  # (layers, vocab_size)
            "num_layers": self.model.config.num_hidden_layers
        }

    def _process_attentions_from_forward(
        self,
        attentions_tuple: Tuple
    ) -> np.ndarray:
        """
        Process attention weights from a forward pass.

        Returns mean-over-heads attention for all layers.
        Shape: (num_layers, seq_len, seq_len)
        """
        # attentions_tuple is (num_layers,) tuple
        # Each element has shape (batch=1, heads, seq, seq)

        layer_attentions = []
        for layer_attn in attentions_tuple:
            # layer_attn: (batch=1, heads, seq, seq)
            mean_attn = layer_attn[0].mean(dim=0).cpu().numpy()  # (seq, seq)
            layer_attentions.append(mean_attn)

        return np.stack(layer_attentions, axis=0)  # (layers, seq, seq)

    def _process_hidden_states_from_forward(
        self,
        hidden_states_tuple: Tuple
    ) -> np.ndarray:
        """
        Process hidden states from a forward pass.

        Returns hidden states for the last token across all layers.
        Shape: (num_layers, hidden_dim)
        """
        # hidden_states_tuple is (num_layers + 1,) tuple (includes embedding layer)
        # Each element has shape (batch=1, seq, hidden_dim)
        # Skip the first (embedding layer), use layers 1 onwards

        layer_hidden = []
        for layer_h in hidden_states_tuple[1:]:  # Skip embedding layer
            # layer_h: (batch=1, seq, hidden_dim)
            last_token_hidden = layer_h[0, -1, :].cpu().numpy()  # (hidden_dim,)
            layer_hidden.append(last_token_hidden)

        return np.stack(layer_hidden, axis=0)  # (layers, hidden_dim)

    def _extract_logit_lens(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Apply LM head to hidden states at each layer (logit lens).

        Parameters
        ----------
        hidden_states : np.ndarray
            Shape (num_layers, hidden_dim)

        Returns
        -------
        logits_per_layer : np.ndarray
            Shape (num_layers, vocab_size)
        """
        # Get LM head (tied to embeddings in most models)
        lm_head = self.model.lm_head

        logits_per_layer = []
        for layer_h in hidden_states:
            # Convert back to tensor
            h_tensor = torch.from_numpy(layer_h).to(self.device).unsqueeze(0)  # (1, hidden_dim)

            with torch.no_grad():
                logits = lm_head(h_tensor)  # (1, vocab_size)

            logits_per_layer.append(logits[0].cpu().numpy())

        return np.stack(logits_per_layer, axis=0)  # (layers, vocab_size)

    def get_top_k_tokens(
        self,
        logits: np.ndarray,
        k: int = 5,
        temperature: float = 1.0
    ) -> Tuple[List[str], np.ndarray]:
        """
        Get top-k tokens and their probabilities from logits.

        Parameters
        ----------
        logits : np.ndarray
            Shape (vocab_size,)
        k : int
            Number of top tokens
        temperature : float
            Softmax temperature

        Returns
        -------
        tokens : list[str]
        probs : np.ndarray
        """
        # Apply temperature and softmax
        scaled_logits = logits / temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exp_logits / exp_logits.sum()

        # Get top-k
        top_idx = np.argsort(probs)[-k:][::-1]
        top_tokens = [self.tokenizer.decode([idx]) for idx in top_idx]
        top_probs = probs[top_idx]

        return top_tokens, top_probs


if __name__ == "__main__":
    # Test the module
    print("🧪 Testing LLMWithInternals...")

    llm = LLMWithInternals(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    result = llm.generate_with_internals(
        question="What is the capital of Australia?",
        max_new_tokens=20
    )

    print(f"\n📝 Response: {result['response']}")
    print(f"🔢 Output tokens: {len(result['tokens'])}")
    print(f"👁️ Attention shape: {result['attentions'].shape}")
    print(f"🧠 Hidden states shape: {result['hidden_states'].shape}")
    print(f"📊 Logits per layer shape: {result['logits_per_layer'].shape}")

    # Test top-k extraction
    last_layer_logits = result['logits_per_layer'][-1]
    tokens, probs = llm.get_top_k_tokens(last_layer_logits, k=5)
    print(f"\n🏆 Top-5 tokens at last layer:")
    for tok, prob in zip(tokens, probs):
        print(f"  {tok}: {prob:.3f}")

    print("\n✅ Module test passed!")
