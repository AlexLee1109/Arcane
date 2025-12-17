import time
import torch
import tiktoken

from Arcane.gpt import GPT, GPTConfig

CONFIG = GPTConfig(n_layer=24, n_head=16, n_kv_head=4, n_embd=1344)

CHECKPOINT_PATH = "models/arcane2.pt"

MAX_TOKENS = 100
TEMPERATURE = 0.9
TOP_K = 40
TYPING_DELAY = 0.015

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(device: torch.device) -> GPT:
    model = GPT(CONFIG).to(device).eval()

    try:
        print(f"Loading checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model"])
        print("Checkpoint loaded.")
    except Exception as e:
        print(f"Checkpoint load failed: {e}")
        print("Using randomly initialized weights.")

    if device.type in {"cuda", "mps"}:
        print("Compiling model...")
        model = torch.compile(model, dynamic=False)

    return model

class SinonChat:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.inference_mode()
    def stream_generate(self, prompt):
        """
        Generate text and yield characters incrementally
        (token-level generation, character-level display).
        """
        enc = self.tokenizer

        tokens = enc.encode(prompt, allowed_special={"<|endoftext|>"})
        tokens = tokens[-CONFIG.block_size:]

        eos_token = enc.encode(".")[0]

        token_ids = self.model.generate(
            tokens=tokens,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            eos_tokens=eos_token,
            min_tokens=int(MAX_TOKENS * 0.8),
            kv_cache=None,
        )

        for token_id in token_ids:
            text = enc.decode([token_id])
            text = text.replace("�", "'")

            for ch in text:
                yield ch

    def run(self) -> None:
        print("=== Sinon Chatbot ===")
        print("Type 'quit' to exit.\n")

        while True:
            try:
                prompt = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                return

            if not prompt:
                continue
            if prompt.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                return

            print("Sinon:", end=" ", flush=True)

            start = time.time()
            chars = 0
            first_char = True

            for ch in self.stream_generate(prompt):
                if first_char and ch.isspace():
                    continue
                first_char = False

                print(ch, end="", flush=True)
                chars += 1
                time.sleep(max(0.001, TYPING_DELAY))

            elapsed = time.time() - start
            speed = chars / elapsed if elapsed > 0 else 0.0

            print()
            print(f"({speed:.1f} chars/sec • {elapsed:.2f}s)\n")

def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    model = load_model(device)
    tokenizer = tiktoken.get_encoding("gpt2")

    SinonChat(model, tokenizer, device).run()


if __name__ == "__main__":
    main()