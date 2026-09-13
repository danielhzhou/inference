from dataclasses import dataclass

@dataclass
class InferenceRequest:
    prompt: list[int] # list of tokens
    max_tokens: int # max total number of tokens generated incl the prompt
    id: int

    generated_tokens: list[int] # list of newly generated tokens
    num_cached: int = 0 # number of tokens we actually have kv values for
