"""SKILLSHOT — mothership-directed swarm of 1-bit experts over a shared Ortho-LoRA cache.

Planes:
  ortho_lora      orthogonal LoRA skill primitive (kept from O-TITANS)
  ternary         1.58-bit BitNet expert substrate (QAT-trainable)
  looped_expert   Track-A weight-tied recurrent-depth expert (Geiping)
  memory_gate     MIRAS/TITANS recurrent memory gate (data-dependent retention/surprise)
  projected_lora  Text-to-LoRA hypernetwork (cache-miss contingency)
  adapter_cache   LRU VRAM cache + disk store (the real "turbocache")
  registry        semantic skill registry (miss detection)
  memorialize     project -> test -> memorialize loop
  consensus       two-step firing (draft -> sync -> vote)
  expert          expert interfaces + MockExpert
  mothership      the orchestrator
"""
__all__ = [
    "ortho_lora", "ternary", "looped_expert", "memory_gate", "projected_lora",
    "adapter_cache", "registry", "memorialize", "consensus", "expert", "mothership",
]
__version__ = "0.0.1"
