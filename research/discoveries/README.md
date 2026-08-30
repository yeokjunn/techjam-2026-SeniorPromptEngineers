# Persistent Research Discoveries

Autonomous researcher passes write web-search-backed ideas to
`discoveries.json` in this directory. These records are prompt context for later
iterations; they do not create new executable experiment families by themselves.

To make a discovery trainable as a new family, register it in `src/agent/families.py`
and add the corresponding runtime, policy, and safety support.
