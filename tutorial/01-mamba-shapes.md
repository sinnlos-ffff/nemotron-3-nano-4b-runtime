# Exercise 01: Where does each channel go?

First read [the architecture primer](00-architecture.md) for state, groups,
the SSM equations, and the complete mixer diagram.

Only edit the three properties in `src/nemotron/config.py`.
Leave the mixer and the model alone for now. You write the tensor/model logic;
I review it and give the next exercise after it is correct.

Let batch size be N and sequence length be L. The input is `[N, L, hidden_size]`.
One linear projection will produce five contiguous pieces in this order:

| Piece | Meaning | Width |
| --- | --- | --- |
| gate | Later controls the SSM output | all head channels |
| x | Input to the state recurrence | all head channels |
| B | Input-dependent state write vectors | groups times state size |
| C | Input-dependent state read vectors | groups times state size |
| dt | Input-dependent time steps | number of heads |

Only x, B, and C pass through the causal convolution. B and C are shared
across heads within a group. Each head has its own channels and time step.
After reshaping, x has shape `[N, L, heads, head_dim]`; each of B and C has
shape `[N, L, groups, state_size]`.

Your task:

1. Derive `inner_size` from the head count and head dimension.
2. Derive `conv_dim` by counting the channels passing through convolution.
3. Derive `projection_size` by counting all five pieces.
4. Remove the two exercise-test `xfail` decorators and run `uv run pytest`.
5. Tell me why B and C use the group count rather than the head count.

For the real checkpoint, hidden_size is 3136, heads is 96, head_dim is 80,
groups is 8, and state_size is 128. The checkpoint also carries `expand=2`,
but this mixer explicitly uses heads times head_dim. Follow that definition.

Use ordinary integer arithmetic. No tensors or model allocations are needed.
When ready, ask me to review your diff. I will identify issues and provide hints
without rewriting your solution.
