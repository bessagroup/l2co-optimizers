# L2CO Optimizers

The bare optimizers of the L2CO ecosystem: what an optimizer *is* to
the rest of the stack (registry, container, spec, contract), and the
per-optimizer properties a meta-optimizer needs to switch between them.
Running an optimizer on a task, and choosing between optimizers, live
in `l2co`.

## Language

**Bare optimizer**:
An optimizer that optimizes a task directly — Adam, CMA-ES, SHADE,
L-BFGS — as opposed to a **meta-optimizer** (`l2co`, `rl2co`,
`agentic-l2co`) that selects among bare optimizers. Only bare
optimizers ship here; meta-optimizers register into the same registry
from their own packages.
_Avoid_: "static optimizer" (l2co_experiments' name for the same thing
in pipeline code), "base optimizer"

**TaskLike**:
The only view of a task this package takes: any object with `model`,
`loss_fn` and `pass_rng`. Structural, so `l2co_tasks.Task` satisfies it
without either package importing the other. Dimensionality is not part
of it — it is derived from `model` (`count_parameters`).
_Avoid_: importing `l2co_tasks.Task` here (the ruff ban will stop you)

**Registry**:
`optimizers` / `optimizer_mapping`: normalized optimizer name → factory.
Every factory takes the same keywords — `task=`, `opt_hash=`,
`bounded=`, `stop_fn=`, plus hyperparameters — and returns an
`UpdateClass`. `register_optimizer` is the extension point
meta-optimizers use.

**UpdateClass**:
The static, JIT-friendly container one optimizer is reduced to:
`init_fn`, `step_fn`, `popsize`, `hash`, `stop_fn`, and the handshake
hooks (`family`, `ask_fn`, `transfer_read_fn`, `transfer_write_fn`). A
container only — factories live next to the library they wrap.

**OptimizationStep**:
The serializable spec of one optimizer configuration: registry name,
hyperparameters, stopping criteria. What the Hydra configs instantiate,
and — through `.name` and `.hash` — the **schedule** key every stored
trajectory is filed under.
_Avoid_: "optimizer" when the configured instance is meant

**Schedule name**:
`OptimizationStep.name`: the `alias` hyperparameter if set, else the
namer registered for the optimizer (`register_schedule_namer`), else
the generic `{optimizer}_{k=v}` join. Databank keys depend on it, so a
change to it orphans stored data.

**Generation**:
One ask/eval/tell cycle: exactly one iteration of the active
optimizer, one history entry, one budget tick.
_Avoid_: "step" (overloaded with optimizer-internal steps), "epoch"

**Handshake**:
The state transfer applied when the active optimizer changes: which
population the incoming optimizer sees, and whether its internal
state is reset or continued. Named variants: `best`, `best_one`,
`reset_opt_state`, `continue_opt_state`. Which variants a given
optimizer needs is tabulated once, in `HANDSHAKE_POLICY`, and shared by
every consumer.

**Own ask**:
A per-optimizer property (`TransferSpec.own_ask`, surfaced as
`UpdateClass.ask_fn`): after a handshake writes its state, the
optimizer draws its own post-switch population instead of being handed
one. Set for every distribution-based algorithm and the mutation GAs;
never inferred from `family`.

**Menu**:
The ordered tuple of candidate optimizers (`OptimizationStep`s) a
meta-optimizer chooses from.
_Avoid_: "portfolio" (reserved for the benchmark algorithm portfolio
run during dataset creation), "optimizer list"

**Generation width**:
The fixed leading axis of every population a meta-optimizer emits:
`max(popsize)` over the menu.

**Evaluation width**:
How many rows of an emitted population a generation actually
evaluates: the active optimizer's `popsize`, or 1 for a gradient-based
one. Never larger than the generation width; `menu_loss_and_grad`
spends only this.

**Padding row**:
A row of an emitted population beyond the current evaluation width.
Carried as `NaN` and never evaluated.
_Avoid_: "dummy candidate", "slack slot"
