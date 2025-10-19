# rusty-axon

Experimental Rust port of the micrograd autograd engine as a part of AI course term project. 

## Project structure

- `src/lib.rs`: crate entry point that exposes the core modules.
- `src/engine`: low-level autograd primitives (`Value`, graph tracking, ops).
- `src/nn`: neural-network building blocks composed from the engine.
- `src/optim`: optimizers for training loops.
- `src/main.rs`: simple binary that links against the library crate.

Peoject is on early stage of development and not ready for use.
## Getting started

```bash
cargo check
```

## License

Choose a license once the implementation stabilises. Until then, the project is
shared informally for learning purposes.
