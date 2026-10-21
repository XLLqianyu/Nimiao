## Summary

<!-- What changes, and why. Link the issue this closes, if any. -->

Closes #

## Type of change

- [ ] Bug fix (`fix`)
- [ ] New feature (`feat`)
- [ ] Refactor or performance work (`refactor`, `perf`)
- [ ] Documentation or repository tooling (`docs`, `ci`, `chore`)
- [ ] Breaking change (`!`)

## How it was tested

<!-- Commands you ran, and anything a reviewer should try by hand. -->

```bash
cargo test --workspace
```

## Checklist

- [ ] The branch targets `develop`, not `main`.
- [ ] Commit subjects follow Conventional Commits.
- [ ] `cargo fmt --all -- --check` passes.
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` passes.
- [ ] `cargo test --workspace` passes.
- [ ] New behaviour has tests, or the pull request explains why it cannot be tested.
- [ ] Documentation is updated when behaviour or the public API changes.
