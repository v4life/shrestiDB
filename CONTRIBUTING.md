# CONTRIBUTING

Thank you for your interest in contributing to ShrestiDB! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, inclusive, and professional in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally: `git clone https://github.com/YOUR_USERNAME/shrestidb.git`
3. **Create a feature branch**: `git checkout -b feature/your-feature-name`
4. **Set up development environment**:
   ```bash
   cargo build
   cargo test
   cargo fmt
   cargo clippy
   ```

## Development Workflow

### Code Style

- Follow Rust conventions using `rustfmt`:
  ```bash
  cargo fmt
  ```

- Run clippy for linting:
  ```bash
  cargo clippy -- -D warnings
  ```

- Write clear, descriptive comments for complex logic
- Use meaningful variable and function names

### Testing

- Write tests for new functionality:
  ```bash
  #[test]
  fn test_my_feature() {
      // Test code here
  }
  ```

- Run tests before submitting PR:
  ```bash
  cargo test
  ```

- Ensure benchmarks compile:
  ```bash
  cargo bench --no-run
  ```

### Performance

- Profile hot paths:
  ```bash
  cargo build --release
  ```

- Use criterion for benchmarking:
  ```bash
  cargo bench
  ```

- Document performance implications in comments

## Areas for Contribution

### High Priority

1. **Performance Optimization**
   - SIMD improvements
   - Memory efficiency
   - Cache optimization

2. **SQL Features**
   - Advanced parsing (window functions, CTEs)
   - Subquery support
   - JOIN improvements

3. **Index Enhancements**
   - Multi-dimensional indexing
   - Adaptive model selection
   - Index persistence

### Medium Priority

1. **Testing & Robustness**
   - Fuzzing
   - Stress testing
   - Edge case handling

2. **Documentation**
   - API documentation
   - Usage examples
   - Architecture diagrams

3. **Tools**
   - Profiling tools
   - Query analysis tools
   - Statistics gathering

### Lower Priority

1. **Advanced Features**
   - Distributed query execution
   - GPU acceleration
   - Reinforcement learning optimization

## Submitting Changes

### Before Submitting

1. **Format your code**: `cargo fmt`
2. **Lint**: `cargo clippy`
3. **Test**: `cargo test`
4. **Benchmark (if relevant)**: `cargo bench --no-run`
5. **Update documentation** if needed

### Pull Request Process

1. **Push to your fork**: `git push origin feature/your-feature-name`
2. **Create a Pull Request** on GitHub
3. **Fill out the PR template**:
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Performance improvement
   - [ ] Documentation
   
   ## Related Issues
   Closes #(issue number)
   
   ## Testing
   How to test these changes
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Tests pass
   - [ ] Documentation updated
   ```

### PR Review Process

- Maintainers will review your PR
- Address requested changes
- Respond to feedback promptly
- PRs should be merged within 1-2 weeks if approved

## Reporting Issues

### Bug Reports

Include:
- Rust version: `rustc --version`
- System info
- Minimal reproduction code
- Expected vs actual behavior
- Error messages/logs

### Feature Requests

Include:
- Use case/motivation
- Proposed solution
- Alternative approaches
- Potential impact

## Commit Message Guidelines

Use descriptive commit messages:

```
type(scope): Brief description

Detailed explanation of changes if needed.
References issue #123.
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `perf`: Performance improvement
- `docs`: Documentation
- `test`: Testing
- `refactor`: Code refactoring
- `ci`: CI/CD changes

Example:
```
feat(index): Add learned index prefetching

Implement Markov chain-based prefetching for buffer pool
to predict next page access and reduce cache misses.
Improves TPC-H performance by 15% on typical workloads.
```

## Documentation

### Code Comments

- Document public APIs with rustdoc:
  ```rust
  /// Brief description
  ///
  /// Detailed explanation
  ///
  /// # Example
  /// ```
  /// let result = function();
  /// ```
  pub fn function() {}
  ```

- Document WHY, not just WHAT
- Include examples for complex functions

### Project Documentation

- Update README.md for user-facing changes
- Update DESIGN.md for architectural changes
- Add examples for new features

## Performance Benchmarking

### Adding Benchmarks

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_my_feature(c: &mut Criterion) {
    c.bench_function("my_feature", |b| {
        b.iter(|| {
            // Benchmark code
        });
    });
}

criterion_group!(benches, bench_my_feature);
criterion_main!(benches);
```

### Running Benchmarks

```bash
cargo bench --bench my_benchmark
```

### Interpreting Results

- Watch for significant regressions
- Document expected performance
- Compare before/after on same hardware

## Building for Release

```bash
# Clean build
cargo clean

# Release build with optimizations
cargo build --release

# Run tests on release build
cargo test --release

# Benchmark release build
cargo bench --release
```

## Debugging

### Common Commands

```bash
# Verbose output
RUST_BACKTRACE=1 cargo test

# With logging
RUST_LOG=debug cargo run --example tpc_h

# Memory profiling
valgrind cargo run --example learned_index_demo
```

## Questions?

- Check [DESIGN.md](DESIGN.md) for architecture
- Review existing issues and PRs
- Ask in PR comments
- Open a discussion issue

## Recognition

Contributors will be:
- Added to CONTRIBUTORS.md
- Mentioned in release notes
- Credited in commit messages

Thank you for contributing! 🎉
