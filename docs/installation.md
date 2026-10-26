# Installation

`integrator-lib` supports Python 3.12 through 3.14. Install the package and its
declared runtime dependencies from the repository root with:

```bash
python -m pip install .
```

For development, install the checkout in editable mode with the test
dependencies:

```bash
python -m pip install --no-build-isolation -e ".[dev]"
```

The HyQD development environment currently uses NumPy 2.5.2 and SciPy 1.18.0.
The authoritative supported version ranges are declared in
[`pyproject.toml`](../pyproject.toml).

Run the complete test suite with:

```bash
python -m pytest
```

When developing a consuming package such as `grid-tdse`, install this checkout
in the same environment before installing the consumer:

```bash
python -m pip install --no-build-isolation -e /path/to/integrator-lib
python -m pip install --no-build-isolation -e /path/to/grid-tdse
```

After pulling changes, an editable installation normally sees Python-source
updates immediately. Re-run the editable installation when package metadata,
dependencies, entry points, or the build configuration have changed.
