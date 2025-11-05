# Filters Directory

This directory is for organizing filters by category when the repository grows.

## Suggested Structure

```
filters/
├── core/           # Core memory layers (episodic, semantic, emotional)
├── advanced/       # Advanced layers (temporal, associative, symbolic)
├── experimental/   # Experimental features
└── utils/          # Shared utilities
```

## Current Status

Currently, all filters are in the root directory for simplicity. As the project grows, filters can be organized here while maintaining backward compatibility through symbolic links or import aliases.

## Usage

When this directory is populated:

1. **Organize by type:**
   ```
   filters/
   ├── core/
   │   ├── episodic.py
   │   ├── semantic.py
   │   └── emotional.py
   ├── advanced/
   │   ├── temporal.py
   │   ├── associative.py
   │   └── symbolic.py
   ```

2. **Add to Open WebUI:**
   - Import each filter individually
   - Or use package imports if Open WebUI supports them

3. **Maintain compatibility:**
   - Keep root-level filters for existing users
   - Gradually migrate to organized structure

## Contributing

When adding new filters:
- Place core functionality in `core/`
- Experimental features in `experimental/`
- Shared code in `utils/`
