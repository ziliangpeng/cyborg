# Halftone Web

Simple website for `visual.halftone` that lets you upload an image and preview all 8 styles.

## Run

```bash
bazel run //apps/halftone_web:halftone_web -- --port 8080
```

Then open:

- `http://127.0.0.1:8080/`

## Notes

- Uploads are processed in-memory and returned as PNG data URLs.
- Default parameters are hard-coded in `apps/halftone_web/server.py`.
