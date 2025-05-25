# Building Model
- Requires a manual execution of python script from jupyter environment.

## Opening the Jupyter Notebook
```bash
  make notebook
```

# Staring the Server

```bash
  pip install -r requirements.txt

  flask --app app.py --debug run
```

# Test

*input*:
```bash
  curl -X POST http://localhost:5000/predict -F "image=@./local-test-image/2.png"
```

*output*:
```json
{
  "confidence": 0.9999767541885376,
  "predicted_digit": 2
}
```
