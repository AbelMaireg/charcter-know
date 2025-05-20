# Building Model
- Requires a manual execution of python script from jupyter environment.

## Opening the Jupyter Notebook
```bash
  cd model
  docker compose -f tensorflow.compose.yml up
```

# Staring the Server

```bash
  # cd to root directory
  # cd ..  

  pip install -r requirements.txt

  flask --app app.py --debug run
```
