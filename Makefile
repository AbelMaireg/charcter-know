NOTEBOOK_COMPOSE_PATH := notebook.compose.yml

install:
	pip install -r requirements.txt

run:
	flask --app app.py --debug run 

notebook:
	jupyter notebook
