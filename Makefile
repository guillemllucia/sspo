install:
	pip install -e .

retrain:
	python sspo/grid_search.py

get_best_model:
	python  -c "from sspo.grid_search import get_best_model_filename;get_best_model_filename()"
