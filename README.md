# HD Face Recognition

To run `cavaface-pytorch` adaptation:

- `cd cavaface`
- `pip install -r requirements.txt`
- check relative paths in `config.py` (`./data/highres/`, `./data/highres_eval/`, `./data/lowres/`, etc. directories must exist)
- check batch size, num_epoch, num_workers, etc for multi gpu
- to run: `python ./train.py`