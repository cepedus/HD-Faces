# HD Face Recognition

To run `cavaface-pytorch` adaptation:

- `cd cavaface`
- `pip install -r requirements.txt`
- check relative paths in `config.py` (`./data/highres/`, `./data/highres_eval/`, `./data/lowres/`, etc. directories must exist)
- check batch size, num_epoch, num_workers, etc for multi gpu
- to run: `python ./train.py`

Debug logs are available [here](https://drive.google.com/drive/folders/1rA_g5p0cDlsArIWo1twb6ASQ1LbLTpoP?usp=sharing).