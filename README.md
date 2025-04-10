## Architecture:

## Usage:
- Clone the repository:
```
git clone https://github.com/quangliz/image-to-latex.git
cd thesis
```
- Install virtual environment and requirements:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
- Prepare data:
```
python scripts/prepare_data.py
```
- Train the model:
```
python main.py train
```
- Inference:
```
python main.py infer --image <path_to_image> --checkpoint <path_to_checkpoint>
```

### or
- use the notebook with instructions