import os, sys
from pathlib import Path

module_1 = 'model'
module_2 = 'utils'

list_of_files = [
    f'{module_1}/layers.py',
    f'{module_1}/activations.py',
    f'{module_1}/losses.py',
    f'{module_1}/network.py',
    f'{module_2}/dataset.py',
    f'{module_2}/visualization.py',
    'main.py',
    'requirements.txt'
]



for file in list_of_files:
    filepath = Path(file)
    file_dir, file_name = os.path.split(filepath)

    if file_dir != '':
        os.makedirs(file_dir, exist_ok=True)

    if((not os.path.exists(filepath)) or (os.path.getsize(filepath)==0)):
        with open(filepath, 'w') as f:
            pass





    