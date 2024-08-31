import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

project_name = 'StudentsPerformance'

list_of_files = [
    '.github/workflows/.gitkeep',
    f'{project_name}/__init__.py',
    f'{project_name}/entity/__init__.py',
    f'{project_name}/config/__init__.py',
    f'{project_name}/component/__init__.py',
    f'{project_name}/component/data_ingestion.py',
    f'{project_name}/component/data_transformation.py',
    f'{project_name}/component/model_trainer.py',
    f'{project_name}/exception.py',
    f'{project_name}/logger.py',
    f'{project_name}/utils.py',
    f'templates/index.html',
    f'config/config.yaml',
    f'requirements.txt',
    f'setup.py',
    f'main.py'
]

for filepath in list_of_files:
    path = Path(filepath)

    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f'Created directory at {path.parent} for {path.name}')

    if not path.exists() or path.stat().st_size() == 0:
        path.touch()
        logging.info(f'Created empty file {path}')
    else:
        logging.info(f'{path.name} already exists')









