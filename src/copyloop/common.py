"""CPU-only configuration and record utilities."""
import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent


def load(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def models():
    return load(PACKAGE / 'presets/models.json')


def preset(name):
    return models()[name]


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_rows(path, rows):
    with Path(path).open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def validate_model(path, config):
    """Validate revision provenance and recurrent depth before using local weights."""
    path = Path(path)
    provenance = load(path / 'copyloop_model.json')
    if any(provenance.get(k) != config[k] for k in ('repo', 'revision')):
        raise ValueError('Local model provenance differs from preset; use copyloop download')
    actual = load(path / 'config.json')
    # Ouro and Raven/Huginn expose different recurrence configuration fields.
    if config['loops'] == 4 and actual.get('total_ut_steps') != 4:
        raise ValueError('Ouro recurrent depth is not four')
    if config['loops'] == 32 and actual.get('mean_recurrence') != 32:
        raise ValueError('Raven/Huginn recurrent depth is not 32')
    return actual
