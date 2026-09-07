"""Gemeinsame Bausteine und Hilfsfunktionen der RQ3-Analyseskripte

Dieses Modul kapselt die Logik zum Auslesen und Parsen des Dateisystems. 
Alle Auswertungen in RQ3 basieren auf einer strikt festgelegten Ordnerstruktur.
Daher ist dieses Layout das Fundament für alle automatisierten Datenextraktionen:

Erwartetes Layout eines RQ3-Laufs:
experiments/RQ3/<defense>/<model>/<attack>/seed_<s>/rate_<r>/

Beispiel:
experiments/RQ3/spectral_signature/resnet18/RQ1/seed_42/rate_0.2/
"""
import re

from implementation.shared import paths

GROUP_KEYS = ["defense", "group", "model", "attack", "rate"]
RUN_PATTERN = re.compile(r"seed_(?P<seed>\d+)[\\/]+rate_(?P<rate>[\d.]+)$")


def defense_group(defense):
    """Ermittelt die Überkategorie (Gruppe) einer spezifischen Verteidigungsstrategie.
    
    Args:
        defense (str): Der exakte Name der Verteidigung
        
    Returns:
        str: Der Gruppenname
    """
    from implementation.shared.defenses.registry import DEFENSE_GROUPS
    return DEFENSE_GROUPS.get(defense, "unknown")


def attack_dir_metadata(attack_dir):
    """Extrahiert die Metadaten eines Experiments

    Args:
        attack_dir (Path): Pfad zum Angriffsordner (z.B. .../spectral/resnet/RQ1)

    Returns:
        tuple: (defense, group, model, attack)
    """
    defense, model, attack = attack_dir.parts[-3], attack_dir.parts[-2], attack_dir.parts[-1]
    return defense, defense_group(defense), model, attack


def seed_and_rate(run_dir):
    """Parst Seed und Vergiftungsrate aus dem Namen eines Ausführungsverzeichnisses

    Args:
        run_dir (Path): Pfad zum Ordner (z.B. .../seed_42/rate_0.2)

    Returns:
        tuple: (seed als int, rate als float) oder None
    """
    match = RUN_PATTERN.search(str(run_dir))
    if not match:
        return None
    return int(match.group("seed")), float(match.group("rate"))


def iter_rq3_runs(root=None):
    """Generator, der alle gültigen RQ3-Trainingsläufe iteriert und deren Metadaten liefert.

    Ablauf:
    1. Sucht mit `glob` nach der Datei `epoch_metrics.csv`
       Nur wenn diese existiert, war der Trainingslauf erfolgreich
    2. `.parents[1]` geht vom Pfad `.../telemetry/epoch_metrics.csv` zwei Ebenen 
       nach oben zum Ordner `.../rate_X/`
    3. Parst Seed und Rate aus diesem Ordner
    4. Geht weitere zwei Ebenen nach oben (`.parents[1]`), um Defense/Model/Attack 
       zu extrahieren

    Args:
        root (Path, optional): Verzeichnis der RQ3-Experimente

    Yields:
        tuple: (defense, group, model, attack, seed, rate, run_dir) für jeden Lauf
    """
    root = root or paths.RQ3_DIR
    for epoch_csv in sorted(root.glob("*/*/*/seed_*/rate_*/telemetry/epoch_metrics.csv")):
        run_dir = epoch_csv.parents[1]
        params = seed_and_rate(run_dir)
        if params is None:
            continue
        seed, rate = params
        defense, group, model, attack = attack_dir_metadata(run_dir.parents[1])
        yield defense, group, model, attack, seed, rate, run_dir


def write_csv(df, output_path, label):
    """Speichert einen DataFrame als CSV-Datei ab

    Args:
        df (pd.DataFrame): Die Tabellendaten
        output_path (Path): Zielpfad der Datei
        label (str): Label
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
