"""Erstellung der vergifteten Datensätze für RQ2.2

"""
from implementation.rq2.rq2_2.poisoner import SinglePairPoisoner
from implementation.shared import paths

PAIR_KEY_BY_RQ = {"RQ2.2-sim": "sim_pair", "RQ2.2-dissim": "dissim_pair"}


def generate(data_config, rq):
    """Startet den Poisoning-Prozess für ein spezifisches Klassenpaar für SIM oder DISSIM

    Args:
        data_config (dict): Das geladene Dictionary der Daten-Konfigurationsdatei 
        rq (str): Der exakte Bezeichner des auszuführenden Experiments
                  Muss "RQ2.2-sim" oder "RQ2.2-dissim" sein

    Raises:
        KeyError: Wenn ein `rq`-String übergeben wird, der nicht 
                  in `PAIR_KEY_BY_RQ` definiert ist
    """
    poisoning_cfg = data_config["poisoning"]
    rq2_2_cfg = data_config["rq2_2"]

    tag = rq.removeprefix("RQ2.2-")
    target_pair = rq2_2_cfg[PAIR_KEY_BY_RQ[rq]]
    print(f"{rq}: Paar {target_pair}")

    SinglePairPoisoner(
        train_h5_path=paths.hdf5_path("train"),
        output_dir=paths.POISONED_DIR,
        seeds=poisoning_cfg["seeds"],
        rates=rq2_2_cfg["rates"],
        num_classes=poisoning_cfg["num_classes"],
        pair=target_pair,
        tag=tag).run()
