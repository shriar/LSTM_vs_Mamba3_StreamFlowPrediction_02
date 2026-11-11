"""Single source of truth for the experiment directory layout.

Every path is absolute and independent of the working directory:

    experiment/
        src/        python modules                  <- SRC_DIR (this file)
        data/       inputs: pickles, gauge ids, CAMELS attributes
        output/     run outputs, one folder per run (seed in the name)
        figures/    rendered figures: paper/, other/, variants/
        logs/       running_log*.txt
        results/    derived tables: paired tests, exported numbers
        archive/    retired outputs and scripts
"""

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "output"
FIGURES_DIR = PROJECT_ROOT / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"
ARCHIVE_DIR = PROJECT_ROOT / "archive"
MPL_CONFIG_DIR = PROJECT_ROOT / ".matplotlib"
