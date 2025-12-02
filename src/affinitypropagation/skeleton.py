"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
``[options.entry_points]`` section in ``setup.cfg``::

    console_scripts =
         fibonacci = affinitypropagation.skeleton:run

Then run ``pip install .`` (or ``pip install -e .`` for editable mode)
which will install the command ``fibonacci`` inside your current environment.

Besides console scripts, the header (i.e. until ``_logger``...) of this file can
also be used as template for Python modules.

Note:
    This file can be renamed depending on your needs or safely removed if not needed.

References:
    - https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    - https://pip.pypa.io/en/stable/reference/pip_install
"""

import argparse
import logging
import sys
import numpy as np


from affinitypropagation import AffinityPropagation

__author__ = "AH-repo"
__copyright__ = "AH-repo"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


# ---- Python API ----
# The functions defined in this section can be imported by users in their
# Python scripts/interactive interpreter, e.g. via
# `from affinitypropagation.skeleton import fib`,
# when using this Python module as a library.



def load_data(filepath):
    """Load data from a file (CSV, NPY, or TXT format)

    Args:
        filepath (str): path to the data file

    Returns:
        np.ndarray: loaded data array
    """
    try:
        if filepath.endswith('.npy'):
            data = np.load(filepath)
        elif filepath.endswith('.csv'):
            data = np.genfromtxt(filepath, delimiter=',')
        else:
            data = np.loadtxt(filepath)

        _logger.info(f"Loaded data with shape: {data.shape}")
        return data
    except Exception as e:
        _logger.error(f"Error loading data: {e}")
        raise


def run_clustering(data, damping=0.5, max_iter=200, convergence_iter=15,
                   preference=None, affinity='euclidean'):
    """Run Affinity Propagation clustering on the data

    Args:
        data (np.ndarray): input data array
        damping (float): damping factor (0.5 to 1.0)
        max_iter (int): maximum number of iterations
        convergence_iter (int): iterations with no change to stop
        preference (float): preference value (None for median)
        affinity (str): 'euclidean' or 'precomputed'

    Returns:
        tuple: (labels, cluster_centers, n_clusters)
    """
    _logger.info("Starting Affinity Propagation clustering...")

    ap = AffinityPropagation(
        damping=damping,
        max_iter=max_iter,
        convergence_iter=convergence_iter,
        preference=preference,
        affinity=affinity
    )

    labels = ap.fit_predict(data)
    n_clusters = len(ap.cluster_centers_indices_)

    _logger.info(f"Clustering complete. Found {n_clusters} clusters.")

    return labels, ap.cluster_centers_, n_clusters


def save_results(labels, output_file):
    """Save clustering results to a file

    Args:
        labels (np.ndarray): cluster labels
        output_file (str): output file path
    """
    try:
        np.savetxt(output_file, labels, fmt='%d')
        _logger.info(f"Results saved to {output_file}")
    except Exception as e:
        _logger.error(f"Error saving results: {e}")
        raise


# ---- CLI ----
# The functions defined in this section are wrappers around the main Python
# API allowing them to be called directly from the terminal as a CLI
# executable/script.


def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Affinity Propagation")

    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    return parser.parse_args(args)


def parse_clustering_args(args):
    """Parse command line parameters for clustering

    Args:
      args (List[str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Affinity Propagation Clustering Algorithm"
    )

    parser.add_argument(
        "input_file",
        nargs="?",
        help="Input data file (CSV, NPY, or TXT). Optional if using --data.",
        type=str,
        metavar="FILE"
    )

    parser.add_argument(
        "--data",
        dest="inline_data",
        help="Wprowadź dane ręcznie, format: '1 2; 3 4; 5 6'",
        type=str,
        default=None
    )

    parser.add_argument(
        "-o", "--output",
        dest="output_file",
        help="Output file for cluster labels (default: labels.txt)",
        type=str,
        default="labels.txt",
        metavar="FILE"
    )

    parser.add_argument(
        "-d", "--damping",
        dest="damping",
        help="Damping factor between 0.5 and 1.0 (default: 0.5)",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "-m", "--max-iter",
        dest="max_iter",
        help="Maximum number of iterations (default: 200)",
        type=int,
        default=200
    )

    parser.add_argument(
        "-c", "--convergence-iter",
        dest="convergence_iter",
        help="Iterations with no change to stop (default: 15)",
        type=int,
        default=15
    )

    parser.add_argument(
        "-p", "--preference",
        dest="preference",
        help="Preference value (default: median of similarities)",
        type=float,
        default=None
    )

    parser.add_argument(
        "-a", "--affinity",
        dest="affinity",
        help="Affinity type: 'euclidean' or 'precomputed' (default: euclidean)",
        type=str,
        choices=['euclidean', 'precomputed'],
        default='euclidean'
    )

    parser.add_argument(
        "-v", "--verbose",
        dest="loglevel",
        help="Set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )

    parser.add_argument(
        "-vv", "--very-verbose",
        dest="loglevel",
        help="Set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )

    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )

def parse_inline_data(text):
    """
    Format: "1 2; 3 4; 5 6"
    """
    rows = text.split(";")
    return np.array([list(map(float, r.split())) for r in rows])

def main(args):
    """
    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("Starting crazy calculations...")
    _logger.info("Script ends here")



def main_clustering(args):
    """Main function that runs the Affinity Propagation clustering

    Args:
      args (List[str]): command line parameters as list of strings
    """
    args = parse_clustering_args(args)
    setup_logging(args.loglevel)

    _logger.debug("Starting Affinity Propagation clustering...")

    try:
        # Load data
        if args.inline_data:
            data = parse_inline_data(args.inline_data)
        else:
            data = load_data(args.input_file)

        # Validate damping parameter
        if not 0.5 <= args.damping <= 1.0:
            _logger.error("Damping must be between 0.5 and 1.0")
            sys.exit(1)

        # Run clustering
        labels, centers, n_clusters = run_clustering(
            data,
            damping=args.damping,
            max_iter=args.max_iter,
            convergence_iter=args.convergence_iter,
            preference=args.preference,
            affinity=args.affinity
        )

        # Save results
        save_results(labels, args.output_file)

        # Print summary
        print(f"\n{'='*50}")
        print(f"Affinity Propagation Clustering Results")
        print(f"{'='*50}")
        print(f"Number of clusters found: {n_clusters}")
        print(f"Data points: {len(labels)}")
        print(f"Output saved to: {args.output_file}")
        print(f"{'='*50}\n")

        _logger.info("Clustering completed successfully")

    except Exception as e:
        _logger.error(f"Error during clustering: {e}")
        sys.exit(1)


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


def run_clustering_cli():
    """Calls :func:`main_clustering` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main_clustering(sys.argv[1:])


if __name__ == "__main__":
    # ^  This is a guard statement that will prevent the following code from
    #    being executed in the case someone imports this file instead of
    #    executing it as a script.
    #    https://docs.python.org/3/library/__main__.html

    # After installing your project with pip, users can also run your Python
    # modules as scripts via the ``-m`` flag, as defined in PEP 338::
    #
    #     python -m affinitypropagation.skeleton 42
    #
    run()