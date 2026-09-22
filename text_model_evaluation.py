import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import warnings
warnings.filterwarnings("ignore")

import json
import logging
import argparse

import numpy as np
from openpyxl import Workbook

from text_evaluation import text_similarity

DATA_DIR = './data'
CHECK_DIR = './check'
TEXT_MODEL_THRESHOLD = 0.65


def core_name(filename):
    return os.path.splitext(filename)[0]


def get_task_names(model):
    return [task['name'] for task in model.get('tasks', []) if task.get('name')]


def text_model_similarity(sentences, tasks, threshold):
    """Each task picks its best-matching sentence; the task is matched if that score is at or above the
    threshold, and the picked sentence counts as a matched sentence (once, however many tasks pick it).
    Returns (recall, precision) = (matched sentences / sentences, matched tasks / tasks); precision is
    undefined without tasks."""
    if not tasks:
        return 0.0, 'N/A'

    text_similarity.warm_cache(sentences + tasks)
    sims = np.array([[text_similarity.sts_bert(sentence, task) for task in tasks] for sentence in sentences])

    task_scores = sims.max(axis=0)
    matched_sentences = len(np.unique(sims.argmax(axis=0)[task_scores >= threshold]))
    matched_tasks = int((task_scores >= threshold).sum())

    return matched_sentences / len(sentences), matched_tasks / len(tasks)


def text_model_evaluation(dataset, threshold):
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger("TextModelLogger")
    logger.setLevel(logging.INFO)

    datasets = sorted(d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)))
    if dataset not in datasets:
        logger.error(f"Dataset '{dataset}' not found in {DATA_DIR}. Available: {', '.join(datasets)}")
        return

    ground_truth_dir = os.path.join(DATA_DIR, dataset, 'ground_truth')
    descriptions_dir = os.path.join(DATA_DIR, dataset, 'process_descriptions')
    for path in (ground_truth_dir, descriptions_dir):
        if not os.path.isdir(path):
            logger.error(f"Missing folder {path}")
            return

    description_files = sorted(f for f in os.listdir(descriptions_dir) if f.endswith('.txt'))

    workbook = Workbook()
    sheet = workbook.active
    sheet.append(['file', 'recall', 'precision'])

    logger.info(f'Evaluating {len(description_files)} descriptions of dataset {dataset}')

    for i, description_file in enumerate(description_files):
        logger.info(f'Evaluating file: {description_file}')
        print('--------------------------{}/{}-----------------------------------'.format(i, len(description_files)))
        try:
            with open(os.path.join(descriptions_dir, description_file), 'r') as infile:
                sentences = text_similarity.split_and_clean(infile.read())
            if not sentences:
                raise ValueError('description contains no sentences')

            with open(os.path.join(ground_truth_dir, core_name(description_file) + '.json'), 'r') as infile:
                tasks = get_task_names(json.load(infile))

            recall, precision = text_model_similarity(sentences, tasks, threshold)
            sheet.append([description_file, recall, precision])
        except Exception as e:
            logger.error(f"An error occurred while evaluating file {description_file}: {e}")

    os.makedirs(CHECK_DIR, exist_ok=True)
    output_path = os.path.join(CHECK_DIR, 'text_model_{}.xlsx'.format(dataset))
    workbook.save(output_path)
    logger.info(f'Saved {output_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check how well the tasks of the ground-truth process models match the sentences of their process descriptions.")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset folder inside ./data (e.g. domain, pet, realset, sapsam, mad)')
    parser.add_argument('--threshold', type=float, default=TEXT_MODEL_THRESHOLD, help='Similarity at or above which a task counts as matched')

    args = parser.parse_args()
    text_model_evaluation(args.dataset, args.threshold)
