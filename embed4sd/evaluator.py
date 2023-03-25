import os
import csv
import re

from embed4sd.extractors import FineTuningDataExtractor, IndicatorTestDataExtractor, GoalTestDataExtractor
from embed4sd.encoders import Encoder, CustomEncoderOne, CustomEncoderTwo, CustomEncoderThree
from embed4sd.tasks import Task, IndicatorToMultipleGoalsClassificationTask, GoalToMultipleGoalsClassificationTask, \
    MultiClassClassificationTask


class EncoderFactory:
    ENCODER_DIMENSION = {
        'USE-TRANSFORMER': 512, 'USE-MULTILINGUAL-CONVOLUTION': 512, 'USE-MULTILINGUAL-TRANSFORMER': 512,
        'ST5-BASE': 768, 'ST5-LARGE': 768,
        'SBERT-BERT-BASE': 768,
        'SBERT-MINILM-L6': 384, 'SBERT-MINILM-L12': 384,
        'SBERT-DISTILROBERTA': 768, 'SBERT-MPNET-BASE': 768,
        'SIMCSE-UNSUP-BERT-BASE': 768, 'SIMCSE-SUP-BERT-BASE': 768
    }

    @staticmethod
    def get_encoder(encoder: str) -> [Encoder, int]:
        if encoder in ['USE-TRANSFORMER', 'USE-MULTILINGUAL-CONVOLUTION', 'USE-MULTILINGUAL-TRANSFORMER',
                       'ST5-BASE', 'ST5-LARGE']:
            return CustomEncoderOne(encoder), EncoderFactory.ENCODER_DIMENSION[encoder]
        elif encoder in ['SBERT-BERT-BASE', 'SBERT-MINILM-L6', 'SBERT-MINILM-L12',
                         'SBERT-DISTILROBERTA', 'SBERT-MPNET-BASE']:
            return CustomEncoderTwo(encoder), EncoderFactory.ENCODER_DIMENSION[encoder]
        elif encoder in ['SIMCSE-UNSUP-BERT-BASE', 'SIMCSE-SUP-BERT-BASE']:
            return CustomEncoderThree(encoder), EncoderFactory.ENCODER_DIMENSION[encoder]
        else:
            raise ValueError(f'Invalid encoder name: {encoder}.')


class TestTaskFactory:
    @staticmethod
    def get_task(test_task: str) -> Task:
        if test_task == 'MC-IND-SDG':
            return MultiClassClassificationTask(test_task)
        elif test_task == 'MC-IND-TRG':
            return MultiClassClassificationTask(test_task)
        elif test_task == 'ML-IND-SDG':
            return IndicatorToMultipleGoalsClassificationTask(test_task)
        elif test_task == 'ML-SDG-SDG':
            return GoalToMultipleGoalsClassificationTask(test_task)
        else:
            raise ValueError(f'Invalid task name: {test_task}.')


class Utilities:
    @staticmethod
    def write(save_file: str, append: bool, data: list, header: bool = True):
        if not header:
            data = data[1:]

        if append:
            with open(save_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for row in data:
                    writer.writerow(row)
        else:
            with open(save_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for row in data:
                    writer.writerow(row)

    @staticmethod
    def extract_goal(indicator):
        goal_pattern = r'([0-9]{1,2})([.][0-9a-z]{1,2}[.][0-9]{1,2})'
        result = re.search(goal_pattern, indicator, re.IGNORECASE)

        if result.lastindex < 2:
            raise Exception(f'Indicator {indicator} does not have the expected format "x.y.z".')
        else:
            return result.group(1)

    @staticmethod
    def extract_target(indicator):
        target_pattern = r'([0-9]{1,2}[.][0-9a-z]{1,2})([.][0-9]{1,2})'
        result = re.search(target_pattern, indicator, re.IGNORECASE)

        if result.lastindex < 2:
            raise Exception(f'Indicator {indicator} does not have the expected format "x.y.z".')
        else:
            return result.group(1)


class Evaluator:

    @staticmethod
    def _multiple_labels(repeats, test_task):
        repeats = repeats.split(',')
        if (len(repeats) == 1) and (repeats[0] == ''):
            return ''

        if test_task in ['MC-IND-SDG', 'ML-IND-SDG']:
            labels = [Utilities.extract_goal(r) for r in repeats]
        elif test_task == 'MC-IND-TRG':
            labels = [Utilities.extract_target(r) for r in repeats]
        elif test_task == 'ML-SDG-SDG':
            labels = repeats
        else:
            raise Exception(f'Unknown task: {test_task}.')

        if labels[0] == labels[-1]:
            return ''
        else:
            return labels

    @staticmethod
    def _embed(encoder, base_model_directory, checkpoint_directory, checkpoint_number, data):
        enc, dim = EncoderFactory.get_encoder(encoder)
        return enc.run(base_model_directory=base_model_directory,
                       embeddings_dimension=dim,
                       checkpoint_directory=checkpoint_directory,
                       checkpoint_number=checkpoint_number, data=data)

    @staticmethod
    def _validate(task, k):
        if task in ['MC-IND-SDG', 'ML-IND-SDG', 'ML-SDG-SDG']:
            if k not in [(14, 6), (14, 17), (22, 6), (22, 17)]:
                raise ValueError(f'Invalid number of kNN training examples by SDG {k} for test task {task}.')
        elif task in ['MC-IND-TRG']:
            if k not in [(0, 6), (0, 17)]:
                raise ValueError(f'Invalid number of kNN training examples by SDG {k} for test task {task}.')
        else:
            raise ValueError(f'Invalid task name: {task}.')

    @staticmethod
    def _evaluate(test_task: str,
                  training_embeddings, training_labels,
                  validation_embeddings, validation_labels, validation_labels_multiple,
                  test_embeddings, test_labels, test_labels_multiple,
                  encoder, checkpoint_number, random_seed):
        task = TestTaskFactory.get_task(test_task)
        return task.run(training_embeddings, training_labels,
                        validation_embeddings, validation_labels, validation_labels_multiple,
                        test_embeddings, test_labels, test_labels_multiple,
                        encoder, checkpoint_number, random_seed)

    def run(self, input_files, flags, test_input_file,
            encoder, base_model_directory, checkpoint_directory, checkpoint_number,
            test_task, k, random_seed, result_directory):
        self._validate(test_task, k)
        (g, t) = k

        train_label_column = 'goal' if test_task.endswith('SDG') else 'target'

        if test_task.startswith('ML'):
            test_label_column = 'repeats' if 'IND' in test_task else 'links'
        else:
            test_label_column = 'goal' if test_task.endswith('SDG') else 'target'

        training_extractor = FineTuningDataExtractor(input_files=input_files, flags=flags)

        if test_task != 'ML-SDG-SDG':
            test_extractor = IndicatorTestDataExtractor(input_file=test_input_file)
        else:
            test_extractor = GoalTestDataExtractor(input_file=test_input_file)
        validation_set, test_set = test_extractor.run()

        training_set = training_extractor.run(goal_count=g, target_count=t)
        training_embeddings = self._embed(encoder, base_model_directory,
                                          checkpoint_directory, checkpoint_number,
                                          training_set['modified_text_excerpt'].values.tolist())
        training_labels_ = training_set[training_set[train_label_column] != ''][train_label_column].values

        test_embeddings = self._embed(encoder, base_model_directory,
                                      checkpoint_directory, checkpoint_number,
                                      test_set['text'].values.tolist())

        if test_task.startswith('MC'):
            validation_embeddings = self._embed(encoder, base_model_directory,
                                                checkpoint_directory, checkpoint_number,
                                                validation_set['text'].values.tolist())
            validation_embeddings_ = validation_embeddings[validation_set[test_label_column] != '']
            validation_labels_ = validation_set[validation_set[test_label_column] != ''][test_label_column].values
            validation_labels_multiple_ = validation_set[validation_set[test_label_column] != '']['repeats'].apply(
                lambda x: self._multiple_labels(x, test_task))
            test_labels_multiple_ = test_set[test_set[test_label_column] != '']['repeats'].apply(
                lambda x: self._multiple_labels(x, test_task))

        else:
            validation_embeddings_, validation_labels_ = None, None
            validation_labels_multiple_, test_labels_multiple_ = None, None
            test_set[test_label_column] = test_set[test_label_column].apply(lambda x:
                                                                            self._multiple_labels(x, test_task))

        test_embeddings_ = test_embeddings[test_set[test_label_column] != '']
        test_labels_ = test_set[test_set[test_label_column] != ''][test_label_column].values

        results = self._evaluate(test_task,
                                 training_embeddings, training_labels_,
                                 validation_embeddings_, validation_labels_, validation_labels_multiple_,
                                 test_embeddings_, test_labels_, test_labels_multiple_,
                                 encoder, checkpoint_number, random_seed)

        Utilities.write(os.path.join(result_directory, f'{test_task}.csv'), append=True, data=results)


if __name__ == '__main__':
    # example
    evaluator = Evaluator()
    evaluator.run(
        input_files=[
            r'path\to\xml\file',  # file containing the revision of the general Wikipedia article
            r'path\to\xml\file'],  # file containing the revision of the SDG-specific Wikipedia articles
        flags=[True,  # flag indicating that the first file contains a revision of the general Wikipedia article
               False],  # flag indicating that the second file contains revisions of SDG-specific Wikipedia articles
        test_input_file=r'path\to\indicator\or\goal\csv\file',  # CSV file containing the necessary indicator/Goal data
        encoder='SBERT-MINILM-L6',  # encoder name, see EncoderFactory.ENCODER_DIMENSION dictionary
        base_model_directory=r'path\to\model\files',  # directory containing the pre-trained model files
        checkpoint_directory=r'path\to\checkpoint\file',  # directory containing the checkpoint file (for fine-tuned models)
        checkpoint_number=5,  # checkpoint number
        test_task='MC-IND-SDG',  # test task name, see class TestTaskFactory
        k=(14, 6),  # number of examples to be sampled by Goal and target in the kNN training set
        random_seed=42,  # random seed
        result_directory=r'path\to\output\directory')  # directory where the output CSV file should be saved
