from numpy import sum
from abc import ABC, abstractmethod

from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, ndcg_score
from transformers import set_seed


class Task(ABC):
    """
    Abstract class extended by all validation and test tasks.
    """

    def __init__(self, name: str):
        self.name = name

        self.label_dict = dict()
        self.label_dict_inv = dict()

        self.MultiLabelBinarizer = MultiLabelBinarizer()

    @staticmethod
    def angular_distance(a, b):
        d = 1.0 - sum(a * b)
        return max(d, 0.0)

    @staticmethod
    def get_label_dict(labels: list) -> [dict, dict]:
        label_dict = dict()
        label_dict_inv = dict()
        counter = 0
        for label in labels:
            if label != '' and label not in label_dict.keys():
                label_dict[label] = counter
                label_dict_inv[counter] = label
                counter += 1

        return label_dict, label_dict_inv

    @abstractmethod
    def label_training(self, args: dict) -> list:
        raise NotImplementedError()

    @abstractmethod
    def label_test(self, args: dict) -> list:
        raise NotImplementedError()

    def encode_labels(self, embeddings: list, labels: list, training: bool) -> object:
        if training:
            self.label_dict, self.label_dict_inv = self.get_label_dict(labels)

        y = self.label_training({'labels': labels}) if training else self.label_test({'labels': labels})

        return embeddings, y

    @abstractmethod
    def evaluate(self, k: int,
                 train_y: list, train_emb: list,
                 validation_y: list, validation_y_multiple: list, validation_emb: list,
                 test_y: list, test_y_multiple: list, test_emb: list) -> list:
        raise NotImplementedError()

    def run(self, train_emb: list, train_y: list,
            validation_emb: list, validation_y: list, validation_y_multiple: list,
            test_emb: list, test_y: list, test_y_multiple: list,
            encoder: str, checkpoint_number: str, random_seed: int) -> list:
        set_seed(random_seed)

        train_emb, train_y = self.encode_labels(train_emb, train_y, True)
        if validation_emb is not None and validation_y is not None:
            validation_emb, validation_y = self.encode_labels(validation_emb, validation_y, False)
        test_emb, test_y = self.encode_labels(test_emb, test_y, False)

        if self.name in ['MC-IND-SDG', 'MC-IND-TRG']:
            ks = [1, 3]
        elif self.name == 'ML-IND-SDG':
            ks = [20, 40]
        elif self.name == 'ML-SDG-SDG':
            ks = [16]
        else:
            raise ValueError(f'Unknown task name: {self.name}')

        results = []
        for k in ks:
            results.append(
                [k, encoder, random_seed, checkpoint_number, goal_count, target_count] + self.evaluate(
                    k, train_y, train_emb,
                    validation_y, validation_y_multiple, validation_emb,
                    test_y, test_y_multiple, test_emb))

        return results


class MultiClassClassificationTask(Task):

    def label_training(self, args: dict) -> list:
        labels = args['labels']
        return [self.label_dict[label] for label in labels]

    def label_test(self, args: dict) -> list:
        labels = args['labels']
        return [self.label_dict[label] for label in labels]

    @staticmethod
    def _correct_predictions(labels, labels_multiple, predictions, label_dict_inv):
        new_predictions = []
        for label, label_multiple, prediction in zip(labels, labels_multiple, predictions):
            prediction_ = label_dict_inv[prediction]
            if label_multiple != '' and prediction_ in label_multiple:
                new_predictions.append(label)
            else:
                new_predictions.append(prediction)
        return new_predictions

    def evaluate(self, k, train_y, train_emb,
                 validation_y, validation_y_multiple, validation_emb,
                 test_y, test_y_multiple, test_emb):
        classifier = KNeighborsClassifier(metric=self.angular_distance, n_neighbors=k, weights='distance')
        classifier.fit(train_emb, train_y)
        validation_predictions_incorrect = classifier.predict(validation_emb)
        test_predictions_incorrect = classifier.predict(test_emb)

        validation_predictions = self._correct_predictions(validation_y,
                                                           validation_y_multiple,
                                                           validation_predictions_incorrect,
                                                           self.label_dict_inv)
        validation = accuracy_score(validation_y, validation_predictions)

        test_predictions = self._correct_predictions(test_y,
                                                     test_y_multiple,
                                                     test_predictions_incorrect,
                                                     self.label_dict_inv)
        test = accuracy_score(test_y, test_predictions)

        return [validation, test]


class MultiLabelClassificationTask(Task, ABC):

    def label_training(self, args):
        labels = args['labels']
        self.MultiLabelBinarizer.fit_transform(
            [[self.label_dict[label]] for label in labels])
        return [self.label_dict[label] for label in labels]

    def label_test(self, args):
        label_sets = args['labels']

        label_sets_transformed = []
        for labels in label_sets:
            label_sets_transformed.append([self.label_dict[label] for label in labels])

        label_sets_binarized = self.MultiLabelBinarizer.transform(label_sets_transformed)

        return label_sets_binarized

    @staticmethod
    def normalized_discounted_cumulative_gain(labels, predictions, ndcg_k):
        return ndcg_score(labels, predictions, k=ndcg_k)


class IndicatorToMultipleGoalsClassificationTask(MultiLabelClassificationTask):
    def evaluate(self, k, train_y, train_emb,
                 validation_y, validation_y_multiple, validation_emb,
                 test_y, test_y_multiple, test_emb):
        classifier = KNeighborsClassifier(metric=self.angular_distance, n_neighbors=k, weights='distance')
        classifier.fit(train_emb, train_y)

        test_predictions_probabilities = classifier.predict_proba(test_emb)
        test = self.normalized_discounted_cumulative_gain(test_y,
                                                          test_predictions_probabilities,
                                                          ndcg_k=5)

        return [test]


class GoalToMultipleGoalsClassificationTask(MultiLabelClassificationTask):
    def evaluate(self, k, train_y, train_emb,
                 validation_y, validation_y_multiple, validation_emb,
                 test_y, test_y_multiple, test_emb):
        nearest_centroid = NearestCentroid(metric=self.angular_distance)
        nearest_centroid.fit(train_emb, train_y)

        centroids = nearest_centroid.centroids_
        classes = nearest_centroid.classes_

        classifier = KNeighborsClassifier(metric=self.angular_distance, n_neighbors=k, weights='distance')
        classifier.fit(centroids, classes)

        test_predictions_probabilities = classifier.predict_proba(test_emb)
        test = self.normalized_discounted_cumulative_gain(test_y,
                                                          test_predictions_probabilities,
                                                          ndcg_k=5)

        return [test]
