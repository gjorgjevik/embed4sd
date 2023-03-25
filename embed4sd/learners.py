import os

from numpy import array
from numpy.random import shuffle
from sentence_transformers import losses
from abc import ABC, abstractmethod

from torch import load, tensor, sum, clamp, long, save
from torch.nn.functional import normalize
from torch.optim import Adam
from torch.utils.data import IterableDataset, DataLoader
from transformers import set_seed, AutoTokenizer, AutoModel

from embed4sd.extractors import FineTuningDataExtractor

RANDOM_SEED = 2355764148  # set the random seed to an appropriate value
set_seed(RANDOM_SEED)


class CustomIterableDataset(IterableDataset):

    def __init__(self, x_train, ids_train, y_train, start_indexes, end_indexes, iterations,
                 start, end):
        super(CustomIterableDataset).__init__()

        self.x_train = x_train
        self.ids_train = ids_train
        self.y_train = y_train
        self.start_indexes = start_indexes
        self.end_indexes = end_indexes
        self.iterations = iterations
        self.start = start
        self.end = end

    def __iter__(self):
        set_seed(RANDOM_SEED)
        class_indexes = array(range(17))

        for iteration in range(self.iterations):
            if iteration >= self.end:
                print(f'Skipped iteration {iteration}.')
                continue
            shuffle(class_indexes)  # shuffle class indexes
            classes = class_indexes[:13]  # first 13 classes will have 4 examples, the rest 3

            idx = []
            for c in range(17):
                goal_indexes = list(range(self.start_indexes[c], self.end_indexes[c]))
                shuffle(goal_indexes)

                if c in classes:
                    count = 4
                else:
                    count = 3
                idx = idx + goal_indexes[:count]

            if iteration < self.start:
                print(f'Skipped iteration {iteration}.')
                continue

            print(f'Processing iteration {iteration}.')
            x_ = [x for i, x in enumerate(self.x_train) if i in idx]
            y_ = [y for i, y in enumerate(self.y_train) if i in idx]
            ids_ = [id_ for i, id_ in enumerate(self.ids_train) if i in idx]

            yield x_, ids_, y_


class RepresentationLearner(ABC):
    """
    Abstract class extended by all representation learners.
    """

    BATCH_SIZE = 64
    LEARNING_RATE = 2e-5

    def __init__(self, input_files: list, flags: list, margin: float):
        self.margin = margin
        self.training_data_extractor = FineTuningDataExtractor(input_files=input_files, flags=flags)

    @abstractmethod
    def load_data(self, k: int):
        raise NotImplementedError()

    def train_network(self, k, base_model_dir, output_dir, iterations, start_iteration, end_iteration):
        [x_train, ids_train, y_train, start_indexes, end_indexes] = self.load_data(k)

        train_ds = CustomIterableDataset(x_train=x_train,
                                         ids_train=ids_train,
                                         y_train=y_train,
                                         start_indexes=start_indexes,
                                         end_indexes=end_indexes,
                                         iterations=iterations,
                                         start=start_iteration,
                                         end=end_iteration)
        train_loader = DataLoader(train_ds)

        tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
        model = AutoModel.from_pretrained(base_model_dir)

        optimizer = Adam(params=model.parameters(), lr=self.LEARNING_RATE)
        loss_fn = losses.BatchHardTripletLoss(
            model=model,
            margin=self.margin,
            distance_metric=losses.BatchHardTripletLossDistanceFunction.cosine_distance)

        path = os.path.join(output_dir, f'{str(start_iteration)}_model.pt')
        if os.path.exists(path):
            checkpoint = load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            checkpoint = None

        if checkpoint:
            print(f'Restored from {path}')
        else:
            print('Initializing from scratch.')

        for i, (x, ids, y) in enumerate(train_loader):
            print(f'Iteration: {i}')

            model.train()
            labels = tensor(y, dtype=long)
            
            # implementation based on the examples from https://github.com/UKPLab/sentence-transformers
            text = tokenizer([x_[0] for x_ in x], return_tensors="pt", max_length=128, truncation=True,
                             padding="max_length")
            output = model(**text)
            input_mask_expanded = text['attention_mask'].unsqueeze(-1).expand(output[0].size()).float()
            embeddings = sum(output[0] * input_mask_expanded, 1) / clamp(
                input_mask_expanded.sum(1), min=1e-9)
            embeddings = normalize(embeddings, p=2, dim=1)

            loss = loss_fn.batch_hard_triplet_loss(labels=labels, embeddings=embeddings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i > 0) and ((i + 1) % 5) == 0:
                model.eval()
                save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(output_dir, f'{str(start_iteration + i + 1)}_model.pt'))
                print(f'Saved checkpoint for step {str(start_iteration + i + 1)}: {output_dir}')


class GoalRepresentationLearner(RepresentationLearner):

    def __init__(self, input_files: list, flags: list, margin: float = 0.4):
        super(GoalRepresentationLearner, self).__init__(input_files, flags, margin)

    def load_data(self, k):
        goal_dict = dict()
        start_indexes = []
        end_indexes = []

        data = self.training_data_extractor.run(goal_count=k, target_count=0)
        data = data.fillna('').sort_values(by=['goal', 'target'], ascending=True)

        current_start = 0
        for label in range(1, 18):
            goal_dict[label] = label - 1
            start_indexes += [current_start]
            end_indexes += [current_start + data[data['goal'] == label].shape[0]]
            current_start += data[data['goal'] == label].shape[0]

        x_train = data['modified_text_excerpt'].values.tolist()
        y_train = [goal_dict[label] for label in data['goal'].values]
        ids_train = data['id'].values.tolist()

        return [x_train, ids_train, y_train, start_indexes, end_indexes]


class TargetRepresentationLearner(RepresentationLearner):

    def __init__(self, input_files: list, flags: list, margin: float = 0.2):
        super(TargetRepresentationLearner, self).__init__(input_files, flags, margin)

    def load_data(self, k: int):
        target_dict = dict()
        start_indexes = []
        end_indexes = []

        data = self.training_data_extractor.run(goal_count=0, target_count=k)
        data = data.fillna('').sort_values(by=['goal', 'target'], ascending=True)

        current_start = 0
        for label in range(1, 18):
            start_indexes += [current_start]
            current_start += data[data['goal'] == label].shape[0]
            end_indexes += [current_start]

        counter = 0
        for label in data['target'].values:
            if label not in target_dict.keys():
                target_dict[label] = counter
                counter += 1

        x_train = data['modified_text_excerpt'].values.tolist()
        y_train = [target_dict[label] for label in data['target'].values]
        ids_train = data['id'].values.tolist()

        return [x_train, ids_train, y_train, start_indexes, end_indexes]


if __name__ == '__main__':
    # examples
    learner = GoalRepresentationLearner(
        input_files=[
            r'path\to\xml\file',  # file containing the revision of the general Wikipedia article
            r'path\to\xml\file'],  # file containing the revisions of the SDG-specific Wikipedia articles
        flags=[True,  # flag indicating that the first file contains a revision of the general Wikipedia article
               False])  # flag indicating that the second file contains revisions of the SDG-specific Wikipedia articles
    learner.train_network(k=14,  # number of examples to be sampled by SDG in the fine-tuning set
                          base_model_dir=r'path\to\model\files',  # directory containing the pre-trained model files
                          output_dir=r'path\to\last\checkpoint\file',  # directory containing the last checkpoint
                          iterations=20,  # total number of fine-tuning iterations
                          start_iteration=10,  # start iteration, can be larger than 0 if the fine-tuning is resumed
                          end_iteration=20)  # end iteration

    learner = TargetRepresentationLearner(
        input_files=[
            r'path\to\xml\file',  # file containing the revision of the general Wikipedia article
            r'path\to\xml\file'],  # file containing the revisions of the SDG-specific Wikipedia articles
        flags=[True,  # flag indicating that the first file contains a revision of the general Wikipedia article
               False])  # flag indicating that the second file contains revisions of the SDG-specific Wikipedia articles
    learner.train_network(k=17,  # number of examples to be sampled by target in the fine-tuning set
                          base_model_dir=r'path\to\model\files',  # directory containing the pre-trained model files
                          output_dir=r'path\to\last\checkpoint\file',  # directory containing the last checkpoint
                          iterations=20,  # total number of fine-tuning iterations
                          start_iteration=0,  # start iteration, can be larger than 0 if the fine-tuning is resumed
                          end_iteration=10)  # end iteration
