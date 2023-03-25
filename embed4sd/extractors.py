import re
from abc import ABC, abstractmethod

from numpy import inf
from numpy.random import seed
from pandas import DataFrame, concat, read_csv, cut

from sklearn.model_selection import StratifiedShuffleSplit

from embed4sd.processors import Parser, TextSplitter, SentenceFilter, TextCleaner


class DataExtractor:
    min_length = 20
    max_length = 40
    max_std = 15

    filter_min = True
    filter_max = True

    def __init__(self, input_files: list, flags: list):
        self.input_files = input_files
        self.flags = flags

    def run(self):
        # parse
        wiki_parser = Parser()
        goals = wiki_parser.run(files=self.input_files,
                                flags=self.flags)

        # clean
        cleaner = TextCleaner()
        goals['text'] = goals[['text', 'section']].apply(
            lambda x:
            cleaner.common_phrases_masking(
                cleaner.list_masking(
                    cleaner.numbers_masking(
                        cleaner.generic_indicator_masking(x[0]), x[1]
                    )
                )
            ), axis=1)

        # split
        splitter = TextSplitter(min_length=self.min_length, max_length=self.max_length, max_std=self.max_std)

        data_paragraphs = dict()
        for g, t, s, url, title, revision_id, revision_timestamp, text in zip(
                goals['goal'].values,
                goals['target'].values,
                goals['section'].values,
                goals['source_url'].values,
                goals['title'].values,
                goals['revision_id'].values,
                goals['revision_timestamp'].values,
                goals['text'].values):
            paragraphs = []
            c = str(g) + '~' + str(t) + '~' + str(
                s.replace(',', '')) + '~' + str(url) + '~' + str(title) + '~' + str(
                revision_id) + '~' + str(revision_timestamp)
            if c in data_paragraphs.keys():
                paragraphs = data_paragraphs[c]
            paragraphs.append(splitter.split_to_parts([text]))
            data_paragraphs[c] = paragraphs

        # filter common sentences
        filter_ = SentenceFilter()
        data_paragraphs = filter_.filter(data_paragraphs)

        # concatenate
        data = DataFrame(columns=['id', 'modified_text_excerpt', 'goal', 'target', 'section',
                                  'source_url', 'title',
                                  'revision_id', 'revision_timestamp'])

        counter = dict()
        for c, paragraphs in data_paragraphs.items():
            g = str(c.split('~')[0])
            t = str(c.split('~')[1])
            s = str(c.split('~')[2])
            url = str(c.split('~')[3])
            title = str(c.split('~')[4])
            revision_id = str(c.split('~')[5])
            revision_timestamp = str(c.split('~')[6])
            id_ = str(g) + '_' + str(t)

            if id_ in counter.keys():
                p_counter = counter[id_]
            else:
                p_counter = 0

            concatenated_lowercase = None
            for paragraph in paragraphs:
                [concatenated, concatenated_lowercase] = splitter.concatenate_parts(paragraph, concatenated_lowercase)

                for part in concatenated:
                    if t == '' or p_counter > 0:
                        len_ = len(part.split())
                        if self.filter_min and len_ < (self.min_length - self.max_std):
                            continue
                        if self.filter_max and len_ > (self.max_length + self.max_std):
                            continue
                    row = DataFrame({'id': str(id_) + '_' + str(p_counter),
                                     'modified_text_excerpt': part,
                                     'goal': g,
                                     'target': t,
                                     'section': s,
                                     'source_url': url,
                                     'title': title,
                                     'revision_id': revision_id,
                                     'revision_timestamp': revision_timestamp}, index=[0])
                    data = concat([data, row], ignore_index=True)
                    p_counter += 1
            counter[id_] = p_counter

        # filter records with target labels in nonstandard format (if any)
        data = data[data['target'].apply(lambda t_: (t_ == '') or
                                                    (re.fullmatch(r'[0-9]{1,2}[.][0-9a-z]{1,2}',
                                                                  t_, re.IGNORECASE) is not None))]
        # sort
        return data.sort_values(by=['goal', 'target'], ascending=True)


class FineTuningDataExtractor:
    GOAL_COLUMNS = ['general', 'lead section', 'background',
                    'progress', 'tools', 'challenge',
                    'covid-19 pandemic', 'society and culture',
                    'organizations', 'links with other sdgs']

    def __init__(self, input_files: list, flags: list):
        self.DataExtractor = DataExtractor(input_files=input_files, flags=flags)

    def extract(self, data: DataFrame, goal_count: int, target_count: int) -> DataFrame:
        sample = DataFrame(columns=['id', 'modified_text_excerpt', 'goal', 'target', 'section',
                                    'source_url', 'license', 'title',
                                    'revision_id', 'revision_timestamp'])

        if goal_count > 0:
            for goal in set(data['goal'].values):
                counter = 0
                for col in self.GOAL_COLUMNS:
                    if counter >= goal_count:
                        break
                    paragraphs = data[data[['goal', 'section']].apply(
                        lambda x: (x[0] == goal) & (col in x[1].lower()) & ('target' not in x[1].lower()), axis=1)]

                    count = min(goal_count - counter, paragraphs.shape[0])
                    sample = concat([sample, paragraphs.head(count)])
                    counter += count

        if target_count > 0:
            target_paragraphs_ = data[data['section'].apply(lambda x: 'target' in x.lower())]
            for target in set(target_paragraphs_['target'].values):
                _target_paragraphs_ = target_paragraphs_[target_paragraphs_['target'] == target]

                count = min(target_count, _target_paragraphs_.shape[0])
                sample = concat([sample, _target_paragraphs_.head(count)])
        return sample.drop_duplicates()

    def run(self, goal_count: int, target_count: int):
        data = self.DataExtractor.run()
        data = self.extract(data, goal_count, target_count)
        return data


class TestDataExtractor(ABC):
    RANDOM_SEED = 42

    def __init__(self, input_file: str):
        self.input_file = input_file
        seed(self.RANDOM_SEED)

    @staticmethod
    def _word_count(title: str) -> int:
        return len(title.split())

    @staticmethod
    def _extract_goal(indicator: str) -> str:
        goal_pattern = r'([0-9]{1,2})([.][0-9a-z]{1,2}[.][0-9]{1,2})'
        result = re.search(goal_pattern, indicator, re.IGNORECASE)

        if result.lastindex < 2:
            raise Exception(f'Indicator {indicator} does not have the expected format "x.y.z".')
        else:
            return str(result.group(1))

    @staticmethod
    def _extract_target(indicator: str) -> str:
        target_pattern = r'([0-9]{1,2}[.][0-9a-z]{1,2})([.][0-9]{1,2})'
        result = re.search(target_pattern, indicator, re.IGNORECASE)

        if result.lastindex < 2:
            raise Exception(f'Indicator {indicator} does not have the expected format "x.y.z".')
        else:
            return str(result.group(1))

    @abstractmethod
    def prepare(self) -> DataFrame:
        raise NotImplementedError()

    @abstractmethod
    def extract(self, data: DataFrame) -> [DataFrame, DataFrame]:
        raise NotImplementedError()

    def run(self) -> DataFrame:
        data = self.prepare()
        data = self.extract(data)
        return data


class IndicatorTestDataExtractor(TestDataExtractor):
    """
    The indicators should be ordered by Goal, target and id in ascending order.
    Within a Goal, the targets ending with a digit should come before the targets ending with a letter.
    For example, the ordering of Goal 7 is the following: 7.1.1, 7.1.2, 7.2.1, 7.3.1, 7.a.1, 7.b.1.
    """
    VALIDATION_SIZE = 0.25

    INDICATOR_FILE_COLUMNS = ['id', 'repeats', 'text']  # column 'text' refers to the indicator title
    INDICATOR_OUTPUT_COLUMNS = ['id', 'repeats', 'goal', 'target', 'text']

    def prepare(self) -> DataFrame:
        data = read_csv(self.input_file, sep=',', encoding='utf-8', dtype=str).fillna('')
        assert data.columns.tolist().__eq__(self.INDICATOR_FILE_COLUMNS)

        data = data[data['text'] != '']
        data['goal'] = data['id'].apply(lambda x: self._extract_goal(x))
        data['target'] = data['id'].apply(lambda x: self._extract_target(x))

        data['word_count'] = data['text'].apply(lambda x: self._word_count(x))
        data['word_cat_3'] = cut(data['word_count'],
                                 bins=[0, 10, 20, inf],
                                 labels=[1, 2, 3])
        data['word_cat_2'] = cut(data['word_count'],
                                 bins=[0, 15, inf],
                                 labels=[1, 2])
        return data

    def extract(self, data: DataFrame) -> [DataFrame, DataFrame]:
        goals = data['goal'].drop_duplicates().tolist()
        split = StratifiedShuffleSplit(n_splits=1, test_size=self.VALIDATION_SIZE, random_state=self.RANDOM_SEED)

        validation_indices_all = []
        test_indices_all = []
        for g in goals:
            g_data = data[data['goal'] == g]
            try:
                for test_indices, validation_indices in split.split(g_data, g_data[['goal', 'word_cat_3']]):
                    validation_indices_all += g_data.iloc[validation_indices].index.tolist()
                    test_indices_all += g_data.iloc[test_indices].index.tolist()
            except BaseException as e:
                try:
                    for test_indices, validation_indices in split.split(g_data, g_data[['goal', 'word_cat_2']]):
                        validation_indices_all += g_data.iloc[validation_indices].index.tolist()
                        test_indices_all += g_data.iloc[test_indices].index.tolist()
                except BaseException as e:
                    for test_indices, validation_indices in split.split(g_data, g_data[['goal']]):
                        validation_indices_all += g_data.iloc[validation_indices].index.tolist()
                        test_indices_all += g_data.iloc[test_indices].index.tolist()

        assert len(validation_indices_all) == len([x for x in validation_indices_all if x not in test_indices_all])
        assert len(test_indices_all) == len([x for x in test_indices_all if x not in validation_indices_all])

        validation_set = data.loc[validation_indices_all][self.INDICATOR_OUTPUT_COLUMNS]
        test_set = data.loc[test_indices_all][self.INDICATOR_OUTPUT_COLUMNS]

        return validation_set, test_set


class GoalTestDataExtractor(TestDataExtractor):
    GOAL_FILE_COLUMNS = ['id', 'links', 'text']  # column 'text' refers to the Goal title
    GOAL_OUTPUT_COLUMNS = ['id', 'links', 'goal', 'text']

    def prepare(self):
        data = read_csv(self.input_file, sep=',', encoding='utf-8', dtype=str).fillna('')
        assert data.columns.tolist().__eq__(self.GOAL_FILE_COLUMNS)
        assert data.shape[0] == 16

        data['goal'] = data['id']
        return data

    def extract(self, data):
        test_set = data[self.GOAL_OUTPUT_COLUMNS]
        return None, test_set
