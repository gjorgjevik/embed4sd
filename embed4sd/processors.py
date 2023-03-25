import re
import xml.etree.ElementTree as ElementTree

from numpy import ndarray, abs, array, concatenate, float32
from pandas import DataFrame, concat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN


class Parser:
    ns = {'wiki': 'http://www.mediawiki.org/xml/export-0.10/'}
    source_url = 'https://en.wikipedia.org/wiki/'

    sections = ['background', 'challenge', 'society and culture',
                'overall progress', '= progress =', 'tools',
                'covid-19 pandemic', 'links with other sdgs',
                'organizations']

    @staticmethod
    def _remove_link_markup(text: str) -> str:
        old_text = ''
        new_text = text
        while old_text != new_text:
            old_text = new_text
            new_text = re.sub(r'\[\[(?!:?File:|Image:)([^\[\]]+)\|([^\[\]]+)]]', '\\2', new_text)
            new_text = re.sub(r'\[\[(?!:?File:|Image:)([^\[\]]+)]]', '\\1', new_text)
        text = new_text

        return re.sub(r'\s{2,}', ' ', text)

    @staticmethod
    def _remove_wiki_markup(text: str) -> str:
        text = re.sub(r'<br\s?/>', '', text)

        text = re.sub(r'<sub[^>]*>[^<>]+</sub>', '', text)
        text = re.sub(r'<sub[^<>]+/>', '', text)

        text = re.sub(r'<ref[^>]*>[^<>]+</ref>', '', text)
        text = re.sub(r'<ref[^<>]*/>', '', text)
        text = re.sub(r'<ref[^<>]*>', '', text)

        text = re.sub(r'<[^>]+>([^<>]+)</[^>]+>', '\\1', text)
        text = re.sub(r'<!--[^<>]+-->\.?', '', text)
        text = re.sub(r'&nbsp;', ' ', text)

        old_text = text
        new_text = re.sub(r'{{[^{}]+}}', '', text)
        while new_text != old_text:
            old_text = new_text
            new_text = re.sub(r'{{[^{}]+}}', '', old_text)
        text = new_text

        text = re.sub(r'\[\[(File:|Image:)[^\[\]]+(\[\[[^\[\]]+]]([^\[\]]+)?)*]]', '', text)
        text = re.sub(r'\[\[(File:|Image:)[^\[\]]+(\[[^\[\]]+]([^\[\]]+)?)*]]', '', text)

        text = re.sub(r'\[https?://\S+\s(.+)]', '\\1', text)
        text = re.sub(r'\[https?://\S+]', '', text)

        text = re.sub(r'\'{2,3}', '', text)

        return text

    def _append(self, line_: str, goal: str, target: str, section: str, page_title: str,
                page_revision_id: str, page_revision_timestamp: str, goals: DataFrame) -> DataFrame:
        row = DataFrame({
            'goal': goal,
            'target': target.lower(),
            'section': section,
            'source_url': self.source_url + page_title.replace(' ', '_'),
            'title': page_title,
            'revision_id': page_revision_id,
            'revision_timestamp': page_revision_timestamp,
            'text': line_.strip()
        }, index=[0])
        goals = concat([goals, row], ignore_index=True)

        return goals

    def _parse_lead_section(self, goal: str, page_title: str, page_revision_id: str,
                            page_revision_timestamp: str, lines: list, goals: DataFrame, label: str) -> DataFrame:
        line_ = ''
        in_list = False
        end = False
        for line in lines:
            line = line.strip()
            if line:
                if line.startswith('=') and not in_list:
                    break
                elif line.startswith('='):
                    end = True

                if not in_list and line.endswith(':'):
                    in_list = True
                    line_ = line
                elif in_list and line.startswith('*'):
                    line_ += line.replace('*', '') + ';; '
                elif in_list and not line.startswith('*'):
                    in_list = False
                    goals = self._append(line_, goal, '', label, page_title,
                                         page_revision_id, page_revision_timestamp, goals)
                    line_ = line
                    if line.endswith(':'):
                        in_list = True
                else:
                    line_ = line

                if not in_list:
                    goals = self._append(line_, goal, '', label, page_title,
                                         page_revision_id, page_revision_timestamp, goals)
                    line_ = ''
                if end:
                    break

        return goals

    def _parse_section(self, goal: str, page_title: str, page_revision_id: str,
                       page_revision_timestamp: str, lines: list, goals: DataFrame, section: str) -> DataFrame:
        in_section = False
        in_list = False
        end = False
        section_ = ''
        line_ = ''

        for line in lines:
            line = line.strip()
            if line:
                if (line.startswith('== ') or line.startswith('==\t')) and section in line.lower():
                    in_section = True
                    section_ = line.replace('=', '').replace('\t', '').strip()
                elif in_section and re.match(r'^==[\s\tA-Z].+$', line) and section not in line.lower() and not in_list:
                    break
                elif in_section and re.match(r'^==[\s\tA-Z].+$', line) and section not in line.lower() and in_list:
                    end = True
                elif in_section and line.startswith('===') and 'custodian agencies' in line.lower() and not in_list:
                    break
                elif in_section and line.startswith('===') and 'custodian agencies' in line.lower() and in_list:
                    end = True
                elif in_section:
                    if line.startswith('=='):
                        continue
                    if not in_list and line.endswith(':'):
                        in_list = True
                        line_ = line
                    elif not in_list and line.endswith('*'):
                        in_list = True
                        line_ = line.replace('*', '') + ';; '
                    elif in_list and line.endswith('*'):
                        line_ += line.replace('*', '') + ';; '
                    elif in_list and not line.endswith('*'):
                        in_list = False
                        goals = self._append(line_, goal, '', section_, page_title,
                                             page_revision_id, page_revision_timestamp, goals)
                        line_ = line
                        if line.endswith(':'):
                            in_list = True
                    else:
                        line_ = line

                    if not in_list:
                        goals = self._append(line_, goal, '', section_, page_title,
                                             page_revision_id, page_revision_timestamp, goals)
                        line_ = ''
                    if end:
                        break

        return goals

    def _parse_target(self, goal: str, page_title: str, page_revision_id: str,
                      page_revision_timestamp: str, lines: list, goals: DataFrame) -> DataFrame:
        in_target = False
        in_list = False
        line_ = ''
        target = None

        for line in lines:
            line = line.strip()
            if line:
                if line.startswith('==='):
                    if in_list:
                        goals = self._append(line_, goal, target, f'Targets',
                                             page_title, page_revision_id, page_revision_timestamp, goals)
                    if re.match(r'^===\s?target.+$', line, re.IGNORECASE):
                        if ':' in line:
                            separator = ':'
                        else:
                            separator = '. '
                        t = re.sub(r'===\s?[tT]argets?\s', '', line)
                        target = str(t[:t.index(separator)]).strip()
                        if target.endswith('.'):
                            target = target[:len(target) - 1]

                        in_target = True
                        in_list = False
                        line_ = ''
                        section = line.replace('=', '').strip()
                        goals = self._append(str(section[section.index(separator)+1:]).strip(),
                                             goal, target, f'Targets',
                                             page_title, page_revision_id, page_revision_timestamp, goals)
                    else:
                        in_target = False
                        in_list = False
                        line_ = ''
                elif in_target and line.startswith('== '):
                    break
                elif in_target:
                    if not in_list and line.endswith(':'):
                        in_list = True
                        line_ = line
                    elif in_list and line.startswith('*'):
                        line_ += line.replace('*', '') + ';; '
                    elif in_list and not line.startswith('*'):
                        in_list = False
                        goals = self._append(line_, goal, target, f'Targets',
                                             page_title, page_revision_id, page_revision_timestamp, goals)
                        line_ = line
                        if line.endswith(':'):
                            in_list = True
                    else:
                        line_ = line

                    if not in_list:
                        goals = self._append(line_, goal, target, f'Targets',
                                             page_title, page_revision_id, page_revision_timestamp, goals)

        return goals

    def _parse_specific_article(self, page_text: str, page_title: str,
                                page_revision_id: str, page_revision_timestamp: str,
                                goals: DataFrame) -> DataFrame:
        goal = page_title[len('Sustainable Development Goal '):]
        lines = page_text.splitlines()

        goals = self._parse_lead_section(goal, page_title, page_revision_id, page_revision_timestamp,
                                         lines, goals, 'Lead section')

        for section in self.sections:
            goals = self._parse_section(goal, page_title, page_revision_id, page_revision_timestamp,
                                        lines, goals, section=section)

        goals = self._parse_target(goal, page_title, page_revision_id, page_revision_timestamp,
                                   lines, goals)

        return goals

    def _parse_general_article(self, page_text: str, page_title: str,
                               page_revision_id: str, page_revision_timestamp: str,
                               goals: DataFrame) -> DataFrame:
        lines = page_text.splitlines()

        in_goal = False
        goal = None
        for line in lines:
            if line and not re.match(r'^\s+$', line):
                if line.startswith('==== Goal '):
                    goal = line[len('==== Goal '):line.index(':')]
                    in_goal = True
                elif in_goal and re.match(r'^===[\sA-Z].*$', line):
                    in_goal = False
                elif in_goal:
                    if goal:
                        goals = self._append(line, goal, '', 'General', page_title,
                                             page_revision_id, page_revision_timestamp, goals)

        return goals

    def _parse_xml(self, files: list, flags: list) -> DataFrame:
        goals = DataFrame(columns=['id', 'goal', 'target', 'section',
                                   'source_url', 'title',
                                   'revision_id', 'revision_timestamp', 'text'], dtype=str)

        for file, flag in zip(files, flags):
            tree = ElementTree.parse(file)
            root = tree.getroot()

            for page in root.findall('wiki:page', self.ns):
                page_title = page.find('wiki:title', self.ns).text
                page_revision_id = page.find('wiki:revision', self.ns).find('wiki:id', self.ns).text
                page_revision_timestamp = page.find('wiki:revision', self.ns).find('wiki:timestamp', self.ns).text
                page_text = page.find('wiki:revision', self.ns).find('wiki:text', self.ns).text

                page_text = self._remove_wiki_markup(page_text)

                if flag:
                    goals = self._parse_general_article(page_text, page_title,
                                                        page_revision_id, page_revision_timestamp, goals)
                else:
                    goals = self._parse_specific_article(page_text, page_title,
                                                         page_revision_id, page_revision_timestamp, goals)

        goals['text'] = goals['text'].apply(lambda x: self._remove_link_markup(x))
        return goals

    def run(self, files: list, flags: list) -> DataFrame:
        return self._parse_xml(files, flags)


class TextSplitter:
    def __init__(self, min_length: int = 20, max_length: int = 40, max_std: int = 15):
        self.min_length = min_length
        self.max_length = max_length
        self.max_std = max_std

    @staticmethod
    def _correct_fullstops(text: str, forward: bool) -> str:
        if forward:
            text = re.sub(r'e\.g\.', 'e~g~', text)
            text = re.sub(r'i\.e\.', 'i~e~', text)
            text = re.sub(r'(\W)etc\.(\W+[^A-Z])', '\\1' + 'etc~' + '\\2', text)
            text = re.sub(r'(\W)etc\.$', '\\1' + 'etc~', text)
            text = re.sub(r'(\W)([A-Za-z])\.([A-Za-z])\.(\W)', '\\1' + '\\2' + '~' + '\\3' + '~' + '\\4', text)
            text = re.sub(r'([a-zA-Z)])\.([A-Z])', '\\1' + '. ' + '\\2', text)
            text = re.sub(r'([0-9)]{3,})\.([A-Z])', '\\1' + '. ' + '\\2', text)
        else:
            text = re.sub(r'e~g~', 'e.g.', text)
            text = re.sub(r'i~e~', 'i.e.', text)
            text = re.sub(r'(\W)etc~(\W+[^A-Z])', '\\1' + 'etc.' + '\\2', text)
            text = re.sub(r'(\W)etc~$', '\\1' + 'etc.', text)
            text = re.sub(r'(\W)([A-Za-z])~([A-Za-z])~(\W)', '\\1' + '\\2' + '.' + '\\3' + '.' + '\\4', text)
        return text

    def split_to_parts(self, texts: list, separators: str = r'\. |\? |! |;; ') -> list:
        parts = []
        for text in texts:
            text = self._correct_fullstops(text, forward=True)
            parts_ = re.split(separators, text)
            parts += parts_
        p = [self._correct_fullstops(e, forward=False) for e in parts if e]
        return p

    @staticmethod
    def _concatenate_sentences(sentence_one: str, sentence_two: str) -> str:
        sentence_one = sentence_one.strip()
        sentence_two = sentence_two.strip()

        if re.match(r'^[^A-Z].+', sentence_two):
            sentences = sentence_one + ' ' + sentence_two
        elif re.match(r'^[^A-Z].+', sentence_two) and re.match(r'.+is\s*$', sentence_one):
            sentences = sentence_one + ' ' + sentence_two
        elif re.match(r'.+[:;?!]\s*$', sentence_one):
            sentences = sentence_one + ' ' + sentence_two
        elif not sentence_one:
            sentences = sentence_two
        else:
            sentences = sentence_one + '. ' + sentence_two
        sentences = re.sub(r'(\.\s?){2,}', '. ', sentences)
        sentences = re.sub(r'\s{2,}', ' ', sentences)
        sentences = re.sub(r'^\.\s+', '', sentences)
        return sentences.strip()

    def concatenate_parts(self, parts, concatenated_lowercase):
        concatenated = []
        if not concatenated_lowercase:
            concatenated_lowercase = []
        current_text = ''
        current_len = 0

        parts_len = [len(p.split()) for p in parts]

        for p, l in zip(parts, parts_len):
            if not p or re.match(r'^\s+$', p.strip()) or re.match(r'^\.+$', p.strip()) or re.match('^Category:.+$',
                                                                                                   p.strip()):
                continue

            expected_len = current_len + l
            if (expected_len > self.max_length) and (current_len >= self.min_length):
                current_text_ = self._concatenate_sentences(current_text, '')
                if current_text_.lower() not in concatenated_lowercase:
                    concatenated.append(current_text_)
                    concatenated_lowercase.append(current_text_.lower())
                current_text = p
                current_len = l
            elif (expected_len > self.max_length) and (current_len < self.min_length):
                max_diff = abs(expected_len - self.max_length)
                min_diff = abs(self.min_length - current_len)
                if current_len > 0 and max_diff >= min_diff:
                    current_text_ = self._concatenate_sentences(current_text, '')
                    if current_text_.lower() not in concatenated_lowercase:
                        concatenated.append(current_text_)
                        concatenated_lowercase.append(current_text_.lower())
                    current_text = p
                    current_len = l
                else:
                    current_text_ = self._concatenate_sentences(current_text, p)
                    if current_text_.lower() not in concatenated_lowercase:
                        concatenated.append(current_text_)
                        concatenated_lowercase.append(current_text_.lower())
                    current_text = ''
                    current_len = 0
            else:
                current_text = self._concatenate_sentences(current_text, p)
                current_len = expected_len

        if current_text:
            current_text_ = self._concatenate_sentences(current_text, '')
            if current_text_.lower() not in concatenated_lowercase:
                concatenated.append(current_text_)
                concatenated_lowercase.append(current_text_.lower())

        return [concatenated, concatenated_lowercase]


class TextCleaner:
    indicator_masking_patterns = [
        (r'^(This|The) target has one indicator: [“”"]?[A-Za-z,()\-\s]+[“”"]?\.', '', False),
        (r'(\*\s?)Indicator\s[0-9]{1,2}\.[0-9a-z]{1,2}\.[0-9]{1,2}(\.[a-z])?\.?:?.+;;', '', True),
        (r'Indicator\s(is\s)?[0-9]{1,2}\.[0-9a-z]{1,2}\.[0-9]{1,2}(\.[a-z])?\.?\s(it\s)?is\s(the\s)?[“”"].+[“”"]\.', '', True),
        (r'Indicator\s(is\s)?[0-9]{1,2}\.[0-9a-z]{1,2}\.[0-9]{1,2}(\.[a-z])?\.?\sis\sthe\s[“”"].+[“”"]\.', '', True),
        (r'^.+Indicator\s[0-9]{1,2}\.[0-9a-z]{1,2}\.[0-9]{1,2}(\.[a-z])?\.?:?.+$', '', True),
        (r'(\*\s?)?[iI]ndicator\s[0-9]{1,2}\.[0-9a-z]{1,2}\.[0-9]{1,2}(\.[a-z])?\.?:?[^.]+\.', '', True),
        (r'^(\*\s?)?[iI]ndicator\s[0-9]{1,2}\.[0-9a-z]{1,2}\.[0-9]{1,2}(\.[a-z])?\.?:?[^.]+$', '', True),
        (r'[0-9]{1,2}\.[0-9a-z]{1,2}\.[0-9]{1,2}(\.[a-z])?\.?\sis\s(the\s)?[“”"].+[“”"]\.$', '', True)
    ]

    list_masking_patterns = [
        (r'\s\(?[a-z]\)', ' ', False),
        (r'^\(?[a-z]\)', ' ', False),
        (r'\s\[?[a-z]]', ' ', False),
        (r'\(i+\)', ' ', False),
        (r'\s\(?[0-9]{1,2}\)', ' ', False),
        (r'\s?\[[0-9]{1,2}]', ' ', False)
    ]

    number_masking_gen_patterns = [
        (r'["“”*]', '', False),
        (r'\W\'', ' ', False),
        (r'\'\W', ' ', False),
        (r'\s{2,}', ' ', False)
    ]

    number_masking_patterns = [
        (r'(indicator\s+)?[0-9]{1,2}\.[a-z0-9]{1,2}\.[0-9]{1,2}(\([a-z]\))?', 'indicator', True),
        (r'indicator [0-9]{1,2}\.[a-z0-9]{1,2}', 'indicator', True),
        (r'^Target\s+[0-9]{1,2}\.[a-zA-Z0-9]{1,2}\.?', 'The target', False),
        (r'(\.\s)Target\s+[0-9]{1,2}\.[a-zA-Z0-9]{1,2}\.?', '\\1' + 'The target', False),
        (r'([a-z:]\s)(SDG\s)?[tT]arget\s+[0-9]{1,2}\.[a-zA-Z0-9]{1,2}\.?', '\\1' + 'the target', False),
        (r'\([tT]arget\s+[0-9]{1,2}\.[a-zA-Z0-9]{1,2}\.?\)', '', False),
        (r'\([tT]argets? [0-9]{1,2}\.[a-zA-Z0-9]{1,2}\.?(, [0-9]{1,2}\.[a-zA-Z0-9]{1,2}\.?)* '
         r'and [0-9]{1,2}\.[a-zA-Z0-9]{1,2}\.?\)', '', False),
        (r'[0-1]{1,2}\.[0-9a-z]{1,2}\s–', '', False),
        (r'[0-9]{1,2}\.[a-z0-9]{1,2}\.?: ', '', False),
        (r'(\.\s)Targets\s+[0-9]{1,2}\.[a-zA-Z0-9]{1,2}\.? through [0-9]{1,2}\.[a-zA-Z0-9]{1,2}\.?',
         '\\1' + 'The targets', False),
        (r'Target[s]?\s+[0-9]{1,2}\.[a-zA-Z0-9]{1,2}\.? and [0-9]{1,2}\.[a-zA-Z0-9]{1,2}\.?', 'The targets', False),
        (r'[iI]ndicator\s+[0-9]{1,2}\.[a-zA-Z0-9]{1,2}\.?', 'the target', False),
        (r'\((Sustainable Development Goal|SDG|Goal) [0-9]{1,2}( or (SDG|Goal) [0-9]{1,2})?\s?\)', '', False),
        (r'\((Sustainable Development Goal|SDG|Goal) [0-9]{1,2}( or (Global )?Goal [0-9]{1,2})?\s?\)', '', False),
        (r'\((Sustainable Development Goal|SDG|Goal) [0-9]{1,2}( or the goal)?\s?\)', '', False),
        (r'\((Sustainable Development Goal|SDG|Goal) [0-9]{1,2}\s?,', '(the goal', False),
        (r'^(Sustainable (Development )?Goal|SDG|Goal)\s+[0-9]{1,2}', 'The goal', False),
        (r'(\.\s)(Sustainable Development Goal|SDG|Goal)\s+[0-9]{1,2}', '\\1' + 'The goal', False),
        (r'([a-z:,]\s)(Sustainable_Development_Goal|[A-Z]Sustainable Development Goal|SDG|sdg|Goal|'
         r'SDG Goal)[\s_]+[0-9]{1,2}', '\\1' + 'the goal', False),
        (r'#Goal [0-9]{1,2}', '#Goal ', False),
        (r'\s+\([0-9]{1,2}\.[a-z0-9]{1,2}\.?\)\s+', ' ', False),
        (r'\s{2,}', ' ', False),
    ]

    number_masking_add_patterns = [
        (r'\s+[0-9]{1,2}\.[a-z0-9]{1,2}\.?\s+', ' ', False),
        (r'the goal ;', '', False),
        (r'\(((the goal|the target)\s*)+\)', '', False)
    ]

    common_phrases_masking_patterns = [
        (r'\s+', ' ', False),
        (r'A proposal has been tabled in [0-9]{4} to delete the (indicator|target)\.?', '', True),
        (r'In [0-9]{4} it was proposed to delete the former\.?', '', True),
        (r'The [gG]oal has ([a-z\-]+|[0-9]+) targets and ([a-z\-]+|[0-9]+) indicators to measure progress'
         r'( toward targets)?\.', '', True),
        (r'The [gG]oal has ([a-z\-]+|[0-9]+) indicators to measure progress toward targets\.', '', True),
        (r'(The|This) target (only )?(has|includes) (only )?(a )?([a-z\-]+|[0-9]+) ([iI]ndicator[s]?[:.\s()]*)?', '',
         False),
        (r'This target has ([a-z\-]+|[0-9]+) ([A-Z])', '\\2', False),
        (r'^(Its [a-z]+ )?[iI]ndicators are:\s*;*$', '', False),
        (r'It has (only )?([a-z]+|[0-9]+) [iI]ndicator[s]?[:.\s()]*', '', False),
        (r'The ([a-z\-]+|[0-9]+\s)?targets are:\s*', '', False),
        (r'^Long (version|title):\s*', '', False),
        (r'The full[\s\-](text|title|main aim) (of|for) (the|this) (goal|target)(\sis)?(\sto)?:?\s*', '', False),
        (r'The full[\s\-](text|title)( the target)? is:?\s*', '', False),
        (r'^The target is( to| formulated as)?:?\s*', '', False),
        (r'^The text of the target is:?\s*', '', False),
        (r'^The goal is to:\s*', '', False),
        (r'The official (title|target|wording|text)( (of|for) the (goal|target))?(\sis)?:?', '', False),
        (r'The ([a-z]+|[0-9]+) outcome(-related|-oriented)? targets (are|include):?\s*', '', False),
        (r'The first ([a-z]+|[0-9]+) targets are outcome targets:?\s*', '', False),
        (r'The ([a-z]+|[0-9]+) target[s]? of the goal (is|are) the target[s]?:?\s*', '', False),
        (r'The goal has ([a-z]+|[0-9]+)( outcome)? targets:?\s*', '', False),
        (r'^There (is|are) ([a-z\-]+|[0-9]+) indicator[s]?:?\s*$', '', False),
        (r'^The ([a-z]+|[0-9]+) indicator[s]? (are|include):?\s*(The\s*)?$', '', False),
        (r'^Indicator[s]? (are|include):\s*$', '', False),
        (r'["“”*]', '', False),
        (r'\s\'', ' ', False),
        (r'\'\s', ' ', False),
        (r'^#\s', '', False),
        (r'\s+', ' ', False),
        (r'([tT]he\s)([tT]he\s)+', '\\1', False)
    ]

    @staticmethod
    def masking(s: str, pattern_list: list) -> str:
        for (pattern, replacement, ignore) in pattern_list:
            if ignore:
                s = re.sub(pattern, replacement, s, re.IGNORECASE)
            else:
                s = re.sub(pattern, replacement, s)
        return s

    def generic_indicator_masking(self, s: str) -> str:
        return self.masking(s, self.indicator_masking_patterns)

    def numbers_masking(self, s: str, section: str) -> str:
        s = self.masking(s, self.number_masking_gen_patterns)

        old_s = ''
        while old_s != s:
            old_s = s
            s = self.masking(s, self.number_masking_patterns)

            if section.lower() == 'links with other sdgs':
                s = self.masking(s, self.number_masking_add_patterns)

        return s

    def list_masking(self, s: str) -> str:
        return self.masking(s, self.list_masking_patterns)

    def common_phrases_masking(self, s: str) -> str:
        return self.masking(s, self.common_phrases_masking_patterns)


class SentenceFilter:
    def __init__(self, min_df: int = 10, eps: float = 0.3, min_samples: int = 10):
        kwargs = {
            'ngram_range': (1, 1),
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': 'word',
            'min_df': min_df,
            'sublinear_tf': True,
            'norm': 'l2',
            'stop_words': 'english',
        }
        self.vec = TfidfVectorizer(**kwargs)

        self.eps = eps
        self.min_samples = min_samples

    def _cluster_mask(self, sentences: list) -> ndarray:
        tf_idf_vectors = self.vec.fit_transform(sentences).toarray().astype(float32)
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='cosine').fit(tf_idf_vectors)
        return concatenate([dbscan.labels_.reshape(-1, 1), array(sentences).reshape(-1, 1)], axis=1)

    def filter(self, data_paragraphs: dict) -> dict:
        mask = self._cluster_mask([s for list_ in data_paragraphs.values() for l_ in list_ for s in l_])
        retained_clusters = ['-1']

        filtered_data_paragraphs = dict()
        start_index = 0
        for key, list_ in data_paragraphs.items():
            current_list = []
            for l_ in list_:
                current_l = []
                span = len(l_)
                end_index = start_index + span
                current_mask = mask[start_index:end_index, :]

                for sentence, cluster, expected_sentence in zip(l_, current_mask[:, 0], current_mask[:, 1]):
                    assert sentence == expected_sentence
                    if cluster in retained_clusters:
                        current_l.append(sentence)

                current_list.append(current_l)
                start_index = end_index
            filtered_data_paragraphs[key] = current_list
        return filtered_data_paragraphs
