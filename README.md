# Embed4SD: Enhancing Sustainable Development Progress Monitoring Through Natural Language Processing

This repository contains the code for the research article "Embed4SD: Enhancing Sustainable Development Progress Monitoring Through Natural Language Processing", authored by A. Gjorgjevikj, K. Mishev, D. Trajanov, and Lj. Kocarev.

## Requirements

- python  3.10.6
- numpy 1.23.3
- pandas  1.4.4
- scikit-learn  1.1.2
- tensorflow  2.10.0
- tensorflow-hub  0.12.0
- tensorflow-text 2.10.0
- torch 1.13.0
- transformers  4.24.0
- sentence-transformers 2.2.2

## Sentence encoder fine-tuning and testing

Sentence encoder fine-tuning and testing processes, applicable to the sentence encoders listed in the research article.

### Fine-Tuning Process

1. Download the XML files containing the Wikipedia article revisions and copy them to an appropriate directory. There should be one XML file containing the general Wikipedia article revision and at least one XML file containing the SDG-specific article revisions.
2. Download the pre-trained general-purpose sentence encoder which should be fine-tuned and copy the files to an appropriate directory.
3. Set the appropriate input parameter values in script [learners.py](embed4sd/learners.py). Examples for the two fine-tuning task categories are given in the script.
4. Set the appropriate random seed in script [learners.py](embed4sd/learners.py).
5. Run script [learners.py](embed4sd/learners.py). The checkpoints are saved in the directory specified by the user.

### Testing Process

1. Download the XML files containing the Wikipedia article revisions and copy them to an appropriate directory. There should be one XML file containing the general Wikipedia article revision and at least one XML file containing the SDG-specific article revisions.
2. Prepare the test CSV file. For the expected file format which depends on the test task category, see classes IndicatorTestDataExtractor and GoalTestDataExtractor in script [extractors.py](embed4sd/extractors.py).
2. Copy the pre-trained general-purpose sentence encoder files which should be evaluated to an appropriate directory.
3. Copy the fine-tuned sentence encoder checkpoint which should be evaluated to an appropriate directory (applicable to fine-tuned sentence encoders).
4. Set the appropriate input parameter values in script [evaluator.py](embed4sd/evaluator.py).
5. Set the appropriate random seed in script [evaluator.py](embed4sd/evaluator.py). Use the same one used during the encoder fine-tuning or a random seed of 42 if testing the base model (without fine-tuning).
6. Run script [evaluator.py](embed4sd/evaluator.py). The results are written in a CSV file titled according to the test task category, in the directory specified by the user.
