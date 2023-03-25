# Embed4SD

Sentence encoder fine-tuning and testing code. The code is applicable to the sentence encoders listed in the research article.

## Fine-Tuning Process

1. Download the XML files containing the Wikipedia article revisions and copy them to an appropriate directory. There should be one XML file containing the general Wikipedia article revision and at least one XML file containing the SDG-specific article revisions.
2. Download the pre-trained general-purpose sentence encoder which should be fine-tuned and copy the files to an appropriate directory.
3. Set the appropriate input parameter values in script [learners.py](./learners.py). Examples for the two fine-tuning task categories are given in the script.
4. Set the appropriate random seed in script [learners.py](./learners.py).
5. Run script [learners.py](./learners.py). The checkpoints are saved in the directory specified by the user.

## Testing Process

1. Download the XML files containing the Wikipedia article revisions and copy them to an appropriate directory. There should be one XML file containing the general Wikipedia article revision and at least one XML file containing the SDG-specific article revisions.
2. Prepare the test CSV file. For the expected file format which depends on the test task category, see classes IndicatorTestDataExtractor and GoalTestDataExtractor in script [extractors.py](./extractors.py).
2. Copy the pre-trained general-purpose sentence encoder files which should be evaluated to an appropriate directory.
3. Copy the fine-tuned sentence encoder checkpoint which should be evaluated to an appropriate directory (applicable to fine-tuned sentence encoders).
4. Set the appropriate input parameter values in script [evaluator.py](./evaluator.py).
5. Set the appropriate random seed in script [evaluator.py](./evaluator.py). Use the same one used during the encoder fine-tuning or a random seed of 42 if testing the base model (without fine-tuning).
6. Run script [evaluator.py](./evaluator.py). The results are written in a CSV file titled according to the test task category, in the directory specified by the user.
