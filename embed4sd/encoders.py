import os
from abc import ABC, abstractmethod

import tensorflow as tf
import tensorflow_text as text
import torch
from numpy import empty, concatenate, ndarray
from tensorflow_hub import KerasLayer
from transformers import AutoTokenizer, AutoModel


class Encoder(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def load_model(self, base_model_directory: str, checkpoint_directory: str, checkpoint_number: int):
        raise NotImplementedError()

    @abstractmethod
    def encode(self, embeddings_dimension: int, data: list,
               tokenizer: object, model: object) -> ndarray:
        raise NotImplementedError()

    def run(self, base_model_directory: str,
            embeddings_dimension: int,
            checkpoint_directory: str, checkpoint_number: int,
            data: list) -> ndarray:
        tokenizer, model = self.load_model(base_model_directory, checkpoint_directory, checkpoint_number)
        embeddings = self.encode(embeddings_dimension, data, tokenizer, model)

        return embeddings


class CustomEncoderOne(Encoder):
    """
    Sentence encoder to be used with the Universal Sentence Encoder and Sentence T5 models.
    """
    
    def load_model(self, base_model_directory: str,
                   checkpoint_directory: str, checkpoint_number: int):
        input_ = tf.keras.Input((), dtype=tf.string)
        uni_sent_enc = KerasLayer(base_model_directory, trainable=False)
        embedding = uni_sent_enc(input_)
        model = tf.keras.Model(inputs=input_, outputs=embedding)

        return None, model

    def encode(self, embeddings_dimension: int, data: list,
               tokenizer: tf.keras.Model, model: tf.keras.Model) -> ndarray:
        dataset = tf.data.Dataset.from_tensor_slices(data).batch(64)

        embeddings = empty(shape=(0, embeddings_dimension))
        for i, x in enumerate(dataset):
            y = model(x, training=False)
            embeddings = concatenate([embeddings, y.numpy()], axis=0)

        return embeddings


class CustomEncoderTwo(Encoder):
    """
    Sentence encoder to be used with the Sentence BERT models.
    Based on the examples from https://github.com/UKPLab/sentence-transformers
    """
    
    def load_model(self, base_model_directory: str,
                   checkpoint_directory: str, checkpoint_number: int):
        tokenizer = AutoTokenizer.from_pretrained(base_model_directory)
        model = AutoModel.from_pretrained(base_model_directory)

        path = os.path.join(checkpoint_directory, f'{str(checkpoint_number)}_model.pt')
        if os.path.exists(path):
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            checkpoint = None

        if checkpoint:
            print(f'Restored from {path}')
        else:
            print(f'Checkpoint {checkpoint_number} '
                  f'not found in directory: {checkpoint_directory}.')

        return tokenizer, model

    def encode(self, embeddings_dimension: int, data: list, tokenizer: torch.nn.Module,
               model: torch.nn.Module) -> ndarray:
        batch_size = 64
        total_size = len(data)
        iteration = 0
        start = 0
        end = batch_size if (batch_size <= total_size) else total_size

        embeddings = empty(shape=(0, embeddings_dimension))

        with torch.no_grad():
            while True:
                sentences = tokenizer(data[start:end], padding="max_length", truncation=True,
                                      return_tensors="pt", max_length=128)
                y_ = model(**sentences)

                input_mask_expanded = sentences['attention_mask'].unsqueeze(-1).expand(y_[0].size()).float()
                y_ = torch.sum(y_[0] * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                y_ = torch.nn.functional.normalize(y_, p=2, dim=1)

                embeddings = concatenate([embeddings, y_], axis=0)

                if end == total_size:
                    break

                iteration += 1
                start = end
                if end + batch_size >= total_size:
                    end = total_size
                else:
                    end += batch_size

        return embeddings


class CustomEncoderThree(Encoder):
    """
    Sentence encoder to be used with the Simple Contrastive Learning of Sentence Embeddings (SimCSE) models.
    Based on the examples from https://github.com/princeton-nlp/SimCSE
    """
    
    def load_model(self, base_model_directory: str,
                   checkpoint_directory: str, checkpoint_number: int):
        tokenizer = AutoTokenizer.from_pretrained(base_model_directory)
        model = AutoModel.from_pretrained(base_model_directory)

        path = os.path.join(checkpoint_directory, f'{str(checkpoint_number)}_model.pt')
        if os.path.exists(path):
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            checkpoint = None

        if checkpoint:
            print(f'Restored from {path}')
        else:
            print(f'Checkpoint {checkpoint_number} '
                  f'not found in directory: {checkpoint_directory}.')

        return tokenizer, model

    def encode(self, embeddings_dimension: int, data: list, tokenizer: torch.nn.Module,
               model: torch.nn.Module) -> ndarray:
        batch_size = 64
        total_size = len(data)
        iteration = 0
        start = 0
        end = batch_size if (batch_size <= total_size) else total_size

        embeddings = empty(shape=(0, embeddings_dimension))

        with torch.no_grad():
            while True:
                sentences = tokenizer(data[start:end], padding="max_length", truncation=True,
                                      return_tensors="pt", max_length=128)
                y_ = model(**sentences, output_hidden_states=True, return_dict=True)[1]
                y_ = torch.nn.functional.normalize(y_, p=2, dim=1)

                embeddings = concatenate([embeddings, y_], axis=0)

                if end == total_size:
                    break

                iteration += 1
                start = end
                if end + batch_size >= total_size:
                    end = total_size
                else:
                    end += batch_size

        return embeddings
