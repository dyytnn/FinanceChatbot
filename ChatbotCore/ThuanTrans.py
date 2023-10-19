import logging
from typing import Any, Text, Dict, List, Type

from joblib import dump, load
from scipy.sparse import hstack, vstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.nlu.featurizers.featurizer import Featurizer
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import TEXT, INTENT

import os

import numpy as np
import json
import rasa
import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig
from transformers import Trainer, TrainingArguments



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger(__name__)

class CustomDataset(Dataset):
    """
    Dataset for training the model.
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    """
    Helper function to compute aggregated metrics from predictions.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=True
)
class TransformerClassifier(IntentClassifier, GraphComponent):
    name = "transformer_classifier"
    # provides = ["intent"]
    # requires = ["text"]
    # defaults = {}
    # language_list = ["en"]
    # model_name = "roberta-base"
    @classmethod
    def required_components(cls) -> List[Type]:
        return [Featurizer]


    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {"class_weight": "balanced", "max_iter": 100, "solver": "lbfgs"}
    
    def __init__(
        self,
        config: Dict[Text, Any],
        name: Text,
        model_storage: ModelStorage,
        resource: Resource,
        tokenizer
    ) -> None:
        print("\n------Init Method ---------\n")
        self.name = name
        self.model_name="albert-base-v2"
        #  self.model_name = component_config.get("model_name", "albert-base-v2")
        self.tokenizer=AutoTokenizer.from_pretrained("albert-base-v2")
        
        # We need to use these later when saving the trained component.
        self._model_storage = model_storage
        self._resource = resource
        print(self)

        print("\n-------- Done -------\n")
        # return self

    def _define_model(self):
        """
        Loads the pretrained model and the configuration after the data has been preprocessed.
        """

        self.config = AutoConfig.from_pretrained("albert-base-v2")
        self.config.id2label = self.id2label
        self.config.label2id = self.label2id
        self.config.num_labels = len(self.id2label)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "albert-base-v2", config=self.config
        )
    
    def _compute_label_mapping(self, labels):
        """
        Maps the labels to integers and stores them in the class attributes.
        """

        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(labels)
        self.label2id = {}
        self.id2label = {}
        for label in np.unique(labels):
            self.label2id[label] = int(label_encoder.transform([label])[0])
        for i in integer_encoded:
            self.id2label[int(i)] = label_encoder.inverse_transform([i])[0]
    


    def train(self, training_data: TrainingData) -> Resource:
        # self.tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        print("-----------------------------")
        print(self)
        print("-----------------------------")

        dataset = self.preprocess_data(training_data, self.component_config)
        
        self._define_model()

        training_args = TrainingArguments(
            output_dir="models",
            num_train_epochs=self.component_config.get("epochs", 15),
            evaluation_strategy="no",
            per_device_train_batch_size=self.component_config.get("batch_size", 24),
            warmup_steps=self.component_config.get("warmup_steps", 500),
            weight_decay=self.component_config.get("weight_decay", 0.01),
            learning_rate=self.component_config.get("learning_rate", 2e-5),
            lr_scheduler_type=self.component_config.get("scheduler_type", "constant"),
            save_strategy="no",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        self.persist(self,"albert-base-v2","./customemodel")
        return self._resource
    
    
    def preprocess_data(self, training_data: TrainingData, params):
        """
        Preprocesses the data to be used for training.
        """

        documents = []
        labels = []
        for message in training_data.training_examples:
            if "text" in message.data:
                documents.append(message.data["text"])
                labels.append(message.data["intent"])
        self._compute_label_mapping(labels)
        targets = [self.label2id[label] for label in labels]
        encodings = AutoTokenizer.from_pretrained("albert-base-v2")(
            documents,
            padding="max_length",
            max_length=params.get("max_length", 64),
            truncation=True,
        )
        dataset = CustomDataset(encodings, targets)

        return dataset


    def persist(self, file_name, model_dir) -> None:

        tokenizer_filename = "tokenizer_{}".format(file_name)
        model_filename = "model_{}".format(file_name)
        config_filename = "config_{}".format(file_name)
        tokenizer_path = os.path.join(model_dir, tokenizer_filename)
        model_path = os.path.join(model_dir, model_filename)
        config_path = os.path.join(model_dir, config_filename)
        self.tokenizer.save_pretrained(tokenizer_path)
        self.model.save_pretrained(model_path)
        self.config.save_pretrained(config_path)
    

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        with model_storage.read_from(resource) as model_dir:
            
            with model_storage.read_from(resource) as path:

                model_data_file = path / "model_data.json"
                model_data = json.loads(rasa.shared.utils.io.read_file(model_data_file))
                component = cls(
                    config, execution_context.node_name, model_storage, resource
                )
                component.clf = model_data
                return component
    
    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        self.process(training_data.training_examples)
        return training_data
            
    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Validates that the component is configured properly."""
        pass
    
    def process(self, messages: List[Message]) -> List[Message]:
        """
        Processes the input given from Rasa. Attaches the output to the message object.

        Args:
            message (Message): input message
        """

        X = self._create_X(messages)
        prediction, confidence, intent_ranking = self._predict(X)
        messages.set(
            "intent", {"name": prediction, "confidence": confidence}, add_to_output=True
        )
        messages.set("intent_ranking", intent_ranking, add_to_output=True)
        return messages

    def _create_X(self, messages: List[Message]) -> csr_matrix:
        """This method creates a sparse X array that can be used for predicting"""
        X = []
        for e in messages:
            # First element is sequence features, second is sentence features
            sparse_feats = e.get_sparse_features(attribute=TEXT)[1]
            # First element is sequence features, second is sentence features
            dense_feats = e.get_dense_features(attribute=TEXT)[1]
            together = hstack(
                [
                    csr_matrix(sparse_feats.features if sparse_feats else []),
                    csr_matrix(dense_feats.features if dense_feats else []),
                ]
            )
            X.append(together)
        return vstack(X)