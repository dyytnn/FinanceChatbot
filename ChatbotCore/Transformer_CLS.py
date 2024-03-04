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
import rasa
import os

import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig
from transformers import Trainer, TrainingArguments




logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    # name = "transformer_classifier"
    # provides = ["intent"]
    # requires = ["text"]
    # defaults = {}
    # language_list = ["en"]
    # model_name = "roberta-base"
    model_name ="albert-base-v2"
    dir_save = "./custom_model"
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
        
    ) -> None:
        print("\n------Init Method ---------\n")
        self.name = name
        self.model_name="albert-base-v2"
        # self.model_name = config["model_name"]= "albert-base-v2"
        # self.model
        self.tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        # We need to use these later when saving the trained component.
        self._model_storage = model_storage
        self._resource = resource
        # print(self)
      
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
        self.model_main=self.model
        return self.model
    
    def _process_intent_ranking(self, outputs):
        """
        Processes the intent ranking, sort in descending order based on confidence. Get only top 10

        Args:
            outputs: model outputs

        Returns:
            intent_ranking (list) - list of dicts with intent name and confidence (top 10 only)
        """
        with open('./transformer/label2id.json') as json_file:
            label2id = json.load(json_file)
        confidences = [float(x) for x in nn.functional.softmax(outputs["logits"], dim = -1)[0]]
        intent_names = list(label2id.keys())
        intent_ranking_all = zip(confidences, intent_names)
        intent_ranking_all_sorted = sorted(
            intent_ranking_all, key=lambda x: x[0], reverse=True
        )
        intent_ranking = [
            {"confidence": x[0], "intent": x[1]} for x in intent_ranking_all_sorted[:10]
        ]
        return intent_ranking


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

        with open("./transformer/label2id.json", "w") as fp:
          json.dump(self.label2id , fp) 
        with open("./transformer/id2label.json", "w") as fp:
          json.dump(self.id2label, fp) 

    def train(self, training_data: TrainingData) -> Resource:
        # self.tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        print("-----------------------------")
        # print(self)
        print("-----------------------------")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        dataset = self.preprocess_data(training_data)
        print("Data Set ")
        # print(dataset)
        self.model=self._define_model()

        self.config = AutoConfig.from_pretrained("albert-base-v2")
        self.config.id2label = self.id2label
        self.config.label2id = self.label2id
        self.config.num_labels = len(self.id2label)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "albert-base-v2", config=self.config
        )
        self.model_main= AutoModelForSequenceClassification.from_pretrained(
            "albert-base-v2", config=self.config
        )
        


        training_args = TrainingArguments(
            output_dir="./custom_model",
            num_train_epochs=21,
            evaluation_strategy="no",
            per_device_train_batch_size=24,
            warmup_steps=500,
            weight_decay=0.01,
            learning_rate=2e-5,
            lr_scheduler_type="constant",
            save_strategy="no",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        self.persist("albert-base-v2","./custom_model")
        return self._resource
    
    
    def preprocess_data(self, training_data: TrainingData):
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
            max_length=64,
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
        print("Model path " +model_path)
        print("Config path " +config_path)
        print("Tokenizer path "+tokenizer_path)
        self.tokenizer.save_pretrained(tokenizer_path)
        self.model.save_pretrained(model_path)
        self.config.save_pretrained(config_path)
    
    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        return cls(config, execution_context.node_name, model_storage, resource)
    
    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        print("--------------- Load Method Start ---------")
        # print(resource)
        # print(resource.name)
        # print("Model Storage")
        # print(model_storage)
        model_data_file ="./custom_model/model_albert-base-v2/config.json"
        print("Model Path Loaded "+ model_data_file)
        
        
        # model_data = json.loads(rasa.shared.utils.io.read_file(model_data_file))
        component = cls(
              config, execution_context.node_name, model_storage, resource
          )
        # component.clf = model_data
        print("--------------- Load Method End ---------")
        return component
            

    
    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        self.process(training_data.training_examples)
        return training_data
            
    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Validates that the component is configured properly."""
        pass
    
    def _predict(self, text):
        """
        Predicts the intent of the input text.

        Args:
            text (str): input text

        Returns:
            prediction (string) - intent name
            confidence (float) - confidence of the intent
            intent_ranking (list) - list of dicts with intent name and confidence (top 10 only)
        """
        print("------- Predict Method Start --------")
        

        inputs = self.tokenizer(
              text,
              padding="max_length",
              max_length=64,
              truncation=True,
              return_tensors="pt",
        ).to(DEVICE)
        print("Model Loaded From Self");
        model_name_or_path = "./custom_model/model_albert-base-v2" #path/to/your/model/or/name/on/hub
        # device = "cpu" # or "cuda" if you have a GPU
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path).to(DEVICE)
        outputs = model(**inputs)
        with open('./transformer/id2label.json') as json_file:
            id2label = json.load(json_file)
        
        print(" id2Label   ")
        print(id2label)
        print("Result Predicted Started")
        # arr = outputs["logits"]
        arr = nn.functional.softmax(outputs["logits"], dim = -1)
        print(arr)
        # arr = nn.functional.softmax(outputs.logits, dim = -1)
        maxvalue=arr[0].tolist() 
        print(" List $ " )
        print(maxvalue )
        print("\n")  
        print("Index  Value " + str(int(max(maxvalue))))
        confidence = float(max(maxvalue))
        predict_index=maxvalue.index(max(maxvalue))
        prediction = id2label[str(int(predict_index))]
        intent_ranking = self._process_intent_ranking(outputs)
        print("Result Predicted Ended")
        print("------- Predict Method End --------")
        return prediction, confidence, intent_ranking


    def process(self, messages: List[Message]) -> List[Message]:
        """
        Processes the input given from Rasa. Attaches the output to the message object.

        Args:
            message (Message): input message
        """
        print("-------- Process Method Start -----------------")

        for item in messages:
            print("Each Message ",item.data["text"])
            prediction, confidence, intent_ranking = self._predict(item.data["text"])
            
            item.set(
                "intent", {"name": prediction, "confidence": confidence}, add_to_output=True
            )
            item.set("intent_ranking", intent_ranking, add_to_output=True)


        # X = self._create_X(messages)
        # print("X Variable")
        # print(X)
        # text = messages.data["text"]
       
        print("-------- Process Method Start -----------------")
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
    





