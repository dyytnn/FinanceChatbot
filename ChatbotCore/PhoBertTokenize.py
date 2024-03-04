import logging
from typing import Any, Text, Dict, List, Type
from sklearn.model_selection import train_test_split
from joblib import dump, load
from scipy.sparse import hstack, vstack, csr_matrix
from sklearn.linear_model import LogisticRegression

from transformers import AutoModel, AutoTokenizer  # Thư viện BERT
import torch
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
from sklearn.svm import SVC
import underthesea
import numpy
import re
import os
import json
from joblib import dump, load
import pickle
logger = logging.getLogger(__name__)

"""
    model_storage: provides access to data from all graph components.
    resource allows to uniquely identify graph component's location in the model storage
"""

@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=True
)
class PhoBert_SVM(IntentClassifier, GraphComponent):
    @classmethod
    def required_components(cls) -> List[Type]:
        return [Featurizer]

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["sklearn"]

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
        os.makedirs("./custom_model/phobert_svm", exist_ok=True)

        print("\n------Init Method ---------\n")
        self.name = name
        self.v_phobert, self.v_tokenizer = self.load_bert()
        self.clf = SVC(kernel="linear", probability=True, gamma=0.125)

        # We need to use these later when saving the trained component.
        self._model_storage = model_storage
        self._resource = resource
        print("\n-------- Done -------\n")

    def make_bert_features(self, v_text):
        global phobert, sw
        v_tokenized = []
        max_len = 100  # Mỗi câu dài tối đa 100 từ
        for i_text in v_text:
            print("Đang xử lý line = ", i_text)
            # Phân thành từng từ
            line = underthesea.word_tokenize(i_text)

            # Lọc các từ vô nghĩa
            # filtered_sentence = [w for w in line if not w in sw]
            # # Ghép lại thành câu như cũ sau khi lọc
            # line = " ".join(filtered_sentence)
            # line = underthesea.word_tokenize(line, format="text")
            # print("Word segment  = ", line)
            # Tokenize bởi BERT
            line = self.v_tokenizer.encode(line)
            v_tokenized.append(line)

        # Chèn thêm số 1 vào cuối câu nếu như không đủ 100 từ
        padded = numpy.array([i + [1] * (max_len - len(i)) for i in v_tokenized])
        print("padded:", padded[0])
        print("len padded:", padded.shape)

        # Đánh dấu các từ thêm vào = 0 để không tính vào quá trình lấy features
        attention_mask = numpy.where(padded == 1, 0, 1)
        print("attention mask:", attention_mask[0])

        # Chuyển thành tensor
        padded = torch.tensor(padded).to(torch.long)
        print("Padd = ", padded.size())
        attention_mask = torch.tensor(attention_mask)

        # Lấy features dầu ra từ BERT
        with torch.no_grad():
            last_hidden_states = self.v_phobert(
                input_ids=padded, attention_mask=attention_mask
            )

        v_features = last_hidden_states[0][:, 0, :].numpy()
        print(v_features.shape)
        return v_features

    def load_bert(self):
        v_phobert = AutoModel.from_pretrained("vinai/phobert-base")
        v_tokenizer = AutoTokenizer.from_pretrained(
            "vinai/phobert-base", use_fast=False
        )
        return v_phobert, v_tokenizer

    def _create_X(self, messages: List[Message]) -> csr_matrix:
        """This method creates a sparse X array that can be used for predicting"""
        X = []
        for e in messages:
            print(e)
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

    # Hàm chuẩn hoá câu
    def standardize_data(row):
        # Xóa dấu chấm, phẩy, hỏi ở cuối câu
        row = re.sub(r"[\.,\?]+$-", "", row)
        # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
        row = (
            row.replace(",", " ")
            .replace(".", " ")
            .replace(";", " ")
            .replace("“", " ")
            .replace(":", " ")
            .replace("”", " ")
            .replace('"', " ")
            .replace("'", " ")
            .replace("!", " ")
            .replace("?", " ")
            .replace("-", " ")
            .replace("?", " ")
        )
        row = row.strip().lower()
        return row

    def computeLabelToId(self, training_data: TrainingData):
        mySetIntent = set()
        for e in training_data.training_examples:
            # print(e)
            if e.get(INTENT):
                print("  e.get(INTENT)   ")
                print(e.get(INTENT))

                if e.get("text"):
                    mySetIntent.add(e.get(INTENT))
                #     print(" e.get("text") ")
                #     print(e.get("text"))
        label2id = {}
        id2label = {}
        mySetIntent = list(mySetIntent)
        for index in range(len(mySetIntent)):
            label2id[mySetIntent[index]] =index
            id2label[index] = mySetIntent[index]
        with open("./custom_model/phobert_svm/label2id.json", "w") as fp:
            json.dump(label2id, fp)
        with open("./custom_model/phobert_svm/id2label.json", "w") as fp:
            json.dump(id2label, fp)
        return label2id, id2label

    def _create_training_matrix(self, training_data: TrainingData):
        """
        This method creates a scikit-learn compatible (X, y)-pair for training
        the logistic regression model.
        """
        X = []
        y = []

        for e in training_data.training_examples:
            print(e)
            if e.get(INTENT):
                print("  e.get(INTENT)   ")
                print(e.get(INTENT))
                if e.get("text"):
                    print('e.get("text") ')
                    print(e.get("text"))
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
                    y.append(e.get(INTENT))
        return vstack(X), y

    def load_data(self, training_data: TrainingData):
        v_text = []
        v_label = []
        with open("./custom_model/phobert_svm/label2id.json") as json_file:
            label2id = json.load(json_file)
        print(label2id)
        for e in training_data.training_examples:
            # print(e)
            if e.get(INTENT):
                print("  e.get(INTENT)   ")
                print(e.get(INTENT))
                if e.get("text"):
                    print('e.get("text")')
                    print(e.get("text"))
                    # v_text.append(self.standardize_data(e.get("text")))
                    v_text.append(e.get("text"))
                    v_label.append(int(label2id[e.get(INTENT)]))

        # print(v_label)
        return v_text, v_label

    def train(self, training_data: TrainingData) -> Resource:
        self.computeLabelToId(training_data)
        text, label = self.load_data(training_data)
        features = self.make_bert_features(text)
        print("-----------------------------")
        print(self)
        print("-----------------------------")
        X_train, X_test, y_train, y_test = train_test_split(
            features, label, test_size=0.2, random_state=0
        )

        self.clf.fit(features, label)
        sc = self.clf.score(X_test, y_test)
        print("Kết quả train model, độ chính xác = ", sc * 100, "%")

        self.persist()

        return self._resource

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        return cls(config, execution_context.node_name, model_storage, resource)
    def prePredict(self,text):
      
      v_tokenized = []
      max_len = 100  # Mỗi câu dài tối đa 100 từ
      for i_text in v_text:
          print("Đang xử lý line = ", i_text)
          # Phân thành từng từ
          line = underthesea.word_tokenize(i_text)
          # Lọc các từ vô nghĩa
          # filtered_sentence = [w for w in line if not w in sw]
          # # Ghép lại thành câu như cũ sau khi lọc
          # line = " ".join(filtered_sentence)
          # line = underthesea.word_tokenize(line, format="text")
          # print("Word segment  = ", line)
          # Tokenize bởi BERT
          line = self.v_tokenizer.encode(line)
          v_tokenized.append(line)
      # Chèn thêm số 1 vào cuối câu nếu như không đủ 100 từ
      padded = numpy.array([i + [1] * (max_len - len(i)) for i in v_tokenized])
      print("padded:", padded[0])
      print("len padded:", padded.shape)
      # Đánh dấu các từ thêm vào = 0 để không tính vào quá trình lấy features
      attention_mask = numpy.where(padded == 1, 0, 1)
      print("attention mask:", attention_mask[0])
      # Chuyển thành tensor
      padded = torch.tensor(padded).to(torch.long)
      print("Padd = ", padded.size())
      attention_mask = torch.tensor(attention_mask)
      # Lấy features dầu ra từ BERT
      with torch.no_grad():
          last_hidden_states = self.v_phobert(
              input_ids=padded, attention_mask=attention_mask
          )
      v_features = last_hidden_states[0][:, 0, :].numpy()


    def process(self, messages: List[Message]) -> List[Message]:
        # cls = SVC(kernel="linear", probability=True, gamma=0.125)
        # with open('./custom_model/phobert_svm/phobert_svm.pkl', 'rb') as f:
        #     cls=pickle.load(f)
        print("-------------- Process Method ---------------")
        with open("./custom_model/phobert_svm/id2label.json") as json_file:
            id2label = json.load(json_file)
        # with open('./custom_model/phobert_svm/phobert_svm.pkl', 'rb') as f:
        #     cls = pickle.load(f)
        # cls = load('./custom_model/phobert_svm/phobert_svm.pkl')
        listMessages = []
        for item in messages:
          listMessages.append(
            item.data["text"]
          )
        X = self.make_bert_features(listMessages)
        pred = self.clf.predict(X)
        print(" ---- Model Predicted ----")
        print(pred)
        print("-------- % model -- - ---")
        
        probas = self.clf.predict_proba(X)
        print(probas)
        print("Predict Progress Inside")
        for idx, message in enumerate(messages):
            intent = {"name": pred[idx], "confidence": probas[idx].max()}
            intents = self.clf.classes_
            intent_info = {
                k: v
                for i, (k, v) in enumerate(zip(intents, probas[idx]))
                if i < LABEL_RANKING_LENGTH
            }
            intent_ranking = [
                {"name": k, "confidence": v} for k, v in intent_info.items()
            ]
            print("Intent Predicted")
            print(intent)
            intent["name"]=id2label[str(intent["name"])]
            print(intent)
            
            message.set("intent", intent, add_to_output=True)
            message.set("intent_ranking", intent_ranking, add_to_output=True)
        print("-------------- Process Method ---------------")   
        return messages

    def persist(self) -> None:
        print("----- Persit method ---------")
        print(self.clf)
        dump(self.clf, "./custom_model/phobert_svm/phobert_svm.joblib")
        print("Đã lưu model SVM vào file save_model.pkl")
        print("----- Persit method ---------")

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
            classifier = load("./custom_model/phobert_svm/phobert_svm.joblib")
            # with open('./custom_model/phobert_svm/phobert_svm.pkl', 'wb') as f:
            #     pickle.dump(classifier, f)
            # classifier = load(model_dir / f"{resource.name}.joblib")
            component = cls(
                config, execution_context.node_name, model_storage, resource
            )
            component.clf = classifier
            return component
            

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        self.process(training_data.training_examples)
        return training_data

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Validates that the component is configured properly."""
        pass