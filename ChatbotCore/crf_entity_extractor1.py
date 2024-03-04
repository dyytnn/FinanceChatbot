from __future__ import annotations

from collections import OrderedDict
from enum import Enum
import logging
import typing

import numpy as np
from typing import Any, Dict, List, Optional, Text, Tuple, Callable, Type

import rasa.nlu.utils.bilou_utils as bilou_utils
import rasa.shared.utils.io
import rasa.utils.train_utils
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.test import determine_token_labels
from rasa.nlu.tokenizers.spacy_tokenizer import POS_TAG_KEY

# from rasa.nlu.extractors.extractor import EntityExtractorMixin

from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer

from rasa.shared.nlu.training_data.training_data import TrainingData
# from content.drive.MyDrive.Thuan.components.training_data import TrainingData
# from


from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.constants import TOKENS_NAMES
from rasa.shared.nlu.constants import (
    TEXT,
    ENTITIES,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_GROUP,
    ENTITY_ATTRIBUTE_ROLE,
    NO_ENTITY_TAG,
    SPLIT_ENTITIES_BY_COMMA,
    SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE,
)
from rasa.shared.constants import DOCS_URL_COMPONENTS
from rasa.utils.tensorflow.constants import BILOU_FLAG, FEATURIZERS



import abc
from typing import Any, Dict, List, NamedTuple, Text, Tuple, Optional

import rasa.shared.utils.io
from rasa.shared.constants import DOCS_URL_TRAINING_DATA_NLU
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.constants import (
    TOKENS_NAMES,
    ENTITY_ATTRIBUTE_CONFIDENCE_TYPE,
    ENTITY_ATTRIBUTE_CONFIDENCE_ROLE,
    ENTITY_ATTRIBUTE_CONFIDENCE_GROUP,
)
from rasa.shared.nlu.constants import (
    TEXT,
    INTENT,
    ENTITIES,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    EXTRACTOR,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_GROUP,
    ENTITY_ATTRIBUTE_ROLE,
    NO_ENTITY_TAG,
    SPLIT_ENTITIES_BY_COMMA,
    SINGLE_ENTITY_ALLOWED_INTERLEAVING_CHARSET,
)
import rasa.utils.train_utils


class EntityTagSpec(NamedTuple):
    """Specification of an entity tag present in the training data."""

    tag_name: Text
    ids_to_tags: Dict[int, Text]
    tags_to_ids: Dict[Text, int]
    num_tags: int


class EntityExtractorMixin(abc.ABC):
    """Provides functionality for components that do entity extraction.

    Inheriting from this class will add utility functions for entity extraction.
    Entity extraction is the process of identifying and extracting entities like a
    person's name, or a location from a message.
    """

    @property
    def name(self) -> Text:
        """Returns the name of the class."""
        return self.__class__.__name__

    def add_extractor_name(
        self, entities: List[Dict[Text, Any]]
    ) -> List[Dict[Text, Any]]:
        """Adds this extractor's name to a list of entities.

        Args:
            entities: the extracted entities.

        Returns:
            the modified entities.
        """
        for entity in entities:
            entity[EXTRACTOR] = self.name
        return entities

    def add_processor_name(self, entity: Dict[Text, Any]) -> Dict[Text, Any]:
        """Adds this extractor's name to the list of processors for this entity.

        Args:
            entity: the extracted entity and its metadata.

        Returns:
            the modified entity.
        """
        if "processors" in entity:
            entity["processors"].append(self.name)
        else:
            entity["processors"] = [self.name]

        return entity

    @staticmethod
    def filter_irrelevant_entities(extracted: list, requested_dimensions: set) -> list:
        """Only return dimensions the user configured."""
        if requested_dimensions:
            return [
                entity
                for entity in extracted
                if entity[ENTITY_ATTRIBUTE_TYPE] in requested_dimensions
            ]
        return extracted

    @staticmethod
    def find_entity(
        entity: Dict[Text, Any], text: Text, tokens: List[Token]
    ) -> Tuple[int, int]:
        offsets = [token.start for token in tokens]
        ends = [token.end for token in tokens]

        if entity[ENTITY_ATTRIBUTE_START] not in offsets:
            message = (
                "Invalid entity {} in example '{}': "
                "entities must span whole tokens. "
                "Wrong entity start.".format(entity, text)
            )
            raise ValueError(message)

        if entity[ENTITY_ATTRIBUTE_END] not in ends:
            message = (
                "Invalid entity {} in example '{}': "
                "entities must span whole tokens. "
                "Wrong entity end.".format(entity, text)
            )
            raise ValueError(message)

        start = offsets.index(entity[ENTITY_ATTRIBUTE_START])
        end = ends.index(entity[ENTITY_ATTRIBUTE_END]) + 1
        return start, end

    def filter_trainable_entities(
        self, entity_examples: List[Message]
    ) -> List[Message]:
        """Filters out untrainable entity annotations.

        Creates a copy of entity_examples in which entities that have
        `extractor` set to something other than
        self.name (e.g. 'CRFEntityExtractor') are removed.
        """
        filtered = []
        for message in entity_examples:
            entities = []
            for ent in message.get(ENTITIES, []):
                extractor = ent.get(EXTRACTOR)
                if not extractor or extractor == self.name:
                    entities.append(ent)
            data = message.data.copy()
            
            data[ENTITIES] = entities
            filtered.append(
                Message(
                    text=message.get(TEXT),
                    data=data,
                    output_properties=message.output_properties,
                    time=message.time,
                    features=message.features,
                )
            )

        return filtered

    @staticmethod
    def convert_predictions_into_entities(
        text: Text,
        tokens: List[Token],
        tags: Dict[Text, List[Text]],
        split_entities_config: Optional[Dict[Text, bool]] = None,
        confidences: Optional[Dict[Text, List[float]]] = None,
    ) -> List[Dict[Text, Any]]:
        """Convert predictions into entities.

        Args:
            text: The text message.
            tokens: Message tokens without CLS token.
            tags: Predicted tags.
            split_entities_config: config for handling splitting a list of entities
            confidences: Confidences of predicted tags.

        Returns:
            Entities.
        """
        import rasa.nlu.utils.bilou_utils as bilou_utils

        entities = []

        last_entity_tag = NO_ENTITY_TAG
        last_role_tag = NO_ENTITY_TAG
        last_group_tag = NO_ENTITY_TAG
        last_token_end = -1

        for idx, token in enumerate(tokens):
            current_entity_tag = EntityExtractorMixin.get_tag_for(
                tags, ENTITY_ATTRIBUTE_TYPE, idx
            )

            if current_entity_tag == NO_ENTITY_TAG:
                last_entity_tag = NO_ENTITY_TAG
                last_token_end = token.end
                continue

            current_group_tag = EntityExtractorMixin.get_tag_for(
                tags, ENTITY_ATTRIBUTE_GROUP, idx
            )
            current_group_tag = bilou_utils.tag_without_prefix(current_group_tag)
            current_role_tag = EntityExtractorMixin.get_tag_for(
                tags, ENTITY_ATTRIBUTE_ROLE, idx
            )
            current_role_tag = bilou_utils.tag_without_prefix(current_role_tag)

            group_or_role_changed = (
                last_group_tag != current_group_tag or last_role_tag != current_role_tag
            )

            if bilou_utils.bilou_prefix_from_tag(current_entity_tag):
                # checks for new bilou tag
                # new bilou tag begins are not with I- , L- tags
                new_bilou_tag_starts = last_entity_tag != current_entity_tag and (
                    bilou_utils.LAST
                    != bilou_utils.bilou_prefix_from_tag(current_entity_tag)
                    and bilou_utils.INSIDE
                    != bilou_utils.bilou_prefix_from_tag(current_entity_tag)
                )

                # to handle bilou tags such as only I-, L- tags without B-tag
                # and handle multiple U-tags consecutively
                new_unigram_bilou_tag_starts = (
                    last_entity_tag == NO_ENTITY_TAG
                    or bilou_utils.UNIT
                    == bilou_utils.bilou_prefix_from_tag(current_entity_tag)
                )

                new_tag_found = (
                    new_bilou_tag_starts
                    or new_unigram_bilou_tag_starts
                    or group_or_role_changed
                )
                last_entity_tag = current_entity_tag
                current_entity_tag = bilou_utils.tag_without_prefix(current_entity_tag)
            else:
                new_tag_found = (
                    last_entity_tag != current_entity_tag or group_or_role_changed
                )
                last_entity_tag = current_entity_tag

            if new_tag_found:
                print(" Hàm tạo entity mới , đã tắt")
                # new entity found
                   #  Tắt tạo entity mới
                entity = EntityExtractorMixin._create_new_entity(
                    list(tags.keys()),
                    current_entity_tag,
                    current_group_tag,
                    current_role_tag,
                    token,
                    idx,
                    confidences,
                )
                entities.append(entity)
            elif EntityExtractorMixin._check_is_single_entity(
                text, token, last_token_end, split_entities_config, current_entity_tag
            ):
                # print(" Hàm tạo entity mới , đã tắt")
                # current token has the same entity tag as the token before and
                # the two tokens are separated by at most 3 symbols, where each
                # of the symbols has to be either punctuation (e.g. "." or ",")
                # and a whitespace.
                entities[-1][ENTITY_ATTRIBUTE_END] = token.end
                if confidences is not None:
                    EntityExtractorMixin._update_confidence_values(
                        entities, confidences, idx
                    )

            else:
                # print(" Hàm tạo entity mới , đã tắt")
                # the token has the same entity tag as the token before but the two
                # tokens are separated by at least 2 symbols (e.g. multiple spaces,
                # a comma and a space, etc.) and also shouldn't be represented as a
                # single entity
                #  Tắt tạo entity mới
                entity = EntityExtractorMixin._create_new_entity(
                    list(tags.keys()),
                    current_entity_tag,
                    current_group_tag,
                    current_role_tag,
                    token,
                    idx,
                    confidences,
                )
                entities.append(entity)

            last_group_tag = current_group_tag
            last_role_tag = current_role_tag
            last_token_end = token.end

        for entity in entities:
            entity[ENTITY_ATTRIBUTE_VALUE] = text[
                entity[ENTITY_ATTRIBUTE_START] : entity[ENTITY_ATTRIBUTE_END]
            ]

        return entities

    @staticmethod
    def _update_confidence_values(
        entities: List[Dict[Text, Any]], confidences: Dict[Text, List[float]], idx: int
    ) -> None:
        # use the lower confidence value
        entities[-1][ENTITY_ATTRIBUTE_CONFIDENCE_TYPE] = min(
            entities[-1][ENTITY_ATTRIBUTE_CONFIDENCE_TYPE],
            confidences[ENTITY_ATTRIBUTE_TYPE][idx],
        )
        if ENTITY_ATTRIBUTE_ROLE in entities[-1]:
            entities[-1][ENTITY_ATTRIBUTE_CONFIDENCE_ROLE] = min(
                entities[-1][ENTITY_ATTRIBUTE_CONFIDENCE_ROLE],
                confidences[ENTITY_ATTRIBUTE_ROLE][idx],
            )
        if ENTITY_ATTRIBUTE_GROUP in entities[-1]:
            entities[-1][ENTITY_ATTRIBUTE_CONFIDENCE_GROUP] = min(
                entities[-1][ENTITY_ATTRIBUTE_CONFIDENCE_GROUP],
                confidences[ENTITY_ATTRIBUTE_GROUP][idx],
            )

    @staticmethod
    def _check_is_single_entity(
        text: Text,
        token: Token,
        last_token_end: int,
        split_entities_config: Dict[Text, bool],
        current_entity_tag: Text,
    ) -> bool:
        # current token has the same entity tag as the token before and
        # the two tokens are only separated by at most one symbol (e.g. space,
        # dash, etc.)
        if token.start - last_token_end <= 1:
            return True

        # Tokens need to be no further than 3 positions apart
        # The magic number 3 is chosen such that the following two cases can be
        # extracted
        #   - Schönhauser Allee 175, 10119 Berlin
        #     (address compounds separated by 2 tokens (", "))
        #   - 22 Powderhall Rd., EH7 4GB
        #     (abbreviated "Rd." results in a separation of 3 tokens ("., "))
        # More than 3 might already introduce cases that shouldn't be considered by
        # this logic
        tokens_within_range = token.start - last_token_end <= 3

        # The interleaving tokens *must* be a full stop, a comma, or a whitespace
        interleaving_text = text[last_token_end : token.start]
        tokens_separated_by_allowed_chars = all(
            filter(
                lambda char: True
                if char in SINGLE_ENTITY_ALLOWED_INTERLEAVING_CHARSET
                else False,
                interleaving_text,
            )
        )

        # The current entity type must match with the config (default value is True)
        default_value = split_entities_config[SPLIT_ENTITIES_BY_COMMA]
        split_current_entity_type = split_entities_config.get(
            current_entity_tag, default_value
        )

        return (
            tokens_within_range
            and tokens_separated_by_allowed_chars
            and not split_current_entity_type
        )

    @staticmethod
    def get_tag_for(tags: Dict[Text, List[Text]], tag_name: Text, idx: int) -> Text:
        """Get the value of the given tag name from the list of tags.

        Args:
            tags: Mapping of tag name to list of tags;
            tag_name: The tag name of interest.
            idx: The index position of the tag.

        Returns:
            The tag value.
        """
        if tag_name in tags:
            return tags[tag_name][idx]
        return NO_ENTITY_TAG

    @staticmethod
    def _create_new_entity(
        tag_names: List[Text],
        entity_tag: Text,
        group_tag: Text,
        role_tag: Text,
        token: Token,
        idx: int,
        confidences: Optional[Dict[Text, List[float]]] = None,
    ) -> Dict[Text, Any]:
        """Create a new entity.

        Args:
            tag_names: The tag names to include in the entity.
            entity_tag: The entity type value.
            group_tag: The entity group value.
            role_tag: The entity role value.
            token: The token.
            confidence: The confidence value.

        Returns:
            Created entity.
        """
        entity = {
            ENTITY_ATTRIBUTE_TYPE: entity_tag,
            ENTITY_ATTRIBUTE_START: token.start,
            ENTITY_ATTRIBUTE_END: token.end,
        }

        if confidences is not None:
            entity[ENTITY_ATTRIBUTE_CONFIDENCE_TYPE] = confidences[
                ENTITY_ATTRIBUTE_TYPE
            ][idx]

        if ENTITY_ATTRIBUTE_ROLE in tag_names and role_tag != NO_ENTITY_TAG:
            entity[ENTITY_ATTRIBUTE_ROLE] = role_tag
            if confidences is not None:
                entity[ENTITY_ATTRIBUTE_CONFIDENCE_ROLE] = confidences[
                    ENTITY_ATTRIBUTE_ROLE
                ][idx]
        if ENTITY_ATTRIBUTE_GROUP in tag_names and group_tag != NO_ENTITY_TAG:
            entity[ENTITY_ATTRIBUTE_GROUP] = group_tag
            if confidences is not None:
                entity[ENTITY_ATTRIBUTE_CONFIDENCE_GROUP] = confidences[
                    ENTITY_ATTRIBUTE_GROUP
                ][idx]

        return entity

    @staticmethod
    def check_correct_entity_annotations(training_data: TrainingData) -> None:
        """Check if entities are correctly annotated in the training data.

        If the start and end values of an entity do not match any start and end values
        of the respected token, we define an entity as misaligned and log a warning.

        Args:
            training_data: The training data.
        """
        for example in training_data.entity_examples:
            entity_boundaries = [
                (entity[ENTITY_ATTRIBUTE_START], entity[ENTITY_ATTRIBUTE_END])
                for entity in example.get(ENTITIES)
            ]
            token_start_positions = [t.start for t in example.get(TOKENS_NAMES[TEXT])]
            token_end_positions = [t.end for t in example.get(TOKENS_NAMES[TEXT])]

            for entity_start, entity_end in entity_boundaries:
                if (
                    entity_start not in token_start_positions
                    or entity_end not in token_end_positions
                ):
                    entities_repr = [
                        (
                            entity[ENTITY_ATTRIBUTE_START],
                            entity[ENTITY_ATTRIBUTE_END],
                            entity[ENTITY_ATTRIBUTE_VALUE],
                        )
                        for entity in example.get(ENTITIES)
                    ]
                    tokens_repr = [
                        (t.start, t.end, t.text)
                        for t in example.get(TOKENS_NAMES[TEXT])
                    ]
                    rasa.shared.utils.io.raise_warning(
                        f"Misaligned entity annotation in message "
                        f"'{example.get(TEXT)}' with intent '{example.get(INTENT)}'. "
                        f"Make sure the start and end values of entities "
                        f"({entities_repr}) in the training "
                        f"data match the token boundaries ({tokens_repr}). "
                        "Common causes: \n  1) entities include trailing whitespaces "
                        "or punctuation"
                        "\n  2) the tokenizer gives an unexpected result, due to "
                        "languages such as Chinese that don't use whitespace for word "
                        "separation",
                        docs=DOCS_URL_TRAINING_DATA_NLU,
                    )
                    break










logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from sklearn_crfsuite import CRF



class CRFToken:
    def __init__(
        self,
        text: Text,
        pos_tag: Text,
        pattern: Dict[Text, Any],
        dense_features: np.ndarray,
        entity_tag: Text,
        entity_role_tag: Text,
        entity_group_tag: Text,
    ):
        self.text = text
        self.pos_tag = pos_tag
        self.pattern = pattern
        self.dense_features = dense_features
        self.entity_tag = entity_tag
        self.entity_role_tag = entity_role_tag
        self.entity_group_tag = entity_group_tag


class CRFEntityExtractorOptions(str, Enum):
    """Features that can be used for the 'CRFEntityExtractor'."""

    PATTERN = "pattern"
    LOW = "low"
    TITLE = "title"
    PREFIX5 = "prefix5"
    PREFIX2 = "prefix2"
    SUFFIX5 = "suffix5"
    SUFFIX3 = "suffix3"
    SUFFIX2 = "suffix2"
    SUFFIX1 = "suffix1"
    BIAS = "bias"
    POS = "pos"
    POS2 = "pos2"
    UPPER = "upper"
    DIGIT = "digit"
    TEXT_DENSE_FEATURES = "text_dense_features"
    ENTITY = "entity"


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR, is_trainable=True
)
class CRFEntityExtractor(GraphComponent, EntityExtractorMixin):
    """Implements conditional random fields (CRF) to do named entity recognition."""

    CONFIG_FEATURES = "features"

    function_dict: Dict[Text, Callable[[CRFToken], Any]] = {  # noqa: RUF012
        CRFEntityExtractorOptions.LOW: lambda crf_token: crf_token.text.lower(),
        CRFEntityExtractorOptions.TITLE: lambda crf_token: crf_token.text.istitle(),
        CRFEntityExtractorOptions.PREFIX5: lambda crf_token: crf_token.text[:5],
        CRFEntityExtractorOptions.PREFIX2: lambda crf_token: crf_token.text[:2],
        CRFEntityExtractorOptions.SUFFIX5: lambda crf_token: crf_token.text[-5:],
        CRFEntityExtractorOptions.SUFFIX3: lambda crf_token: crf_token.text[-3:],
        CRFEntityExtractorOptions.SUFFIX2: lambda crf_token: crf_token.text[-2:],
        CRFEntityExtractorOptions.SUFFIX1: lambda crf_token: crf_token.text[-1:],
        CRFEntityExtractorOptions.BIAS: lambda _: "bias",
        CRFEntityExtractorOptions.POS: lambda crf_token: crf_token.pos_tag,
        CRFEntityExtractorOptions.POS2: lambda crf_token: crf_token.pos_tag[:2]
        if crf_token.pos_tag is not None
        else None,
        CRFEntityExtractorOptions.UPPER: lambda crf_token: crf_token.text.isupper(),
        CRFEntityExtractorOptions.DIGIT: lambda crf_token: crf_token.text.isdigit(),
        CRFEntityExtractorOptions.PATTERN: lambda crf_token: crf_token.pattern,
        CRFEntityExtractorOptions.TEXT_DENSE_FEATURES: (
            lambda crf_token: CRFEntityExtractor._convert_dense_features_for_crfsuite(  # noqa: E501
                crf_token
            )
        ),
        CRFEntityExtractorOptions.ENTITY: lambda crf_token: crf_token.entity_tag,
    }

    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        return [Tokenizer]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {
            # BILOU_flag determines whether to use BILOU tagging or not.
            # More rigorous however requires more examples per entity
            # rule of thumb: use only if more than 100 egs. per entity
            BILOU_FLAG: True,
            # Split entities by comma, this makes sense e.g. for a list of ingredients
            # in a recipie, but it doesn't make sense for the parts of an address
            SPLIT_ENTITIES_BY_COMMA: True,
            # crf_features is [before, token, after] array with before, token,
            # after holding keys about which features to use for each token,
            # for example, 'title' in array before will have the feature
            # "is the preceding token in title case?"
            # POS features require SpacyTokenizer
            # pattern feature require RegexFeaturizer
            CRFEntityExtractor.CONFIG_FEATURES: [
                [
                    CRFEntityExtractorOptions.LOW,
                    CRFEntityExtractorOptions.TITLE,
                    CRFEntityExtractorOptions.UPPER,
                ],
                [
                    CRFEntityExtractorOptions.LOW,
                    CRFEntityExtractorOptions.BIAS,
                    CRFEntityExtractorOptions.PREFIX5,
                    CRFEntityExtractorOptions.PREFIX2,
                    CRFEntityExtractorOptions.SUFFIX5,
                    CRFEntityExtractorOptions.SUFFIX3,
                    CRFEntityExtractorOptions.SUFFIX2,
                    CRFEntityExtractorOptions.UPPER,
                    CRFEntityExtractorOptions.TITLE,
                    CRFEntityExtractorOptions.DIGIT,
                    CRFEntityExtractorOptions.PATTERN,
                ],
                [
                    CRFEntityExtractorOptions.LOW,
                    CRFEntityExtractorOptions.TITLE,
                    CRFEntityExtractorOptions.UPPER,
                ],
            ],
            # The maximum number of iterations for optimization algorithms.
            "max_iterations": 50,
            # weight of the L1 regularization
            "L1_c": 0.1,
            # weight of the L2 regularization
            "L2_c": 0.1,
            # Name of dense featurizers to use.
            # If list is empty all available dense features are used.
            "featurizers": [],
        }

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        entity_taggers: Optional[Dict[Text, "CRF"]] = None,
    ) -> None:
        """Creates an instance of entity extractor."""
        self.component_config = config
        self._model_storage = model_storage
        self._resource = resource

        self.entity_taggers = entity_taggers

        self.crf_order = [
            ENTITY_ATTRIBUTE_TYPE,
            ENTITY_ATTRIBUTE_ROLE,
            ENTITY_ATTRIBUTE_GROUP,
        ]

        self._validate_configuration()

        self.split_entities_config = rasa.utils.train_utils.init_split_entities(
            config[SPLIT_ENTITIES_BY_COMMA], SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE
        )

    def _validate_configuration(self) -> None:
        if len(self.component_config.get(self.CONFIG_FEATURES, [])) % 2 != 1:
            raise ValueError(
                "Need an odd number of crf feature lists to have a center word."
            )

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> CRFEntityExtractor:
        """Creates a new untrained component (see parent class for full docstring)."""
        return cls(config, model_storage, resource)

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["sklearn_crfsuite", "sklearn"]

    def train(self, training_data: TrainingData) -> Resource:
        """Trains the extractor on a data set."""
        # checks whether there is at least one
        # example with an entity annotation
        training_data.entities
        if not training_data.entity_examples:
            logger.debug(
                "No training examples with entities present. Skip training"
                "of 'CRFEntityExtractor'."
            )
            return self._resource

        self.check_correct_entity_annotations(training_data)

        if self.component_config[BILOU_FLAG]:
            bilou_utils.apply_bilou_schema(training_data)

        # only keep the CRFs for tags we actually have training data for
        self._update_crf_order(training_data)
      
        # filter out pre-trained entity examples
        entity_examples = self.filter_trainable_entities(training_data.nlu_examples)





        entity_examples = [
            message
            for message in entity_examples
            if message.features_present(
                attribute=TEXT, featurizers=self.component_config.get(FEATURIZERS)
            )
        ]
        dataset = [self._convert_to_crf_tokens(example) for example in entity_examples]

        self._train_model(dataset)

        self.persist()

        return self._resource

    def _update_crf_order(self, training_data: TrainingData) -> None:
        """Train only CRFs we actually have training data for."""
        _crf_order = []

        for tag_name in self.crf_order:
            if tag_name == ENTITY_ATTRIBUTE_TYPE and training_data.entities:
                _crf_order.append(ENTITY_ATTRIBUTE_TYPE)
            elif tag_name == ENTITY_ATTRIBUTE_ROLE and training_data.entity_roles:
                _crf_order.append(ENTITY_ATTRIBUTE_ROLE)
            elif tag_name == ENTITY_ATTRIBUTE_GROUP and training_data.entity_groups:
                _crf_order.append(ENTITY_ATTRIBUTE_GROUP)

        self.crf_order = _crf_order

    def process(self, messages: List[Message]) -> List[Message]:
        """Augments messages with entities."""
        for message in messages:
            entities = self.extract_entities(message)
            entities = self.add_extractor_name(entities)
            message.set(
                ENTITIES, message.get(ENTITIES, []) + entities, add_to_output=True
            )

        return messages

    def extract_entities(self, message: Message) -> List[Dict[Text, Any]]:
        """Extract entities from the given message using the trained model(s)."""
        if self.entity_taggers is None or not message.features_present(
            attribute=TEXT, featurizers=self.component_config.get(FEATURIZERS)
        ):
            return []

        tokens = message.get(TOKENS_NAMES[TEXT])
        crf_tokens = self._convert_to_crf_tokens(message)

        predictions: Dict[Text, List[Dict[Text, float]]] = {}
        for tag_name, entity_tagger in self.entity_taggers.items():
            # use predicted entity tags as features for second level CRFs
            include_tag_features = tag_name != ENTITY_ATTRIBUTE_TYPE
            if include_tag_features:
                self._add_tag_to_crf_token(crf_tokens, predictions)

            features = self._crf_tokens_to_features(crf_tokens, include_tag_features)
            predictions[tag_name] = entity_tagger.predict_marginals_single(features)

        # convert predictions into a list of tags and a list of confidences
        tags, confidences = self._tag_confidences(tokens, predictions)

        return self.convert_predictions_into_entities(
            message.get(TEXT), tokens, tags, self.split_entities_config, confidences
        )

    def _add_tag_to_crf_token(
        self,
        crf_tokens: List[CRFToken],
        predictions: Dict[Text, List[Dict[Text, float]]],
    ) -> None:
        """Add predicted entity tags to CRF tokens."""
        if ENTITY_ATTRIBUTE_TYPE in predictions:
            _tags, _ = self._most_likely_tag(predictions[ENTITY_ATTRIBUTE_TYPE])
            for tag, token in zip(_tags, crf_tokens):
                token.entity_tag = tag

    def _most_likely_tag(
        self, predictions: List[Dict[Text, float]]
    ) -> Tuple[List[Text], List[float]]:
        """Get the entity tags with the highest confidence.

        Args:
            predictions: list of mappings from entity tag to confidence value

        Returns:
            List of entity tags and list of confidence values.
        """
        _tags = []
        _confidences = []

        for token_predictions in predictions:
            tag = max(token_predictions, key=lambda key: token_predictions[key])
            _tags.append(tag)

            if self.component_config[BILOU_FLAG]:
                # if we are using BILOU flags, we will sum up the prob
                # of the B, I, L and U tags for an entity
                _confidences.append(
                    sum(
                        _confidence
                        for _tag, _confidence in token_predictions.items()
                        if bilou_utils.tag_without_prefix(tag)
                        == bilou_utils.tag_without_prefix(_tag)
                    )
                )
            else:
                _confidences.append(token_predictions[tag])

        return _tags, _confidences

    def _tag_confidences(
        self, tokens: List[Token], predictions: Dict[Text, List[Dict[Text, float]]]
    ) -> Tuple[Dict[Text, List[Text]], Dict[Text, List[float]]]:
        """Get most likely tag predictions with confidence values for tokens."""
        tags = {}
        confidences = {}

        for tag_name, predicted_tags in predictions.items():
            if len(tokens) != len(predicted_tags):
                raise Exception(
                    "Inconsistency in amount of tokens between crfsuite and message"
                )

            _tags, _confidences = self._most_likely_tag(predicted_tags)

            if self.component_config[BILOU_FLAG]:
                _tags, _confidences = bilou_utils.ensure_consistent_bilou_tagging(
                    _tags, _confidences
                )

            confidences[tag_name] = _confidences
            tags[tag_name] = _tags

        return tags, confidences

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> CRFEntityExtractor:
        """Loads trained component (see parent class for full docstring)."""
        import joblib

        try:
            entity_taggers = OrderedDict()
            with model_storage.read_from(resource) as model_dir:
                # We have to load in the same order as we persisted things as otherwise
                # the predictions might be off
                file_names = sorted(model_dir.glob("**/*.pkl"))
                if not file_names:
                    logger.debug(
                        "Failed to load model for 'CRFEntityExtractor'. "
                        "Maybe you did not provide enough training data and "
                        "no model was trained."
                    )
                    return cls(config, model_storage, resource)

                for file_name in file_names:
                    name = file_name.stem[1:]
                    entity_taggers[name] = joblib.load(file_name)

                return cls(config, model_storage, resource, entity_taggers)
        except ValueError:
            logger.warning(
                f"Failed to load {cls.__name__} from model storage. Resource "
                f"'{resource.name}' doesn't exist."
            )
            return cls(config, model_storage, resource)

    def persist(self) -> None:
        """Persist this model into the passed directory."""
        import joblib

        with self._model_storage.write_to(self._resource) as model_dir:
            if self.entity_taggers:
                for idx, (name, entity_tagger) in enumerate(
                    self.entity_taggers.items()
                ):
                    model_file_name = model_dir / f"{idx}{name}.pkl"
                    joblib.dump(entity_tagger, model_file_name)

    def _crf_tokens_to_features(
        self, crf_tokens: List[CRFToken], include_tag_features: bool = False
    ) -> List[Dict[Text, Any]]:
        """Convert the list of tokens into discrete features."""
        configured_features = self.component_config[self.CONFIG_FEATURES]
        sentence_features = []

        for token_idx in range(len(crf_tokens)):
            # the features for the current token include features of the token
            # before and after the current features (if defined in the config)
            # token before (-1), current token (0), token after (+1)
            window_size = len(configured_features)
            half_window_size = window_size // 2
            window_range = range(-half_window_size, half_window_size + 1)

            token_features = self._create_features_for_token(
                crf_tokens,
                token_idx,
                half_window_size,
                window_range,
                include_tag_features,
            )

            sentence_features.append(token_features)

        return sentence_features

    def _create_features_for_token(
        self,
        crf_tokens: List[CRFToken],
        token_idx: int,
        half_window_size: int,
        window_range: range,
        include_tag_features: bool,
    ) -> Dict[Text, Any]:
        """Convert a token into discrete features including words before and after."""
        configured_features = self.component_config[self.CONFIG_FEATURES]
        prefixes = [str(i) for i in window_range]

        token_features = {}

        # iterate over the tokens in the window range (-1, 0, +1) to collect the
        # features for the token at token_idx
        for pointer_position in window_range:
            current_token_idx = token_idx + pointer_position

            if current_token_idx >= len(crf_tokens):
                # token is at the end of the sentence
                token_features["EOS"] = True
            elif current_token_idx < 0:
                # token is at the beginning of the sentence
                token_features["BOS"] = True
            else:
                token = crf_tokens[current_token_idx]

                # get the features to extract for the token we are currently looking at
                current_feature_idx = pointer_position + half_window_size
                features = configured_features[current_feature_idx]

                prefix = prefixes[current_feature_idx]

                # we add the 'entity' feature to include the entity type as features
                # for the role and group CRFs
                # (do not modify features, otherwise we will end up adding 'entity'
                # over and over again, making training very slow)
                additional_features = []
                if include_tag_features:
                    additional_features.append(CRFEntityExtractorOptions.ENTITY)

                for feature in features + additional_features:
                    if feature == CRFEntityExtractorOptions.PATTERN:
                        # add all regexes extracted from the 'RegexFeaturizer' as a
                        # feature: 'pattern_name' is the name of the pattern the user
                        # set in the training data, 'matched' is either 'True' or
                        # 'False' depending on whether the token actually matches the
                        # pattern or not
                        regex_patterns = self.function_dict[feature](token)
                        for pattern_name, matched in regex_patterns.items():
                            token_features[
                                f"{prefix}:{feature}:{pattern_name}"
                            ] = matched
                    else:
                        value = self.function_dict[feature](token)
                        token_features[f"{prefix}:{feature}"] = value

        return token_features

    @staticmethod
    def _crf_tokens_to_tags(crf_tokens: List[CRFToken], tag_name: Text) -> List[Text]:
        """Return the list of tags for the given tag name."""
        if tag_name == ENTITY_ATTRIBUTE_ROLE:
            return [crf_token.entity_role_tag for crf_token in crf_tokens]
        if tag_name == ENTITY_ATTRIBUTE_GROUP:
            return [crf_token.entity_group_tag for crf_token in crf_tokens]

        return [crf_token.entity_tag for crf_token in crf_tokens]

    @staticmethod
    def _pattern_of_token(message: Message, idx: int) -> Dict[Text, bool]:
        """Get the patterns of the token at the given index extracted by the
        'RegexFeaturizer'.

        The 'RegexFeaturizer' adds all patterns listed in the training data to the
        token. The pattern name is mapped to either 'True' (pattern applies to token) or
        'False' (pattern does not apply to token).

        Args:
            message: The message.
            idx: The token index.

        Returns:
            The pattern dict.
        """
        if message.get(TOKENS_NAMES[TEXT]) is not None:
            return message.get(TOKENS_NAMES[TEXT])[idx].get(
                CRFEntityExtractorOptions.PATTERN, {}
            )
        return {}

    def _get_dense_features(self, message: Message) -> Optional[np.ndarray]:
        """Convert dense features to python-crfsuite feature format."""
        features, _ = message.get_dense_features(
            TEXT, self.component_config["featurizers"]
        )

        if features is None:
            return None

        tokens = message.get(TOKENS_NAMES[TEXT])
        if len(tokens) != len(features.features):
            rasa.shared.utils.io.raise_warning(
                f"Number of dense features ({len(features.features)}) for attribute "
                f"'{TEXT}' does not match number of tokens ({len(tokens)}).",
                docs=DOCS_URL_COMPONENTS + "#crfentityextractor",
            )
            return None

        return features.features

    @staticmethod
    def _convert_dense_features_for_crfsuite(
        crf_token: CRFToken,
    ) -> Dict[Text, Dict[Text, float]]:
        """Converts dense features of CRFTokens to dicts for the crfsuite."""
        feature_dict = {
            str(index): token_features
            for index, token_features in enumerate(crf_token.dense_features)
        }
        converted = {"text_dense_features": feature_dict}
        return converted

    def _convert_to_crf_tokens(self, message: Message) -> List[CRFToken]:
        """Take a message and convert it to crfsuite format."""
        crf_format = []
        tokens = message.get(TOKENS_NAMES[TEXT])

        text_dense_features = self._get_dense_features(message)
        tags = self._get_tags(message)

        for i, token in enumerate(tokens):
            pattern = self._pattern_of_token(message, i)
            entity = self.get_tag_for(tags, ENTITY_ATTRIBUTE_TYPE, i)
            group = self.get_tag_for(tags, ENTITY_ATTRIBUTE_GROUP, i)
            role = self.get_tag_for(tags, ENTITY_ATTRIBUTE_ROLE, i)
            pos_tag = token.get(POS_TAG_KEY)
            dense_features = (
                text_dense_features[i] if text_dense_features is not None else []
            )

            crf_format.append(
                CRFToken(
                    text=token.text,
                    pos_tag=pos_tag,
                    entity_tag=entity,
                    entity_group_tag=group,
                    entity_role_tag=role,
                    pattern=pattern,
                    dense_features=dense_features,
                )
            )

        return crf_format

    def _get_tags(self, message: Message) -> Dict[Text, List[Text]]:
        """Get assigned entity tags of message."""
        tokens = message.get(TOKENS_NAMES[TEXT])
        tags = {}

        for tag_name in self.crf_order:
            if self.component_config[BILOU_FLAG]:
                bilou_key = bilou_utils.get_bilou_key_for_tag(tag_name)
                if message.get(bilou_key):
                    _tags = message.get(bilou_key)
                else:
                    _tags = [NO_ENTITY_TAG for _ in tokens]
            else:
                _tags = [
                    determine_token_labels(
                        token, message.get(ENTITIES), attribute_key=tag_name
                    )
                    for token in tokens
                ]
            tags[tag_name] = _tags

        return tags

    def _train_model(self, df_train: List[List[CRFToken]]) -> None:
        """Train the crf tagger based on the training data."""
        import sklearn_crfsuite

        self.entity_taggers = OrderedDict()

        for tag_name in self.crf_order:
            logger.debug(f"Training CRF for '{tag_name}'.")

            # add entity tag features for second level CRFs
            include_tag_features = tag_name != ENTITY_ATTRIBUTE_TYPE
            X_train = (
                self._crf_tokens_to_features(sentence, include_tag_features)
                for sentence in df_train
            )
            y_train = (
                self._crf_tokens_to_tags(sentence, tag_name) for sentence in df_train
            )

            entity_tagger = sklearn_crfsuite.CRF(
                algorithm="lbfgs",
                # coefficient for L1 penalty
                c1=self.component_config["L1_c"],
                # coefficient for L2 penalty
                c2=self.component_config["L2_c"],
                # stop earlier
                max_iterations=self.component_config["max_iterations"],
                # include transitions that are possible, but not observed
                all_possible_transitions=True,
            )
            entity_tagger.fit(X_train, y_train)

            self.entity_taggers[tag_name] = entity_tagger

            logger.debug("Training finished.")
