"""
Reference: 

[1] https://github.com/Cyanogenoid/pytorch-vqa/blob/master/data.py

"""

import json
import os
import os.path
import re
import random

from PIL import Image
import h5py
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import config


def get_loader(train=False, val=False, test=False):
    """Returns a data loader for the desired split """
    assert train + val + test == 1 # need to set exactly one of {train, val, test} to True
    if train:
        questions_path = config.QUESTIONS_TRAIN_FILEPATH
        answers_path = config.ANNOTATIONS_TRAIN_FILEPATH
        image_features_path = config.PREPROCESSED_TRAIN_FILEPATH
        dataset = VQA(
            questions_path, 
            answers_path, 
            image_features_path, 
        )
    elif val:
        questions_path = config.QUESTIONS_VAL_FILEPATH
        answers_path = config.ANNOTATIONS_VAL_FILEPATH
        image_features_path = config.PREPROCESSED_VAL_FILEPATH
        dataset = VQA(
            questions_path, 
            answers_path, 
            image_features_path, 
        )
    else:
        questions_path = config.QUESTIONS_TEST_FILEPATH
        answers_path = None
        image_features_path = config.PREPROCESSED_TEST_FILEPATH
        dataset = VQATest(
            questions_path, 
            image_features_path, 
        )

    loader = data.DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=train, # only shuffle the data in training
        pin_memory=True,
        num_workers=config.DATA_WORKERS, 
    )
    return loader


class VQA(data.Dataset):
    """dataset for image embedding, question index, answer index, true false boolean and item index"""
    def __init__(self, questions_path, answers_path, image_features_path):
        super(VQA, self).__init__()
        with open(questions_path, 'r') as fd:
            questions_json = json.load(fd)
        with open(answers_path, 'r') as fd:
            answers_json = json.load(fd)
        
        self.seed = 17
        random.seed(self.seed)
        
        # q, c and a
        self.qca = self._create_qca_dataset(questions_json, answers_json)

        # v
        self.image_features_path = image_features_path
        self.image_id_to_index = self._create_image_id_to_index()

    @property
    def max_question_length(self):
        if not hasattr(self, '_max_length'):
            self._max_length = max(map(len, self.questions))
        return self._max_length
    
    def _create_image_id_to_index(self):
        """Create a mapping from a abstract image id into the corresponding index into the h5 file"""
        with h5py.File(self.image_features_path, 'r') as features_file:
            image_ids = features_file['ids'][()]
            #print(f"image_ids: {image_ids}")
        image_id_to_index = {id: i for i, id in enumerate(image_ids)}
        #print(f"image_id_to_index: {image_id_to_index}")
        return image_id_to_index
    
    @staticmethod
    def _preprocess_question(q):
        # this is used for normalizing questions
        _special_chars = re.compile('[^a-z0-9 ]*')
        q = q.lower()[:-1]
        return q
    
    @staticmethod
    def _preprocess_choices(c):
        # The only normalization that is applied to both machine generated answers as well as
        # ground truth answers is replacing most punctuation with space (see [0] and [1]).
        # Since potential machine generated answers are just taken from most common answers, applying the other
        # normalizations is not needed, assuming that the human answers are already normalized.
        # [0]: http://visualqa.org/evaluation.html
        # [1]: https://github.com/VT-vision-lab/VQA/blob/3849b1eae04a0ffd83f56ad6f70ebd0767e09e0f/PythonEvaluationTools/vqaEvaluation/vqaEval.py#L96
        # these try to emulate the original normalization scheme for answers
        _period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
        _comma_strip = re.compile(r'(\d)(,)(\d)')
        _punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
        _punctuation = re.compile(r'([{}])'.format(re.escape(_punctuation_chars)))
        _punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(_punctuation_chars))

        if _punctuation.search(c) is None:
            return c
        c = _punctuation_with_a_space.sub('', c)
        if re.search(_comma_strip, c) is not None:
            c = c.replace(',', '')
        c = _punctuation.sub(' ', c)
        c = _period_strip.sub('', c)
        return c.strip()
    
    def _create_qca_dataset(self, question_json, answers_json):
        """Create a list of data points in the form of dictionary containing question, 
        choice and answer triples
        Also assuming the inputs in question_json and answers_json have the same order
        """
        qca = []
        id = 0
        for question_dict, answers_dict in zip(question_json["questions"], answers_json["annotations"]):
            assert(question_dict["image_id"] == answers_dict["image_id"])
            assert(question_dict["question_id"] == answers_dict["question_id"])
            answer = answers_dict["multiple_choice_answer"]
            pos_index = question_dict["multiple_choices"].index(answer)

            # random sample 2 neg examples without replacement
            neg_indices = list(range(len(question_dict["multiple_choices"])))
            neg_indices.pop(pos_index)
            neg_indices = random.sample(neg_indices, k=2)
            all_indices = [pos_index] + neg_indices

            for choice_index in all_indices:
                choice = question_dict["multiple_choices"][choice_index]
                qca_i = {
                    "question": VQA._preprocess_question(question_dict["question"]), 
                    "choice": VQA._preprocess_choices(choice), 
                    "answer": 1 if choice == answer else 0, 
                    "id": id, 
                    "image_id": question_dict["image_id"], 
                    "question_id": question_dict["question_id"], 
                }
                id += 1
                qca.append(qca_i)
        return qca
    
    def _load_image(self, image_id):
        if not hasattr(self, 'features_file'):
            # Loading the h5 file has to be done here and not in __init__ because when the DataLoader
            # forks for multiple works, every child would use the same file object and fail
            # Having multiple readers using different file objects is fine though, so we just init in here.
            self.features_file = h5py.File(self.image_features_path, 'r')
        index = self.image_id_to_index[image_id]
        dataset = self.features_file["features"]
        img = dataset[index].astype('float32')
        return torch.from_numpy(img)
    
    def __len__(self):
        return len(self.qca)
    
    def __getitem__(self, item):
        qca_dict = self.qca[item]
        q = qca_dict["question"]
        c = qca_dict["choice"]
        a = qca_dict["answer"]
        id = qca_dict["id"]
        image_id = qca_dict["image_id"]
        v = self._load_image(image_id)
        q_id = qca_dict["question_id"]
        return v, q, c, a, id, q_id


class VQATest(VQA):
    def __init__(self, questions_path, image_features_path):
        super(VQA, self).__init__()
        with open(questions_path, 'r') as fd:
            questions_json = json.load(fd)
        
        # q, c and a
        self.qc = self._create_qc_dataset(questions_json)

        # v
        self.image_features_path = image_features_path
        self.image_id_to_index = self._create_image_id_to_index()

    def _create_qc_dataset(self, question_json):
        """Create a list of data points in the form of dictionary containing question, 
        choice tuples
        """
        qc = []
        id = 0
        for question_dict in question_json["questions"]:
            for choice in question_json["multiple_choices"]:
                qca_i = {
                    "question": VQA._preprocess_question(question_dict["question"]), 
                    "choice": VQA._preprocess_choices(choice), 
                    "id": id, 
                    "image_id": question_dict["image_id"], 
                    "question_id": question_dict["question_id"], 
                }
                id += 1
                qc.append(qca_i)
        return qc
    
    def __len__(self):
        return len(self.qc)
    
    def __getitem__(self, item):
        qc_dict = self.qc[item]
        q = qc_dict["question"]
        c = qc_dict["choice"]
        id = qc_dict["id"]
        image_id = qc_dict["image_id"]
        v = self._load_image(image_id)
        q_id = qc_dict["question_id"]
        return v, q, c, id, q_id


class AbstractImages(data.Dataset):
    """dataset for Abstract images located in a folder on the filesystem"""
    def __init__(self, path, transform=None):
        super(AbstractImages, self).__init__()
        self.path = path
        self.id_to_filename = self._find_images()
        self.sorted_ids = sorted(self.id_to_filename.keys())
        print(f'found {len(self)} images in {self.path}')
        self.transform = transform

    def _find_images(self):
        id_to_filename = {}
        for filename in os.listdir(self.path):
            if not filename.endswith('png'):
                continue
            id_and_extension = filename.split('_')[-1]
            id = int(id_and_extension.split('.')[0])
            id_to_filename[id] = filename
        return id_to_filename

    def __getitem__(self, item):
        id = self.sorted_ids[item]
        path = os.path.join(self.path, self.id_to_filename[id])
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return id, img
    
    def __len__(self):
        return len(self.sorted_ids)
    