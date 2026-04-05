"""
Medical Question Answering Dataset Module
Handles loading, preprocessing, and creating datasets for medical QA tasks.
Supports various medical QA datasets like PubMedQA, MedQA, and custom datasets.
"""

import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, AutoTokenizer
import torch
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class QAType(Enum):
    """Types of medical question answering tasks."""
    MULTIPLE_CHOICE = "multiple_choice"
    OPEN_ENDED = "open_ended"
    EXTRACTIVE = "extractive"
    YES_NO = "yes_no"
    REASONING = "reasoning"

@dataclass
class QAExample:
    """
    Data class representing a single medical QA example.
    Contains question, context, answer, and metadata.
    """
    question_id: str
    question: str
    context: str
    answer: str
    qa_type: QAType
    options: List[str] = field(default_factory=list)
    correct_option: Optional[str] = None
    evidence_text: Optional[str] = None
    source: str = "unknown"
    difficulty: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'question_id': self.question_id,
            'question': self.question,
            'context': self.context,
            'answer': self.answer,
            'qa_type': self.qa_type.value,
            'options': self.options,
            'correct_option': self.correct_option,
            'evidence_text': self.evidence_text,
            'source': self.source,
            'difficulty': self.difficulty,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QAExample':
        """Create from dictionary."""
        return cls(
            question_id=data['question_id'],
            question=data['question'],
            context=data.get('context', ''),
            answer=data['answer'],
            qa_type=QAType(data.get('qa_type', 'open_ended')),
            options=data.get('options', []),
            correct_option=data.get('correct_option'),
            evidence_text=data.get('evidence_text'),
            source=data.get('source', 'unknown'),
            difficulty=data.get('difficulty', 0.5),
            metadata=data.get('metadata', {})
        )


class MedicalQADataset(Dataset):
    """
    PyTorch Dataset for medical question answering.
    Handles tokenization and formatting for transformer models.
    """
    
    def __init__(
        self,
        examples: List[QAExample],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        include_context: bool = True,
        qa_type: Optional[QAType] = None
    ):
        """
        Initialize medical QA dataset.
        
        Args:
            examples: List of QAExample objects
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            include_context: Whether to include context in input
            qa_type: Filter by QA type if specified
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_context = include_context
        
        # Filter by QA type if specified
        if qa_type:
            self.examples = [ex for ex in examples if ex.qa_type == qa_type]
        else:
            self.examples = examples
        
        logger.info(f"Loaded {len(self.examples)} examples")
        
        # Pre-process all examples for faster training
        self.processed_examples = []
        self._preprocess_all()
    
    def _preprocess_all(self):
        """Pre-process all examples during initialization."""
        for idx, example in enumerate(self.examples):
            processed = self._process_example(example)
            self.processed_examples.append(processed)
    
    def _process_example(self, example: QAExample) -> Dict[str, Any]:
        """
        Process a single example into model inputs.
        
        Args:
            example: QAExample object
            
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        # Build input text
        input_parts = []
        
        if self.include_context and example.context:
            input_parts.append(f"Context: {example.context}")
        
        input_parts.append(f"Question: {example.question}")
        
        # Add options for multiple choice
        if example.qa_type == QAType.MULTIPLE_CHOICE and example.options:
            options_text = " Options: " + " ".join(
                [f"({chr(65+i)}) {opt}" for i, opt in enumerate(example.options)]
            )
            input_parts.append(options_text)
        
        input_text = " ".join(input_parts)
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Process answer based on QA type
        if example.qa_type == QAType.MULTIPLE_CHOICE:
            # Convert answer to index
            if example.correct_option:
                answer_idx = example.options.index(example.correct_option)
            else:
                answer_idx = -1
            labels = torch.tensor(answer_idx)
        elif example.qa_type == QAType.YES_NO:
            # Convert yes/no to binary
            labels = torch.tensor(1 if example.answer.lower() == 'yes' else 0)
        else:
            # For open-ended, tokenize answer
            answer_tokens = self.tokenizer(
                example.answer,
                truncation=True,
                max_length=128,
                padding=True,
                return_tensors="pt"
            )
            labels = answer_tokens['input_ids'].squeeze(0)
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': labels,
            'example': example
        }
    
    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.processed_examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a processed example by index."""
        return self.processed_examples[idx]


class PubMedQALoader:
    """
    Loader for PubMedQA dataset.
    PubMedQA contains biomedical research articles with questions and answers.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize PubMedQA loader.
        
        Args:
            data_path: Path to PubMedQA dataset files
        """
        self.data_path = data_path
        self.examples = []
    
    def load_from_file(self, filepath: Path) -> List[QAExample]:
        """
        Load PubMedQA dataset from file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            List of QAExample objects
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        examples = []
        for qid, qdata in data.items():
            # Create QA example
            example = QAExample(
                question_id=qid,
                question=qdata.get('question', ''),
                context=qdata.get('context', ''),
                answer=qdata.get('answer', ''),
                qa_type=QAType.OPEN_ENDED,
                options=[],
                source='pubmedqa',
                metadata={
                    'final_decision': qdata.get('final_decision', ''),
                    'long_answer': qdata.get('long_answer', '')
                }
            )
            examples.append(example)
        
        logger.info(f"Loaded {len(examples)} examples from PubMedQA")
        self.examples.extend(examples)
        return examples


class MedQALoader:
    """
    Loader for MedQA dataset (USMLE-style questions).
    MedQA contains medical board exam questions with multiple choice answers.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize MedQA loader.
        
        Args:
            data_path: Path to MedQA dataset files
        """
        self.data_path = data_path
        self.examples = []
    
    def load_from_file(self, filepath: Path) -> List[QAExample]:
        """
        Load MedQA dataset from file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            List of QAExample objects
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        examples = []
        for item in data:
            # Extract question and options
            question_text = item.get('question', '')
            options = item.get('options', [])
            answer_idx = item.get('answer_idx', 0)
            answer_text = options[answer_idx] if options else ''
            
            example = QAExample(
                question_id=item.get('id', str(len(examples))),
                question=question_text,
                context=item.get('meta_info', {}).get('context', ''),
                answer=answer_text,
                qa_type=QAType.MULTIPLE_CHOICE,
                options=options,
                correct_option=answer_text,
                source='medqa',
                difficulty=item.get('difficulty', 0.5),
                metadata={
                    'subject': item.get('meta_info', {}).get('subject', ''),
                    'year': item.get('meta_info', {}).get('year', '')
                }
            )
            examples.append(example)
        
        logger.info(f"Loaded {len(examples)} examples from MedQA")
        self.examples.extend(examples)
        return examples


class ClinicalQALoader:
    """
    Loader for custom clinical QA dataset.
    Handles questions from clinical notes and medical records.
    """
    
    def __init__(self):
        """Initialize clinical QA loader."""
        self.examples = []
    
    def load_from_csv(self, filepath: Path) -> List[QAExample]:
        """
        Load clinical QA from CSV file.
        
        Expected columns: question, answer, context, qa_type, source
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            List of QAExample objects
        """
        df = pd.read_csv(filepath)
        
        examples = []
        for idx, row in df.iterrows():
            qa_type = row.get('qa_type', 'open_ended')
            
            example = QAExample(
                question_id=f"clinical_{idx}",
                question=row['question'],
                context=row.get('context', ''),
                answer=row['answer'],
                qa_type=QAType(qa_type),
                options=row.get('options', '').split('|') if 'options' in row else [],
                source=row.get('source', 'clinical'),
                difficulty=row.get('difficulty', 0.5),
                metadata={
                    'specialty': row.get('specialty', ''),
                    'encounter_type': row.get('encounter_type', '')
                }
            )
            examples.append(example)
        
        logger.info(f"Loaded {len(examples)} examples from clinical QA CSV")
        self.examples.extend(examples)
        return examples
    
    def load_from_jsonl(self, filepath: Path) -> List[QAExample]:
        """
        Load clinical QA from JSONL file.
        
        Args:
            filepath: Path to JSONL file
            
        Returns:
            List of QAExample objects
        """
        examples = []
        
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f):
                data = json.loads(line.strip())
                example = QAExample.from_dict(data)
                examples.append(example)
        
        logger.info(f"Loaded {len(examples)} examples from clinical QA JSONL")
        self.examples.extend(examples)
        return examples


class MedicalQADataProcessor:
    """
    Process and augment medical QA datasets for training.
    Handles data cleaning, augmentation, and splitting.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        """
        Initialize medical QA data processor.
        
        Args:
            tokenizer: HuggingFace tokenizer for text processing
        """
        self.tokenizer = tokenizer
        
    def clean_question(self, question: str) -> str:
        """
        Clean and normalize question text.
        
        Args:
            question: Raw question text
            
        Returns:
            Cleaned question
        """
        # Remove extra whitespace
        question = ' '.join(question.split())
        
        # Ensure question ends with question mark
        if not question.endswith('?'):
            question = question + '?'
        
        return question
    
    def clean_answer(self, answer: str) -> str:
        """
        Clean and normalize answer text.
        
        Args:
            answer: Raw answer text
            
        Returns:
            Cleaned answer
        """
        # Remove extra whitespace
        answer = ' '.join(answer.split())
        
        # Capitalize first letter
        if answer:
            answer = answer[0].upper() + answer[1:]
        
        return answer
    
    def augment_questions(
        self,
        examples: List[QAExample],
        augmentation_factor: int = 2
    ) -> List[QAExample]:
        """
        Augment questions by paraphrasing.
        
        Args:
            examples: List of QA examples
            augmentation_factor: How many augmentations per example
            
        Returns:
            Augmented list of examples
        """
        augmented = list(examples)
        
        for example in examples:
            for _ in range(augmentation_factor):
                # Create augmented version
                aug_example = QAExample(
                    question_id=f"{example.question_id}_aug",
                    question=self._paraphrase_question(example.question),
                    context=example.context,
                    answer=example.answer,
                    qa_type=example.qa_type,
                    options=example.options,
                    correct_option=example.correct_option,
                    source=example.source,
                    difficulty=example.difficulty
                )
                augmented.append(aug_example)
        
        logger.info(f"Augmented dataset from {len(examples)} to {len(augmented)} examples")
        return augmented
    
    def _paraphrase_question(self, question: str) -> str:
        """
        Simple paraphrasing by synonym replacement.
        
        Args:
            question: Original question
            
        Returns:
            Paraphrased question
        """
        synonyms = {
            'what': ['which', 'what is'],
            'how': ['in what way', 'how is'],
            'why': ['for what reason', 'why is'],
            'when': ['at what time', 'when is'],
            'where': ['in what location', 'where is']
        }
        
        words = question.lower().split()
        if words and words[0] in synonyms:
            replacement = synonyms[words[0]]
            if isinstance(replacement, list):
                import random
                replacement = random.choice(replacement)
            words[0] = replacement
            question = ' '.join(words)
        
        return question
    
    def create_balanced_dataset(
        self,
        examples: List[QAExample],
        target_size: int
    ) -> List[QAExample]:
        """
        Create balanced dataset by upsampling or downsampling.
        
        Args:
            examples: List of QA examples
            target_size: Desired dataset size
            
        Returns:
            Balanced list of examples
        """
        if len(examples) >= target_size:
            # Downsample
            indices = np.random.choice(len(examples), target_size, replace=False)
            return [examples[i] for i in indices]
        else:
            # Upsample with augmentation
            return self.augment_questions(
                examples,
                augmentation_factor=target_size // len(examples) + 1
            )[:target_size]
    
    def split_dataset(
        self,
        examples: List[QAExample],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ) -> Tuple[List[QAExample], List[QAExample], List[QAExample]]:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            examples: List of QA examples
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train, val, test) examples
        """
        np.random.seed(random_seed)
        indices = np.random.permutation(len(examples))
        
        train_end = int(train_ratio * len(examples))
        val_end = train_end + int(val_ratio * len(examples))
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        train = [examples[i] for i in train_indices]
        val = [examples[i] for i in val_indices]
        test = [examples[i] for i in test_indices]
        
        logger.info(f"Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        return train, val, test
    
    def create_qa_pairs_for_training(
        self,
        examples: List[QAExample],
        include_negative_samples: bool = True
    ) -> List[Tuple[str, str, int]]:
        """
        Create (question, context, answer) pairs for training.
        
        Args:
            examples: List of QA examples
            include_negative_samples: Whether to create negative samples
            
        Returns:
            List of (question, context, label) tuples
        """
        qa_pairs = []
        
        for example in examples:
            # Positive sample
            qa_pairs.append((
                example.question,
                example.context,
                1  # positive label
            ))
            
            # Negative samples
            if include_negative_samples:
                # Create negative by using wrong answer or different context
                other_examples = [ex for ex in examples if ex != example]
                if other_examples:
                    negative = np.random.choice(other_examples)
                    qa_pairs.append((
                        example.question,
                        negative.context,
                        0  # negative label
                    ))
        
        return qa_pairs


def create_medical_qa_dataloaders(
    dataset_path: Path,
    tokenizer_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    batch_size: int = 16,
    max_length: int = 512,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience function to create train, val, test dataloaders from dataset.
    
    Args:
        dataset_path: Path to dataset file
        tokenizer_name: Name of tokenizer to use
        batch_size: Batch size for dataloaders
        max_length: Maximum sequence length
        train_split: Training split proportion
        val_split: Validation split proportion
        test_split: Test split proportion
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Load dataset based on file extension
    if dataset_path.suffix == '.json':
        # Try PubMedQA format
        loader = PubMedQALoader()
        examples = loader.load_from_file(dataset_path)
    elif dataset_path.suffix == '.csv':
        loader = ClinicalQALoader()
        examples = loader.load_from_csv(dataset_path)
    else:
        raise ValueError(f"Unsupported file format: {dataset_path.suffix}")
    
    # Process and split dataset
    processor = MedicalQADataProcessor(tokenizer)
    train_examples, val_examples, test_examples = processor.split_dataset(
        examples, train_split, val_split, test_split
    )
    
    # Create datasets
    train_dataset = MedicalQADataset(train_examples, tokenizer, max_length)
    val_dataset = MedicalQADataset(val_examples, tokenizer, max_length)
    test_dataset = MedicalQADataset(test_examples, tokenizer, max_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader, test_loader


# Example usage
if __name__ == "__main__":
    # Create sample examples
    sample_examples = [
        QAExample(
            question_id="1",
            question="What is the first-line treatment for hypertension?",
            context="Hypertension guidelines recommend lifestyle modifications and medications.",
            answer="ACE inhibitors or ARBs",
            qa_type=QAType.OPEN_ENDED,
            source="clinical"
        ),
        QAExample(
            question_id="2",
            question="Is aspirin indicated for primary prevention of cardiovascular disease?",
            context="Current guidelines recommend aspirin for secondary prevention only.",
            answer="No",
            qa_type=QAType.YES_NO,
            source="clinical"
        ),
        QAExample(
            question_id="3",
            question="Which medication is most appropriate for acute asthma exacerbation?",
            context="Patient presents with wheezing and shortness of breath.",
            options=["Albuterol", "Metformin", "Lisinopril", "Atorvastatin"],
            answer="Albuterol",
            qa_type=QAType.MULTIPLE_CHOICE,
            correct_option="Albuterol",
            source="clinical"
        )
    ]
    
    # Test tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = MedicalQADataset(sample_examples, tokenizer, max_length=256)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample item: {dataset[0]}")
    
    # Test processor
    processor = MedicalQADataProcessor(tokenizer)
    train, val, test = processor.split_dataset(sample_examples)
    print(f"Split: Train={len(train)}, Val={len(val)}, Test={len(test)}")