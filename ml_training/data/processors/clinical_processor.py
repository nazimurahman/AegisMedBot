"""
Clinical Data Processor Module
Handles preprocessing and feature extraction from clinical text and structured medical data
for training machine learning models in healthcare applications.
"""

import re
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from transformers import PreTrainedTokenizer, AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader

# Configure logging for monitoring processing steps
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClinicalTextProcessor:
    """
    Process and clean clinical text data for machine learning models.
    Handles medical terminology normalization, abbreviation expansion,
    and structured feature extraction from unstructured clinical notes.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        max_length: int = 512,
        lowercase: bool = True,
        remove_stopwords: bool = False
    ):
        """
        Initialize the clinical text processor.
        
        Args:
            model_name: Name of the pretrained model for tokenization
            max_length: Maximum sequence length for tokenization
            lowercase: Whether to convert text to lowercase
            remove_stopwords: Whether to remove common stopwords
        """
        self.model_name = model_name
        self.max_length = max_length
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        
        # Load medical tokenizer from HuggingFace
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Successfully loaded tokenizer: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            # Fallback to basic tokenizer if medical one fails
            from transformers import BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            logger.warning("Using fallback BERT tokenizer")
        
        # Define medical abbreviation mapping for expansion
        self.abbreviation_map = self._load_medical_abbreviations()
        
        # Define medical stopwords (words that don't carry clinical meaning)
        self.medical_stopwords = {
            'patient', 'doctor', 'nurse', 'hospital', 'clinic', 'department',
            'room', 'floor', 'unit', 'time', 'day', 'week', 'month', 'year'
        }
    
    def _load_medical_abbreviations(self) -> Dict[str, str]:
        """
        Load mapping of common medical abbreviations to their full forms.
        This helps normalize clinical text for better model understanding.
        
        Returns:
            Dictionary mapping abbreviations to expanded forms
        """
        return {
            'htn': 'hypertension',
            'dm': 'diabetes mellitus',
            'copd': 'chronic obstructive pulmonary disease',
            'chf': 'congestive heart failure',
            'mi': 'myocardial infarction',
            'cva': 'cerebrovascular accident',
            'tia': 'transient ischemic attack',
            'uti': 'urinary tract infection',
            'uri': 'upper respiratory infection',
            'gi': 'gastrointestinal',
            'gu': 'genitourinary',
            'neuro': 'neurological',
            'cardio': 'cardiovascular',
            'pulm': 'pulmonary',
            'renal': 'kidney',
            'hepatic': 'liver',
            'dx': 'diagnosis',
            'rx': 'prescription',
            'hx': 'history',
            'fx': 'fracture',
            'bid': 'twice daily',
            'tid': 'three times daily',
            'qid': 'four times daily',
            'prn': 'as needed',
            'po': 'by mouth',
            'iv': 'intravenous',
            'im': 'intramuscular',
            'sc': 'subcutaneous',
            'sl': 'sublingual'
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize clinical text for processing.
        
        Args:
            text: Raw clinical text
            
        Returns:
            Cleaned and normalized text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase if specified
        if self.lowercase:
            text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s\.\,\-\;\(\)]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Expand medical abbreviations
        words = text.split()
        expanded_words = []
        for word in words:
            # Check if word matches any abbreviation
            clean_word = word.strip('.,;:()')
            if clean_word in self.abbreviation_map:
                expanded_words.append(self.abbreviation_map[clean_word])
            else:
                expanded_words.append(word)
        
        text = ' '.join(expanded_words)
        
        # Remove medical stopwords if specified
        if self.remove_stopwords:
            words = text.split()
            words = [w for w in words if w not in self.medical_stopwords]
            text = ' '.join(words)
        
        return text.strip()
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical entities from clinical text using pattern matching.
        This is a simplified version; in production, use a medical NER model.
        
        Args:
            text: Clinical text
            
        Returns:
            Dictionary of extracted medical entities
        """
        entities = {
            'medications': [],
            'symptoms': [],
            'diagnoses': [],
            'procedures': [],
            'lab_tests': []
        }
        
        # Medication patterns
        medication_patterns = [
            r'\b(aspirin|ibuprofen|acetaminophen|paracetamol|metformin|lisinopril|atorvastatin)\b',
            r'\b(antibiotic|antiviral|antifungal|analgesic|antihypertensive|antidiabetic)\b'
        ]
        
        # Symptom patterns
        symptom_patterns = [
            r'\b(pain|fever|cough|dyspnea|nausea|vomiting|fatigue|dizziness)\b',
            r'\b(chest pain|shortness of breath|abdominal pain|headache)\b'
        ]
        
        # Diagnosis patterns
        diagnosis_patterns = [
            r'\b(hypertension|diabetes|asthma|pneumonia|heart failure)\b',
            r'\b(cancer|tumor|neoplasm|infection|inflammation)\b'
        ]
        
        # Extract using regex patterns
        for pattern in medication_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['medications'].extend(matches)
        
        for pattern in symptom_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['symptoms'].extend(matches)
        
        for pattern in diagnosis_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['diagnoses'].extend(matches)
        
        # Remove duplicates and clean
        for key in entities:
            entities[key] = list(set([item.lower() for item in entities[key]]))
        
        return entities
    
    def tokenize_for_training(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize clinical texts for transformer model training.
        
        Args:
            texts: List of text strings
            labels: Optional list of labels for classification
            padding: Whether to pad sequences to max_length
            truncation: Whether to truncate sequences longer than max_length
            return_tensors: Format to return tensors in
            
        Returns:
            Dictionary with input_ids, attention_mask, and optional labels
        """
        # First clean all texts
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # Tokenize with the medical tokenizer
        encodings = self.tokenizer(
            cleaned_texts,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length,
            return_tensors=return_tensors
        )
        
        # Add labels if provided
        if labels is not None:
            encodings['labels'] = torch.tensor(labels)
        
        return encodings
    
    def create_training_dataset(
        self,
        texts: List[str],
        labels: List[int],
        validation_split: float = 0.2,
        test_split: float = 0.1,
        random_state: int = 42
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Create train, validation, and test datasets for model training.
        
        Args:
            texts: List of input texts
            labels: List of corresponding labels
            validation_split: Proportion for validation
            test_split: Proportion for test
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of train, validation, and test encoded datasets
        """
        # First split into train+val and test
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts, labels,
            test_size=test_split,
            random_state=random_state,
            stratify=labels
        )
        
        # Then split train+val into train and validation
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts, train_val_labels,
            test_size=validation_split / (1 - test_split),
            random_state=random_state,
            stratify=train_val_labels
        )
        
        logger.info(f"Dataset split - Train: {len(train_texts)}, "
                   f"Val: {len(val_texts)}, Test: {len(test_texts)}")
        
        # Tokenize all splits
        train_encodings = self.tokenize_for_training(train_texts, train_labels)
        val_encodings = self.tokenize_for_training(val_texts, val_labels)
        test_encodings = self.tokenize_for_training(test_texts, test_labels)
        
        return train_encodings, val_encodings, test_encodings


class ClinicalDataset(Dataset):
    """
    PyTorch Dataset class for clinical text data.
    Handles loading and batching of tokenized clinical texts.
    """
    
    def __init__(
        self,
        encodings: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None
    ):
        """
        Initialize the clinical dataset.
        
        Args:
            encodings: Dictionary of tokenized inputs
            labels: Optional tensor of labels
        """
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing input tensors and label
        """
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx]
        }
        
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        
        return item


class EHRDataProcessor:
    """
    Process Electronic Health Record structured data for machine learning.
    Handles patient demographics, vitals, lab results, and temporal features.
    """
    
    def __init__(self, time_window_days: int = 30):
        """
        Initialize EHR data processor.
        
        Args:
            time_window_days: Time window for aggregating historical data
        """
        self.time_window_days = time_window_days
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
    
    def extract_patient_features(
        self,
        patient_data: Dict[str, Any]
    ) -> np.ndarray:
        """
        Extract numerical features from patient EHR data.
        
        Args:
            patient_data: Dictionary containing patient information
            
        Returns:
            Numpy array of extracted features
        """
        features = []
        
        # Demographics
        if 'date_of_birth' in patient_data:
            dob = datetime.strptime(patient_data['date_of_birth'], '%Y-%m-%d')
            age = (datetime.now() - dob).days / 365.25
            features.append(age)
        else:
            features.append(0)
        
        # Gender encoding (one-hot)
        gender_map = {'M': [1, 0, 0], 'F': [0, 1, 0], 'O': [0, 0, 1]}
        gender = patient_data.get('gender', 'U')
        features.extend(gender_map.get(gender, [0, 0, 0]))
        
        # Vital signs with normalization
        vital_ranges = {
            'heart_rate': (60, 100),
            'systolic_bp': (90, 140),
            'diastolic_bp': (60, 90),
            'temperature': (36.1, 37.2),
            'oxygen_saturation': (95, 100),
            'respiratory_rate': (12, 20)
        }
        
        for vital, (low, high) in vital_ranges.items():
            value = patient_data.get(vital, None)
            if value is not None:
                # Normalize to [0, 1] range
                normalized = (value - low) / (high - low)
                features.append(max(0, min(1, normalized)))
            else:
                features.append(0.5)  # Default to middle value
        
        # Comorbidity count
        comorbidities = patient_data.get('chronic_conditions', [])
        features.append(len(comorbidities))
        
        # Previous admissions count
        previous_admissions = patient_data.get('admission_count', 0)
        features.append(min(previous_admissions, 10) / 10)  # Normalize
        
        # Lab results (normalized)
        lab_results = patient_data.get('lab_results', {})
        normal_ranges = {
            'glucose': (70, 140),
            'creatinine': (0.6, 1.2),
            'sodium': (135, 145),
            'potassium': (3.5, 5.0),
            'hemoglobin': (12, 16),
            'wbc': (4.5, 11.0)
        }
        
        for lab, (low, high) in normal_ranges.items():
            value = lab_results.get(lab, None)
            if value is not None:
                normalized = (value - low) / (high - low)
                features.append(max(0, min(1, normalized)))
            else:
                features.append(0.5)
        
        return np.array(features)
    
    def create_temporal_features(
        self,
        time_series_data: pd.DataFrame,
        patient_id_col: str,
        timestamp_col: str,
        value_col: str
    ) -> np.ndarray:
        """
        Create temporal features from time series data.
        
        Args:
            time_series_data: DataFrame with time series measurements
            patient_id_col: Column containing patient identifiers
            timestamp_col: Column containing timestamps
            value_col: Column containing measurement values
            
        Returns:
            Array of temporal features including trends and statistics
        """
        temporal_features = []
        
        for patient_id in time_series_data[patient_id_col].unique():
            patient_data = time_series_data[
                time_series_data[patient_id_col] == patient_id
            ].sort_values(timestamp_col)
            
            values = patient_data[value_col].values
            
            if len(values) > 0:
                # Statistical features
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                range_val = max_val - min_val
                
                # Trend features
                if len(values) > 1:
                    trend = (values[-1] - values[0]) / len(values)
                else:
                    trend = 0
                
                # Rate of change
                if len(values) > 1:
                    rates = np.diff(values)
                    mean_rate = np.mean(rates)
                    max_rate = np.max(rates)
                else:
                    mean_rate = 0
                    max_rate = 0
                
                temporal_features.append([
                    mean_val, std_val, min_val, max_val, range_val,
                    trend, mean_rate, max_rate
                ])
            else:
                temporal_features.append([0] * 8)
        
        return np.array(temporal_features)
    
    def create_sequence_data(
        self,
        patient_data: pd.DataFrame,
        patient_id_col: str,
        time_col: str,
        feature_cols: List[str],
        target_col: str,
        sequence_length: int = 24,
        prediction_horizon: int = 6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequence data for LSTM or Transformer time series models.
        
        Args:
            patient_data: DataFrame with sequential patient data
            patient_id_col: Column identifying patients
            time_col: Column with timestamps
            feature_cols: Columns to use as features
            target_col: Column to predict
            sequence_length: Number of time steps in input sequence
            prediction_horizon: Steps ahead to predict
            
        Returns:
            Tuple of (X sequences, y targets) arrays
        """
        X_sequences = []
        y_targets = []
        
        # Group by patient to handle multiple patients
        for patient_id in patient_data[patient_id_col].unique():
            patient_df = patient_data[
                patient_data[patient_id_col] == patient_id
            ].sort_values(time_col)
            
            # Extract feature and target values
            feature_values = patient_df[feature_cols].values
            target_values = patient_df[target_col].values
            
            # Create sliding window sequences
            for i in range(len(feature_values) - sequence_length - prediction_horizon + 1):
                X_seq = feature_values[i:i + sequence_length]
                y_target = target_values[i + sequence_length + prediction_horizon - 1]
                
                X_sequences.append(X_seq)
                y_targets.append(y_target)
        
        return np.array(X_sequences), np.array(y_targets)
    
    def fit_scaler(self, feature_matrix: np.ndarray):
        """
        Fit the scaler to feature matrix for normalization.
        
        Args:
            feature_matrix: Array of features to fit scaler on
        """
        self.scaler.fit(feature_matrix)
        self.is_fitted = True
    
    def transform_features(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted scaler.
        
        Args:
            feature_matrix: Array of features to transform
            
        Returns:
            Normalized feature array
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
        return self.scaler.transform(feature_matrix)
    
    def create_cohort_features(
        self,
        patients: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Create feature matrix for a cohort of patients.
        
        Args:
            patients: List of patient dictionaries
            
        Returns:
            DataFrame with extracted features for all patients
        """
        features_list = []
        
        for patient in patients:
            features = self.extract_patient_features(patient)
            features_list.append(features)
        
        feature_matrix = np.array(features_list)
        
        # Create feature names for the DataFrame
        feature_names = [
            'age', 'gender_M', 'gender_F', 'gender_O',
            'heart_rate', 'systolic_bp', 'diastolic_bp',
            'temperature', 'oxygen_saturation', 'respiratory_rate',
            'comorbidity_count', 'previous_admissions',
            'glucose', 'creatinine', 'sodium', 'potassium',
            'hemoglobin', 'wbc'
        ]
        
        return pd.DataFrame(feature_matrix, columns=feature_names)


class MedicalDataAugmenter:
    """
    Augment medical data to increase dataset size and improve model robustness.
    Uses techniques like synonym replacement, back-translation, and noise injection.
    """
    
    def __init__(self, synonym_dict: Optional[Dict[str, List[str]]] = None):
        """
        Initialize medical data augmenter.
        
        Args:
            synonym_dict: Dictionary of medical terms and their synonyms
        """
        # Default medical synonym dictionary
        self.synonym_dict = synonym_dict or {
            'hypertension': ['high blood pressure', 'elevated blood pressure'],
            'diabetes': ['diabetes mellitus', 'sugar disease'],
            'heart attack': ['myocardial infarction', 'cardiac arrest'],
            'stroke': ['cerebrovascular accident', 'brain attack'],
            'pneumonia': ['lung infection', 'respiratory infection'],
            'fever': ['pyrexia', 'high temperature'],
            'pain': ['discomfort', 'ache', 'soreness'],
            'medication': ['drug', 'medicine', 'prescription']
        }
    
    def synonym_replacement(self, text: str, replacement_prob: float = 0.3) -> str:
        """
        Replace medical terms with their synonyms to create variations.
        
        Args:
            text: Input text
            replacement_prob: Probability of replacing a term
            
        Returns:
            Augmented text with synonym replacements
        """
        words = text.split()
        augmented_words = []
        
        for word in words:
            # Check if word or its lowercase version is in synonym dict
            clean_word = word.strip('.,;:()').lower()
            if clean_word in self.synonym_dict and np.random.random() < replacement_prob:
                # Replace with random synonym
                synonym = np.random.choice(self.synonym_dict[clean_word])
                augmented_words.append(synonym)
            else:
                augmented_words.append(word)
        
        return ' '.join(augmented_words)
    
    def random_insertion(self, text: str, insertion_prob: float = 0.1) -> str:
        """
        Insert random medical terms into text to create variations.
        
        Args:
            text: Input text
            insertion_prob: Probability of insertion at each position
            
        Returns:
            Augmented text with random insertions
        """
        medical_terms = list(self.synonym_dict.keys())
        words = text.split()
        
        augmented_words = []
        for word in words:
            augmented_words.append(word)
            if np.random.random() < insertion_prob:
                # Insert random medical term
                term = np.random.choice(medical_terms)
                augmented_words.append(term)
        
        return ' '.join(augmented_words)
    
    def random_deletion(self, text: str, deletion_prob: float = 0.1) -> str:
        """
        Randomly delete words from text to create variations.
        
        Args:
            text: Input text
            deletion_prob: Probability of deleting each word
            
        Returns:
            Augmented text with random deletions
        """
        words = text.split()
        
        # Keep at least one word
        augmented_words = [w for w in words if np.random.random() > deletion_prob]
        if not augmented_words:
            augmented_words = words[:1]
        
        return ' '.join(augmented_words)
    
    def augment_text(self, text: str, num_augmentations: int = 3) -> List[str]:
        """
        Generate multiple augmented versions of text.
        
        Args:
            text: Input text
            num_augmentations: Number of augmented versions to generate
            
        Returns:
            List of augmented texts
        """
        augmented_texts = [text]  # Include original
        
        for _ in range(num_augmentations - 1):
            augmented = text
            
            # Randomly apply augmentation techniques
            if np.random.random() < 0.5:
                augmented = self.synonym_replacement(augmented)
            if np.random.random() < 0.3:
                augmented = self.random_insertion(augmented)
            if np.random.random() < 0.2:
                augmented = self.random_deletion(augmented)
            
            augmented_texts.append(augmented)
        
        return augmented_texts
    
    def augment_dataset(
        self,
        texts: List[str],
        labels: List[int],
        target_size: int
    ) -> Tuple[List[str], List[int]]:
        """
        Augment entire dataset to reach target size.
        
        Args:
            texts: List of input texts
            labels: List of corresponding labels
            target_size: Desired dataset size after augmentation
            
        Returns:
            Tuple of (augmented_texts, augmented_labels)
        """
        if len(texts) >= target_size:
            return texts[:target_size], labels[:target_size]
        
        augmented_texts = list(texts)
        augmented_labels = list(labels)
        
        # Calculate how many more samples needed
        needed = target_size - len(texts)
        
        # Generate augmented samples
        while len(augmented_texts) < target_size:
            # Randomly select a sample to augment
            idx = np.random.randint(0, len(texts))
            original_text = texts[idx]
            original_label = labels[idx]
            
            # Generate augmentation
            augmented = self.augment_text(original_text, num_augmentations=1)[1]
            augmented_texts.append(augmented)
            augmented_labels.append(original_label)
        
        return augmented_texts[:target_size], augmented_labels[:target_size]


# Example usage and testing
if __name__ == "__main__":
    # Test clinical text processor
    text_processor = ClinicalTextProcessor()
    sample_text = "Patient with HTN and DM presented with chest pain"
    cleaned = text_processor.clean_text(sample_text)
    entities = text_processor.extract_medical_entities(cleaned)
    
    print(f"Original: {sample_text}")
    print(f"Cleaned: {cleaned}")
    print(f"Entities: {entities}")
    
    # Test EHR processor
    ehr_processor = EHRDataProcessor()
    sample_patient = {
        'date_of_birth': '1980-01-01',
        'gender': 'M',
        'heart_rate': 85,
        'systolic_bp': 130,
        'chronic_conditions': ['hypertension', 'diabetes'],
        'admission_count': 2
    }
    features = ehr_processor.extract_patient_features(sample_patient)
    print(f"Extracted features shape: {features.shape}")
    
    # Test data augmenter
    augmenter = MedicalDataAugmenter()
    augmented = augmenter.augment_text(sample_text, num_augmentations=3)
    print(f"Augmented texts: {augmented}")