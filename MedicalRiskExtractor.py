import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, pipeline
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import spacy
from collections import defaultdict
import re
from spacy.tokens import Doc
from scipy.stats import zscore
import datasets
from typing import List, Dict, Any
import random
from types import SimpleNamespace

def set_seed(seed):
    """
    Set random seeds for reproducibility across multiple libraries.
    
    Args:
        seed (int): Seed value to use
    """
    # Python's built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # CuDNN (PyTorch's deep learning backend)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DementiaReportDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512):
        set_seed(12)
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item

class SeverityAnalyzer:
    """Analyzes severity levels in medical text using contextual embeddings."""
    
    def __init__(self):
        set_seed(12)
        self.nlp = spacy.load('en_core_web_lg')
        self.sentence_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
        self.zero_shot_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )
    
    def _analyze_negation_context(self, doc):
        """Uses dependency parsing to understand negation context."""
        neg_states = []
        
        for token in doc:
            # Check for negation markers in dependency tree
            has_neg = any(child.dep_ == 'neg' for child in token.children)
            
            # Check for compound negation in subtree
            subtree_has_neg = any(t.dep_ == 'neg' for t in token.subtree)
            
            # Consider negated state based on local syntax
            is_negated = has_neg or subtree_has_neg
            
            # Store negation state for each token
            neg_states.append(is_negated)
            
        return neg_states
    
    def _get_finding_polarity(self, context):
        """Determines if a medical finding is positive or negative."""
        doc = self.nlp(context)
        
        # Get negation states for all tokens
        negation_states = self._analyze_negation_context(doc)
        
        # Analyze syntactic relationships
        finding_tokens = []
        for token, is_negated in zip(doc, negation_states):
            if self._is_medical_finding(token):
                finding_tokens.append((token, is_negated))
        
        # If we found medical findings, analyze their polarity
        if finding_tokens:
            # Default to positive unless negated
            return not any(neg for _, neg in finding_tokens)
            
        return True  # Default case
    
    def extract_medical_context(self, text):
        set_seed(12)
        """Extract medical terms and their context."""
        doc = self.nlp(text)
        medical_contexts = []
        
        for sent in doc.sents:
            # Get span window around medical terms
            medical_spans = []
            for token in sent:
                if self._is_medical_finding(token):
                    start = max(0, token.i - 5)
                    end = min(len(doc), token.i + 5)
                    medical_spans.append(doc[start:end].text)
            
            if medical_spans:
                medical_contexts.extend(medical_spans)
        
        return medical_contexts
    
    def _is_medical_finding(self, token):
        """Identifies tokens that represent medical findings."""
        return (
            token.pos_ in {'NOUN', 'PROPN'} and
            token.dep_ in {'nsubj', 'dobj', 'attr'} and
            not token.is_stop
        )
    
    def analyze_severity(self, context):
        set_seed(12)
        """Analyze severity level using zero-shot classification."""
        
        is_positive_finding = self._get_finding_polarity(context)
        
        candidate_labels = [
            "severe condition",
            "moderate condition",
            "mild condition",
            "normal condition"
        ]
        
        result = self.zero_shot_classifier(
            context,
            candidate_labels,
            hypothesis_template="The medical condition described is {}."
        )
        
        # Convert classification results to severity score
        severity_weights = {
            "severe condition": 1.0,
            "moderate condition": 0.6,
            "mild condition": 0.3,
            "normal condition": 0.0
        }
        
        # Calculate weighted score based on classification confidence
        scores = dict(zip(result['labels'], result['scores']))
        severity_score = sum(
            severity_weights[label] * score
            for label, score in scores.items()
        )
        if not is_positive_finding:
            severity_score = 1 - severity_score  # Substantial reduction for negative findings
            
        return severity_score

class InitialScorer:
    """Generates initial risk scores using unsupervised learning."""
    
    def __init__(self, device=None):
        set_seed(12)
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sentence_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
        self.nlp = spacy.load('en_core_web_lg')
        self.severity_analyzer = SeverityAnalyzer()
        
    def analyze_text(self, text):
        set_seed(12)
        """Analyze a single text for medical severity and context."""
        # Extract medical contexts
        medical_contexts = self.severity_analyzer.extract_medical_context(text)
        
        if not medical_contexts:
            return 0.0
        
        # Analyze severity for each context
        severity_scores = [
            self.severity_analyzer.analyze_severity(context)
            for context in medical_contexts
        ]
        
        # Combine severity scores
        if severity_scores:
            return np.mean(severity_scores)
        return 0.0
    
    def generate_training_scores(self, texts):
        set_seed(12)
        """Generate initial risk scores for a collection of medical reports."""
        print("Analyzing medical contexts and generating scores...")
        
        # Get embeddings for semantic similarity using batched processing
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            with torch.amp.autocast('cuda'):  # Enable automatic mixed precision
                batch_embeddings = self.sentence_model.encode(
                    batch_texts,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
            embeddings.append(batch_embeddings.cpu().numpy())  # Move to CPU for sklearn
            
        embeddings = np.vstack(embeddings)
        
        # Calculate severity scores with progress bar
        severity_scores = []
        for text in tqdm(texts, desc="Analyzing texts"):
            score = self.analyze_text(text)
            severity_scores.append(score)
        
        severity_scores = np.array(severity_scores)
        
        # Use K-means for additional context
        kmeans = KMeans(n_clusters=3, random_state=0)
        clusters = kmeans.fit_predict(embeddings)
        
        cluster_centers = kmeans.cluster_centers_
        cluster_distances = np.linalg.norm(cluster_centers, axis=1)
        cluster_risks = MinMaxScaler().fit_transform(cluster_distances.reshape(-1, 1)).flatten()
        # cluster_risks = cluster_distances
        
        # Combine scores and normalize
        combined_scores = 0.7 * severity_scores + 0.3 * cluster_risks[clusters]
        final_scores = MinMaxScaler().fit_transform(combined_scores.reshape(-1, 1)).flatten()
        
        print(final_scores)
        
        return final_scores

class RiskAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        set_seed(12)
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x, _ = self.attention(x, x, x)
        x = self.dropout(x)
        return x + residual

class RiskMLP(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        set_seed(12)
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.gelu(self.w1(x))
        x = self.dropout(x)
        x = self.w2(x)
        x = self.dropout(x)
        return x + residual

class RiskTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = RiskAttention(d_model, num_heads, dropout)
        self.mlp = RiskMLP(d_model, d_ff, dropout)

    def forward(self, x):
        x = self.attention(x)
        x = self.mlp(x)
        return x


class DementiaRiskAnalyzer:
    def __init__(self):
        set_seed(12)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = self._create_model()
        self.model.to(self.device)
        if torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler('cuda')  # For automatic mixed precision
            torch.backends.cudnn.benchmark = True  
        self.initial_scorer = InitialScorer()

    def _create_model(self):
        # Use base model instead of sequence classification
        model = AutoModelForSequenceClassification.from_pretrained(
        self.model_name,
        num_labels=1,
        problem_type="regression"
        )
        

        hidden_size = model.config.hidden_size  # 768 for BERT base
        
        # Create new architecture with correct dimensions
        model.classifier = nn.Sequential(      # Output shape: (batch_size, 1)
            nn.Linear(hidden_size, 256),       # Expand to larger dimension for transformer
            nn.LayerNorm(256),
            RiskTransformerBlock(
                d_model=256,
                num_heads=4,
                d_ff=512,
                dropout=0.2
            ),
            RiskTransformerBlock(
                d_model=256,
                num_heads=4,
                d_ff=512,
                dropout=0.2
            ),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        
        return model

    def train(self, train_texts, validation_split=0.2, epochs=7, batch_size=8):
        """Train the model using automatically generated scores with CUDA optimization."""
        # Generate initial scores
        train_scores = self.prepare_training_data(train_texts)
        
        # Split into train and validation sets
        indices = np.arange(len(train_texts))
        train_idx, val_idx = train_test_split(indices, test_size=validation_split, random_state=0)
        
        train_dataset = DementiaReportDataset(
            [train_texts[i] for i in train_idx],
            train_scores[train_idx],
            self.tokenizer
        )
        val_dataset = DementiaReportDataset(
            [train_texts[i] for i in val_idx],
            train_scores[val_idx],
            self.tokenizer
        )
        
        print("Length Train", len(train_dataset))
        print("Length Val", len(val_dataset))

        # Create optimized data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,  # Enable faster data transfer to GPU
            num_workers=4  # Enable parallel data loading
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=4
        )

        # Initialize optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            # Training with automatic mixed precision
            for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
                optimizer.zero_grad()
                
                # Move batch to GPU efficiently
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)

                if torch.cuda.is_available():
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        loss = F.mse_loss(outputs.logits.squeeze(), labels)
                    
                    # Use gradient scaling for mixed precision training
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    loss = F.mse_loss(outputs.logits.squeeze(), labels)
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()

            # Validation phase
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                    attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                    labels = batch['labels'].to(self.device, non_blocking=True)

                    with torch.amp.autocast('cuda'):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        predicted = outputs.logits.squeeze()  # Shape: [batch_size]
                        labels = labels.view(predicted.shape)  # Ensure labels match predicted shape
                        
                        # Calculate loss with matched shapes
                        val_loss += F.mse_loss(predicted, labels).item()

                        
                        # val_loss += F.mse_loss(outputs.logits.squeeze(), labels).item()

            # Print epoch statistics
            avg_train_loss = total_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            print(f'Epoch {epoch + 1}:')
            print(f'Average training loss: {avg_train_loss:.4f}')
            print(f'Average validation loss: {avg_val_loss:.4f}')

            # Clear GPU cache after each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def prepare_training_data(self, texts):
        """Generate training scores and prepare data for model training."""
        print("Generating initial risk scores...")
        initial_scores = self.initial_scorer.generate_training_scores(texts)
        
        print("\nScore distribution:")
        print(pd.Series(initial_scores).describe())
        
        return initial_scores
    @torch.amp.autocast('cuda')
    def analyze_report(self, medical_report):
        """
        Analyze a single medical report and return a risk score.
        
        Parameters:
        medical_report (str): The text of the medical report
        
        Returns:
        dict: Risk analysis results including score and confidence
        """
        self.model.eval()
        
        # Prepare input
        dataset = DementiaReportDataset([medical_report], tokenizer=self.tokenizer)
        loader = DataLoader(dataset, batch_size=1)
        
        with torch.no_grad():
            batch = next(iter(loader))
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            risk_score = outputs.logits.squeeze().item() * 100

        # Determine risk level
        risk_level = 'Low'
        if risk_score >= 75:
            risk_level = 'High'
        elif risk_score >= 40:
            risk_level = 'Moderate'

        return {
            'risk_score': round(risk_score, 2),
            'risk_level': risk_level
        }

    def analyze_batch(self, medical_reports, batch_size=4):
        """Analyze multiple medical reports efficiently using batching."""
        dataset = DementiaReportDataset(medical_reports, tokenizer=self.tokenizer)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=4
        )
        
        results = []
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Analyzing reports"):
                # Move batch to GPU efficiently
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                
                # Use automatic mixed precision for inference
                with torch.amp.autocast('cuda'):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                
                # Process results in batches
                risk_scores = outputs.logits.squeeze().cpu().numpy() * 100
                
                # Convert scores to risk levels and build results
                for score in risk_scores:
                    risk_level = 'Low'
                    if score >= 75:
                        risk_level = 'High'
                    elif score >= 40:
                        risk_level = 'Moderate'
                    
                    results.append({
                        'risk_score': round(float(score), 2),
                        'risk_level': risk_level
                    })
        
        return pd.DataFrame(results)

    # def save_model(self, path):
    #     """Save the trained model and tokenizer."""
    #     self.model.save_pretrained(path)
    #     self.tokenizer.save_pretrained(path)

    # def load_model(self, path):
    #     """Load a trained model and tokenizer."""
    #     self.model = AutoModelForSequenceClassification.from_pretrained(path)
    #     self.tokenizer = AutoTokenizer.from_pretrained(path)
    #     self.model.to(self.device)

class RadiologyDataLoader:
    """Handles loading and preprocessing of radiology datasets"""
    
    def __init__(self):
        self.datasets = {}
        
    def load_datasets(self):
        """Load datasets from Hugging Face hub"""
        print("Loading radiology datasets...")
        
        # Load wentingzhao/radiology
        zhao_dataset = datasets.load_dataset("wentingzhao/radiology")
        
        # Extract report texts from sft dataset
                
        # Extract report texts from zhao dataset
        zhao_reports = []
        for item in zhao_dataset['train']:
            if isinstance(item['text'], str):
                zhao_reports.append(item['text'])
                
        self.datasets = {
            'zhao': zhao_reports
        }
    
        print(f"Loaded {len(zhao_reports)} reports from wentingzhao/radiology")
        
        return self.combine_datasets()
    
    def preprocess_text(self, text: str) -> str:
        """Clean and standardize report text"""
        # Remove multiple spaces
        text = ' '.join(text.split())
        
        # Remove special characters while preserving medical symbols
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # Convert to lowercase
        text = text.lower()
        
        return text
    
    def combine_datasets(self) -> List[str]:
        """Combine and preprocess all datasets"""
        combined_reports = []
        
        for dataset_name, reports in self.datasets.items():
            processed_reports = [self.preprocess_text(report) for report in reports]
            combined_reports.extend(processed_reports)
            
        # Remove duplicates while preserving order
        combined_reports = list(dict.fromkeys(combined_reports))
        
        return combined_reports

class EnhancedDementiaRiskAnalyzer(DementiaRiskAnalyzer):
    """Enhanced version of DementiaRiskAnalyzer with radiology dataset support"""
    
    def __init__(self):
        super().__init__()
        self.data_loader = RadiologyDataLoader()
        
    def train_with_radiology_data(self, validation_split=0.2, epochs=5, batch_size=8):
        """Train the model using radiology datasets"""
        # Load and preprocess datasets
        train_texts = self.data_loader.load_datasets()
        
        # Train the model
        print(f"\nTraining model on {len(train_texts)} radiology reports...")
        self.train(train_texts, validation_split, epochs, batch_size)
        
        return len(train_texts)
    
    
# Example usage
if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = EnhancedDementiaRiskAnalyzer()

    data = pd.read_excel('./cognid.xlsx')
    reports = list(data['MRI full report '])
    reports = [report for report in reports if isinstance(report, str) and report.strip()]
    
    num_reports = analyzer.train(reports)
    
    # num_reports = analyzer.train_with_radiology_data()
    
    print(f"\nTraining completed on {num_reports} reports")
    # Example reports

    result = analyzer.analyze_batch(reports)
    high=0
    low=0
    medium=0
    for i in result:
        print(i)
    print("High", high)
    print("Low", low)
    print("Medium", medium)
    print("\nAnalysis Result:", result)
    
    # Save the trained model
    # analyzer.save_model("radiology_trained_model")