"""
OpenAI Embedding Module
Reusable module for generating embeddings from categorical features using OpenAI API
"""

import numpy as np
import json
import time
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()


def load_labels_json(labels_path='labels.json'):
    """Load labels.json file"""
    with open(labels_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_categorical_mappings(labels_data, feature_to_label_mapping):
    """
    Create code to name and description mappings for categorical features.
    
    Parameters:
    - labels_data: loaded labels.json data
    - feature_to_label_mapping: dict mapping feature names to (category, subcategory) tuples
    
    Returns:
    - categorical_mappings: dict mapping feature names to code->name dicts
    - categorical_descriptions: dict mapping feature names to code->description dicts
    """
    categorical_mappings = {}
    categorical_descriptions = {}
    
    for feat_name, (category, subcategory) in feature_to_label_mapping.items():
        if category in labels_data and subcategory in labels_data[category]:
            code_to_name = {}
            code_to_description = {}
            for item in labels_data[category][subcategory]:
                code = item['code']
                name = item['name']
                description = item.get('description', name)
                code_to_name[code] = name
                code_to_description[code] = description
            categorical_mappings[feat_name] = code_to_name
            categorical_descriptions[feat_name] = code_to_description
    
    return categorical_mappings, categorical_descriptions


def convert_codes_to_names(data, feature_name, feature_idx, categorical_mappings):
    """
    Convert numeric codes to string names for a categorical feature.
    
    Parameters:
    - data: numpy array with shape (N, H, W, C)
    - feature_name: name of the categorical feature
    - feature_idx: index of the feature in the feature dimension
    - categorical_mappings: dict of code to name mappings
    
    Returns:
    - converted_feature: numpy array with codes converted to string names
    """
    if feature_name not in categorical_mappings:
        return data[:, :, :, feature_idx]
    
    mapping = categorical_mappings[feature_name]
    feature_values = data[:, :, :, feature_idx].astype(int)
    converted_names = np.empty(feature_values.shape, dtype=object)
    
    for code, name in mapping.items():
        converted_names[feature_values == code] = name
    
    # Handle unmapped values
    unmapped_mask = ~np.isin(feature_values, list(mapping.keys()))
    converted_names[unmapped_mask] = feature_values[unmapped_mask].astype(str)
    
    return converted_names


def combine_categorical_features(data, categorical_features, categorical_feature_indices,
                                categorical_mappings, categorical_descriptions, feature_labels):
    """
    Combine categorical features into descriptive text strings.
    
    Parameters:
    - data: numpy array with shape (N, H, W, C)
    - categorical_features: list of categorical feature names
    - categorical_feature_indices: dict mapping feature names to indices
    - categorical_mappings: dict of code to name mappings
    - categorical_descriptions: dict of code to description mappings
    - feature_labels: dict mapping feature names to readable labels
    
    Returns:
    - combined_texts: numpy array of shape (N, H, W) with dtype=object containing combined text
    """
    N, H, W = data.shape[:3]
    combined_texts = np.empty((N, H, W), dtype=object)
    
    # Create reverse mapping from name to code for efficient lookup
    name_to_code_mapping = {}
    for feat_name in categorical_features:
        if feat_name in categorical_mappings:
            name_to_code_mapping[feat_name] = {name: code for code, name in categorical_mappings[feat_name].items()}
    
    # Combine features for each spatial location
    for n in range(N):
        for h in range(H):
            for w in range(W):
                feature_parts = []
                for feat_name in categorical_features:
                    if feat_name in categorical_feature_indices:
                        feat_idx = categorical_feature_indices[feat_name]
                        feat_value = data[n, h, w, feat_idx]
                        feat_label = feature_labels.get(feat_name, feat_name)
                        
                        # Handle different data types
                        if isinstance(feat_value, str):
                            code = None
                            if feat_name in name_to_code_mapping:
                                code = name_to_code_mapping[feat_name].get(feat_value)
                            
                            if code is not None and feat_name in categorical_descriptions:
                                description = categorical_descriptions[feat_name].get(code, feat_value)
                                if description != feat_value:
                                    feature_parts.append(f"{feat_label}: {feat_value} ({description})")
                                else:
                                    feature_parts.append(f"{feat_label}: {feat_value}")
                            else:
                                feature_parts.append(f"{feat_label}: {feat_value}")
                        elif isinstance(feat_value, (int, float, np.integer, np.floating)):
                            code = int(feat_value)
                            name = categorical_mappings[feat_name].get(code, str(code))
                            if feat_name in categorical_descriptions:
                                description = categorical_descriptions[feat_name].get(code, name)
                                if description != name:
                                    feature_parts.append(f"{feat_label}: {name} ({description})")
                                else:
                                    feature_parts.append(f"{feat_label}: {name}")
                            else:
                                feature_parts.append(f"{feat_label}: {name}")
                        else:
                            feature_parts.append(f"{feat_label}: {str(feat_value)}")
                
                combined_texts[n, h, w] = "; ".join(feature_parts)
    
    return combined_texts


def generate_embeddings(combined_texts, client=None, model="text-embedding-3-small", 
                        dimensions=64, batch_size=100, verbose=True):
    """
    Generate embeddings for categorical feature texts using OpenAI API.
    
    Parameters:
    - combined_texts: numpy array of shape (N, H, W) with dtype=object containing text strings
    - client: OpenAI client instance (if None, creates new one)
    - model: OpenAI embedding model name
    - dimensions: embedding dimension (for text-embedding-3-small)
    - batch_size: batch size for API calls
    - verbose: whether to print progress
    
    Returns:
    - embeddings: numpy array of shape (N, H, W, dimensions) with embeddings
    - embeddings_dict: dict mapping text to embedding vector
    """
    if client is None:
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key is None:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file or environment.")
        client = OpenAI(api_key=api_key)
    
    # Get unique text combinations to reduce API calls
    unique_texts = np.unique(combined_texts.flatten())
    if verbose:
        print(f"Total unique text combinations: {len(unique_texts)}")
    
    # Batch processing for efficiency
    embeddings_dict = {}
    embedding_dim = None
    
    if verbose:
        print(f"Embedding categorical features using OpenAI ({model})...")
        print("This may take a while depending on the number of unique combinations...")
    
    for i in range(0, len(unique_texts), batch_size):
        batch_texts = unique_texts[i:i+batch_size].tolist()
        
        try:
            response = client.embeddings.create(
                input=batch_texts,
                model=model,
                dimensions=dimensions
            )
            
            # Store embeddings
            for j, embedding_obj in enumerate(response.data):
                text = batch_texts[j]
                embeddings_dict[text] = np.array(embedding_obj.embedding)
                if embedding_dim is None:
                    embedding_dim = len(embedding_obj.embedding)
            
            if verbose:
                print(f"Processed {min(i+batch_size, len(unique_texts))}/{len(unique_texts)} unique texts...")
            
            # Rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            if verbose:
                print(f"Error processing batch {i//batch_size}: {e}")
            continue
    
    if verbose:
        print(f"Embedding completed! Embedding dimension: {embedding_dim}")
    
    # Reconstruct embeddings for all samples
    if verbose:
        print("Reconstructing embeddings for all samples...")
    
    embeddings = np.zeros((combined_texts.shape[0], 
                          combined_texts.shape[1],
                          combined_texts.shape[2],
                          embedding_dim))
    
    for n in range(combined_texts.shape[0]):
        for h in range(combined_texts.shape[1]):
            for w in range(combined_texts.shape[2]):
                text = combined_texts[n, h, w]
                if text in embeddings_dict:
                    embeddings[n, h, w, :] = embeddings_dict[text]
                else:
                    if verbose:
                        print(f"Warning: Text not found in embeddings_dict at ({n}, {h}, {w})")
    
    if verbose:
        print(f"Categorical embeddings shape: {embeddings.shape}")
    
    return embeddings, embeddings_dict


def create_categorical_embeddings(data, feature_names, categorical_features, 
                                  categorical_feature_indices, labels_path='labels.json',
                                  feature_to_label_mapping=None, feature_labels=None,
                                  client=None, model="text-embedding-3-small", 
                                  dimensions=64, batch_size=100, verbose=True):
    """
    Complete pipeline to create embeddings from categorical features.
    
    Parameters:
    - data: numpy array with shape (N, H, W, C) containing feature data
    - feature_names: list of all feature names
    - categorical_features: list of categorical feature names
    - categorical_feature_indices: dict mapping categorical feature names to indices
    - labels_path: path to labels.json file
    - feature_to_label_mapping: dict mapping feature names to (category, subcategory) tuples
    - feature_labels: dict mapping feature names to readable labels
    - client: OpenAI client instance
    - model: OpenAI embedding model name
    - dimensions: embedding dimension
    - batch_size: batch size for API calls
    - verbose: whether to print progress
    
    Returns:
    - embeddings: numpy array of shape (N, H, W, dimensions) with embeddings
    - embeddings_dict: dict mapping text to embedding vector
    """
    # Load labels
    labels_data = load_labels_json(labels_path)
    
    # Create default mappings if not provided
    if feature_to_label_mapping is None:
        feature_to_label_mapping = {
            'geology': ('Geology', 'geology'),
            'landuse': ('Land_use', 'landuse'),
            'soil_drainage': ('Soil_summary', 'soil_drainage'),
            'soil_series': ('Soil_summary', 'soil_series'),
            'soil_texture': ('Soil_summary', 'soil_texture'),
            'soil_thickness': ('Soil_summary', 'soil_thickness'),
            'soil_sub_texture': ('Soil_summary', 'soil_sub_texture'),
            'forest_age': ('Forest_summary', 'forest_age'),
            'forest_diameter': ('Forest_summary', 'forest_diameter'),
            'forest_density': ('Forest_summary', 'forest_density'),
            'forest_type': ('Forest_summary', 'forest_type')
        }
    
    if feature_labels is None:
        feature_labels = {
            'geology': 'Geology',
            'landuse': 'Land Use',
            'soil_drainage': 'Soil Drainage',
            'soil_series': 'Soil Series',
            'soil_texture': 'Soil Texture',
            'soil_thickness': 'Soil Thickness',
            'soil_sub_texture': 'Soil Sub Texture',
            'forest_age': 'Forest Age',
            'forest_diameter': 'Forest Diameter',
            'forest_density': 'Forest Density',
            'forest_type': 'Forest Type'
        }
    
    # Create mappings
    categorical_mappings, categorical_descriptions = create_categorical_mappings(
        labels_data, feature_to_label_mapping
    )
    
    # Convert codes to names
    categorical_features_converted = {}
    for feat_name in categorical_features:
        if feat_name in categorical_feature_indices:
            feat_idx = categorical_feature_indices[feat_name]
            converted = convert_codes_to_names(data, feat_name, feat_idx, categorical_mappings)
            categorical_features_converted[feat_name] = converted
    
    # Combine categorical features into text
    combined_texts = combine_categorical_features(
        data, categorical_features, categorical_feature_indices,
        categorical_mappings, categorical_descriptions, feature_labels
    )
    
    # Generate embeddings
    embeddings, embeddings_dict = generate_embeddings(
        combined_texts, client=client, model=model, 
        dimensions=dimensions, batch_size=batch_size, verbose=verbose
    )
    
    return embeddings, embeddings_dict

