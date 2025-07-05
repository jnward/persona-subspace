#!/usr/bin/env python3
"""
Run entity/behavior/semantic analysis on features from universal_30.csv
"""

import pandas as pd
import asyncio
import pathlib
import re
from entity_behavior_semantic_autointerp import analyze_features_with_claude


def parse_source_info(source: str):
    """Parse source string to extract model, layer and trainer info."""
    # Example: "llama_trainer1_layer11_asst" or "qwen_trainer1_layer15_newline"
    match = re.search(r'(\w+)_trainer(\d+)_layer(\d+)', source)
    if match:
        model_prefix = match.group(1)
        trainer = int(match.group(2))
        layer = int(match.group(3))
        return model_prefix, layer, trainer
    else:
        raise ValueError(f"Could not parse source: {source}")


def get_model_info(model_prefix: str):
    """Get model path and directory name from model prefix."""
    if model_prefix == "llama":
        return "meta-llama/Meta-Llama-3.1-8B-Instruct", "llama-3.1-8b-instruct"
    elif model_prefix == "qwen":
        return "Qwen/Qwen2.5-7B-Instruct", "qwen-2.5-7b-instruct"
    else:
        raise ValueError(f"Unknown model prefix: {model_prefix}")


async def main():
    # Load the CSV file
    csv_path = "/root/git/persona-subspace/sae_feature_analysis/results/universal/universal_30.csv"
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} features from {csv_path}")
    
    # Group features by model and layer combination
    df['model_layer_key'] = df['source'].apply(lambda x: '_'.join(x.split('_')[:3]))  # e.g., "llama_trainer1_layer11"
    groups = df.groupby('model_layer_key')
    
    print(f"Found {len(groups)} different SAE groups:")
    for key, group in groups:
        print(f"  {key}: {len(group)} features")
    
    # Process each group separately
    all_results = {}
    
    for group_key, group_df in groups:
        print(f"\n🔍 Processing group: {group_key}")
        
        # Parse source info for this group
        model_prefix, layer, trainer = parse_source_info(group_df['source'].iloc[0])
        model_path, model_dir = get_model_info(model_prefix)
        
        print(f"  Model: {model_path}")
        print(f"  Layer: {layer}, Trainer: {trainer}")
        print(f"  Features: {len(group_df)}")
        
        # Extract feature IDs for this group
        feature_ids = group_df['feature_id'].tolist()
        
        # Set up paths for feature mining data
        feature_mining_path_template = f"/workspace/sae/{model_dir}/feature_mining/resid_post_layer_{layer}/trainer_1"
        
        # Run Claude analysis for this group
        results = await analyze_features_with_claude(
            feature_ids=feature_ids,
            layer=layer,
            trainer=trainer,
            max_concurrent=20,  # Be conservative with API calls
            output_dir="/root/git/persona-subspace/.cache",
            feature_mining_path_template=feature_mining_path_template,
            claude_model="claude-opus-4-20250514",  # Claude model for inference
            baseline_model_path=model_path  # Model path for tokenizer
        )
        
        # Store results
        all_results.update(results)
        print(f"  ✅ Completed analysis for {len(results)} features")
    
    # Add results to dataframe
    df['claude_completion'] = df['feature_id'].map(lambda x: all_results.get(x, {}).get('claude_completion', ''))
    df['feature_description'] = df['feature_id'].map(lambda x: all_results.get(x, {}).get('feature_description', ''))
    df['type'] = df['feature_id'].map(lambda x: all_results.get(x, {}).get('type', ''))
    
    # Save results
    output_path = "/root/git/persona-subspace/sae_feature_analysis/results/universal/universal_30_autointerp.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Analysis complete! Results saved to {output_path}")
    print(f"Feature type distribution:")
    print(df['type'].value_counts())


if __name__ == "__main__":
    asyncio.run(main())