

import pandas as pd
import glob
import re
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split

# Ignore non-critical warnings for cleaner console output
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def clean_text_formatting(text: str) -> str:
    """
    Strips leading bullet points, numbers, and hyphens from the text.
    """
    if not isinstance(text, str): 
        return ""
    # Remove leading digits (e.g., "1. ", "2- ")
    text = re.sub(r'^\s*\d+[\.\)\-]\s*', '', text) 
    # Remove leading alphabetical bullets (e.g., "a) ", "b. ")
    text = re.sub(r'^\s*[a-zA-Z][\.\)\-]\s*', '', text)
    return text.strip()

def contains_arabic_script(text: str) -> bool:
    """
    Checks if the given string contains Arabic Unicode characters.
    """
    if not isinstance(text, str): 
        return False
    return bool(re.search(r'[\u0600-\u06FF]', text))

def detect_language_columns(df: pd.DataFrame):
    """
    Heuristically identifies the English and Arabic columns in a dataframe,
    regardless of their structural order.
    """
    en_index = -1
    ar_index = -1
    
    # Sample rows to optimize detection speed on large files
    sample_df = df.sample(min(20, len(df)), random_state=42) if len(df) > 20 else df
    
    for col_idx in range(len(df.columns)):
        col_data = sample_df.iloc[:, col_idx].astype(str).tolist()
        combined_text = " ".join(col_data)
        
        if contains_arabic_script(combined_text):
            ar_index = col_idx
        elif re.search(r'[a-zA-Z]', combined_text) and not contains_arabic_script(combined_text):
            # Ensure it's not a metadata column containing filler tags or URLs
            if "TOFILL" not in combined_text and "http" not in combined_text:
                en_index = col_idx
            # Fallback condition
            elif en_index == -1: 
                en_index = col_idx
            
    return en_index, ar_index

def is_valid_clinical_pair(row: pd.Series) -> bool:
    """
    Strict quality filter that rejects noisy or incomplete sentence pairs.
    """
    en_text = str(row['en'])
    ar_text = str(row['ar'])
    
    # Filter 1: Reject unresolved placeholder tags
    if "TOFILL" in en_text or "TOFILL" in ar_text: 
        return False
    
    # Filter 2: Reject URLs as they hold no semantic translation value
    if "http" in en_text or "www." in en_text: 
        return False
    
    # Filter 3: Ensure the Arabic column actually contains Arabic
    if not contains_arabic_script(ar_text): 
        return False
    
    # Filter 4: Reject extremely short, meaningless strings
    if len(en_text) < 3 or len(ar_text) < 3: 
        return False
    
    return True

def run_preprocessing_pipeline():
    print("=== Starting Medical Data Preprocessing Pipeline ===")
    
    # Use Pathlib for robust, OS-independent path handling
    base_raw_folder = Path(r"A:\ai_service\data_raw")
    
    if not base_raw_folder.exists():
        print(f"[Error] Raw data directory not found: {base_raw_folder}")
        return

    processed_dataframes = []
    
    # Locate all relevant file formats recursively
    target_files = []
    target_files.extend(base_raw_folder.rglob("*.csv"))
    target_files.extend(base_raw_folder.rglob("*.tsv"))
    target_files.extend(base_raw_folder.rglob("*.xlsx"))
    target_files.extend(base_raw_folder.rglob("*.xls"))
    
    print(f"Located {len(target_files)} raw data files. Initiating parsing...")

    successful_files_count = 0

    for file_path in target_files:
        try:
            df = None
            if file_path.suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, header=None)
            elif file_path.suffix == '.csv':
                try: 
                    df = pd.read_csv(file_path, header=None, encoding='utf-8', on_bad_lines='skip')
                except: 
                    df = pd.read_csv(file_path, header=None, encoding='utf-8-sig', on_bad_lines='skip')
            elif file_path.suffix == '.tsv':
                df = pd.read_csv(file_path, sep='\t', header=None, on_bad_lines='skip')

            if df is not None and df.shape[1] >= 2:
                en_col, ar_col = detect_language_columns(df)
                
                if en_col != -1 and ar_col != -1:
                    temp_df = pd.DataFrame()
                    temp_df['en'] = df.iloc[:, en_col].astype(str)
                    temp_df['ar'] = df.iloc[:, ar_col].astype(str)
                    
                    # Apply strict quality filter
                    mask = temp_df.apply(is_valid_clinical_pair, axis=1)
                    clean_df = temp_df[mask].copy()
                    
                    if not clean_df.empty:
                        # Normalize text formatting
                        clean_df['en'] = clean_df['en'].apply(clean_text_formatting)
                        clean_df['ar'] = clean_df['ar'].apply(clean_text_formatting)
                        
                        # Remove accidental header rows
                        invalid_headers = ['en', 'english', 'source', 'target']
                        clean_df = clean_df[~clean_df['en'].str.lower().isin(invalid_headers)]
                        
                        processed_dataframes.append(clean_df)
                        successful_files_count += 1
                
        except Exception as e:
            # Silently pass unreadable files, could log error here if needed
            pass

    if not processed_dataframes:
        print("\n[Warning] No valid bilingual pairs could be extracted.")
        return

    # Aggregate all processed dataframes
    final_dataset = pd.concat(processed_dataframes, ignore_index=True)
    
    print(f"\nInitial Aggregated Sentences: {len(final_dataset)}")
    
    # Final Deduplication to prevent overfitting
    final_dataset.drop_duplicates(subset=['en', 'ar'], inplace=True)
    
    # Extra safety check: ensure English column has letters
    final_dataset = final_dataset[final_dataset['en'].str.contains(r'[a-zA-Z]')]
    
    # Randomly shuffle the dataset to eliminate order bias
    final_dataset = final_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Final Pristine Clinical Sentences: {len(final_dataset)}")

    # Strategic Partitioning (Holdout Test Set)
    holdout_size = 2000
    if len(final_dataset) > holdout_size:
        test_dataset = final_dataset.iloc[:holdout_size]
        train_dataset = final_dataset.iloc[holdout_size:]
    else:
        # Fallback if dataset is surprisingly small
        train_dataset, test_dataset = train_test_split(final_dataset, test_size=0.1, random_state=42)

    # Define Output Paths
    output_dir = Path(r"A:\ai_service")
    output_path_train = output_dir / "train_final.csv"
    output_path_test = output_dir / "test_final.csv"
    
    # Save to disk
    train_dataset.to_csv(output_path_train, index=False, encoding='utf-8-sig')
    test_dataset.to_csv(output_path_test, index=False, encoding='utf-8-sig')
    
    print("\n=== Pipeline Execution Completed Successfully ===")
    print(f"Exported Training Set: {output_path_train}")
    print(f"Exported Testing Set:  {output_path_test}")

if __name__ == "__main__":
    run_preprocessing_pipeline()