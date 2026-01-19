"""
Data cleaning and preprocessing module for diabetes hospital analytics.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List


def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw diabetes dataset."""
    return pd.read_csv(filepath)


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Check for missing values including '?' markers."""
    missing_summary = []
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if df[col].dtype == 'object':
            qm_count = (df[col] == '?').sum()
            total_missing = null_count + qm_count
            if total_missing > 0:
                missing_summary.append({
                    'Column': col,
                    'Null Values': null_count,
                    "'?' Values": qm_count,
                    'Total Missing': total_missing,
                    'Missing %': round(total_missing/len(df)*100, 2)
                })
        else:
            if null_count > 0:
                missing_summary.append({
                    'Column': col,
                    'Null Values': null_count,
                    "'?' Values": 0,
                    'Total Missing': null_count,
                    'Missing %': round(null_count/len(df)*100, 2)
                })
    
    if missing_summary:
        return pd.DataFrame(missing_summary).sort_values('Total Missing', ascending=False)
    else:
        return pd.DataFrame()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the diabetes dataset by handling missing values,
    removing duplicates, and processing features.
    """
    df = df.copy()
    
    # Replace '?' with NaN
    df = df.replace('?', np.nan)
    
    # Drop columns with high missing values
    high_missing_cols = ['weight', 'payer_code', 'medical_specialty']
    df = df.drop(columns=[col for col in high_missing_cols if col in df.columns], errors='ignore')
    
    # Remove duplicate encounters
    df = df.drop_duplicates(subset=['encounter_id'], keep='first')
    
    # Handle race missing values
    if 'race' in df.columns:
        df['race'] = df['race'].fillna('Unknown')
    
    # Create age midpoint for numerical analysis
    if 'age' in df.columns:
        age_mapping = {
            '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
            '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
            '[80-90)': 85, '[90-100)': 95
        }
        df['age_midpoint'] = df['age'].map(age_mapping)
        df['age_group'] = df['age']
    
    return df


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Create readmission target variables."""
    df = df.copy()
    
    if 'readmitted' in df.columns:
        # Binary target for 30-day readmission
        df['readmitted_30_days'] = (df['readmitted'] == '<30').astype(int)
        
        # Binary target for any readmission
        df['readmitted_binary'] = (df['readmitted'] != 'NO').astype(int)
    
    return df


def categorize_diagnoses(df: pd.DataFrame) -> pd.DataFrame:
    """Categorize diagnosis codes into meaningful groups."""
    df = df.copy()
    
    def categorize_diag(code):
        """Categorize ICD-9 diagnosis codes."""
        if pd.isna(code):
            return 'Unknown'
        
        code_str = str(code)
        
        # Check for specific patterns
        if code_str.startswith('250'):
            return 'Diabetes'
        elif code_str.startswith(('390', '391', '392', '393', '394', '395', '396', '397', '398')) or \
             (code_str.startswith('4') and 400 <= float(code_str.split('.')[0]) < 460):
            return 'Circulatory'
        elif code_str.startswith(('460', '461', '462', '463', '464', '465', '466', '467', '468', '469', '470', '471', '472', '473', '474', '475', '476', '477', '478', '479', '480', '481', '482', '483', '484', '485', '486', '487', '488', '489', '490', '491', '492', '493', '494', '495', '496', '497', '498', '499', '500', '501', '502', '503', '504', '505', '506', '507', '508', '509', '510', '511', '512', '513', '514', '515', '516', '517', '518', '519')):
            return 'Respiratory'
        elif code_str.startswith(('520', '521', '522', '523', '524', '525', '526', '527', '528', '529', '530', '531', '532', '533', '534', '535', '536', '537', '538', '539', '540', '541', '542', '543', '544', '545', '546', '547', '548', '549', '550', '551', '552', '553', '554', '555', '556', '557', '558', '559', '560', '561', '562', '563', '564', '565', '566', '567', '568', '569', '570', '571', '572', '573', '574', '575', '576', '577', '578', '579')):
            return 'Digestive'
        elif code_str.startswith(('800', '801', '802', '803', '804', '805', '806', '807', '808', '809', '810', '811', '812', '813', '814', '815', '816', '817', '818', '819', '820', '821', '822', '823', '824', '825', '826', '827', '828', '829', '830', '831', '832', '833', '834', '835', '836', '837', '838', '839', '840', '841', '842', '843', '844', '845', '846', '847', '848', '849', '850', '851', '852', '853', '854', '855', '856', '857', '858', '859', '860', '861', '862', '863', '864', '865', '866', '867', '868', '869', '870', '871', '872', '873', '874', '875', '876', '877', '878', '879', '880', '881', '882', '883', '884', '885', '886', '887', '888', '889', '890', '891', '892', '893', '894', '895', '896', '897', '898', '899', '900', '901', '902', '903', '904', '905', '906', '907', '908', '909')):
            return 'Injury'
        elif code_str.startswith(('710', '711', '712', '713', '714', '715', '716', '717', '718', '719', '720', '721', '722', '723', '724', '725', '726', '727', '728', '729', '730', '731', '732', '733', '734', '735', '736', '737', '738', '739')):
            return 'Musculoskeletal'
        elif code_str.startswith(('580', '581', '582', '583', '584', '585', '586', '587', '588', '589', '590', '591', '592', '593', '594', '595', '596', '597', '598', '599', '600', '601', '602', '603', '604', '605', '606', '607', '608', '609', '610', '611', '612', '613', '614', '615', '616', '617', '618', '619', '620', '621', '622', '623', '624', '625', '626', '627', '628', '629')):
            return 'Genitourinary'
        elif code_str.startswith('V') or code_str.startswith('E'):
            return 'External/Supplemental'
        else:
            return 'Other'
    
    # Categorize primary diagnoses
    for col in ['diag_1', 'diag_2', 'diag_3']:
        if col in df.columns:
            df[f'{col}_category'] = df[col].apply(categorize_diag)
    
    # Create primary diagnosis category (same as diag_1)
    if 'diag_1_category' in df.columns:
        df['primary_diagnosis_category'] = df['diag_1_category']
    
    return df


def prepare_modeling_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Prepare data for modeling by selecting features and encoding.
    Returns: (processed_df, numerical_features, categorical_features)
    """
    # Define features
    numerical_features = [
        'age_midpoint',
        'time_in_hospital',
        'num_lab_procedures',
        'num_procedures',
        'num_medications',
        'number_outpatient',
        'number_emergency',
        'number_inpatient',
    ]
    
    categorical_features = [
        'race',
        'gender',
        'admission_type_id',
        'admission_source_id',
        'discharge_disposition_id',
        'diag_1_category',
        'diag_2_category',
        'diag_3_category',
        'primary_diagnosis_category'
    ]
    
    # Select available features
    available_numerical = [f for f in numerical_features if f in df.columns]
    available_categorical = [f for f in categorical_features if f in df.columns]
    
    target_col = 'readmitted_30_days'
    selected_cols = available_numerical + available_categorical + [target_col]
    
    # Filter to only include available columns
    df_selected = df[[col for col in selected_cols if col in df.columns]].copy()
    
    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df_selected, columns=available_categorical, drop_first=True)
    
    return df_encoded, available_numerical, available_categorical


def get_data_summary(df: pd.DataFrame) -> dict:
    """Get summary statistics of the dataset."""
    return {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'unique_patients': df['patient_nbr'].nunique() if 'patient_nbr' in df.columns else None,
        'unique_encounters': df['encounter_id'].nunique() if 'encounter_id' in df.columns else None,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'date_range': (df['encounter_id'].min(), df['encounter_id'].max()) if 'encounter_id' in df.columns else None
    }
