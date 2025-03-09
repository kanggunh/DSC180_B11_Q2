import pandas as pd
import re
import ast
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

def fix_retained_percentage(test_data):
    """
    Fix a single row of test data by swapping retained percentage values, if necessary.
    The extraction model often swaps these values, so this function swaps them back.

    If the 'retained_percentage_cont' key is present and the 'retained_percentage_tret' key is not,
    move the value from the former to the latter and set the former to None.

    :param test_data: A single row of test data, represented as a dictionary.
    :return: The modified test data.
    """
    cont_key = 'retained_percentage_cont'
    tret_key = 'retained_percentage_tret'
    cont = test_data[cont_key] if cont_key in test_data else None 
    tret = test_data[tret_key] if tret_key in test_data else None
    if cont is not None and tret is None:
        test_data['retained_percentage_tret'] = cont
        test_data['retained_percentage_cont'] = None
    return test_data

def compute_molecular_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return [
                Descriptors.MolWt(mol),  # Molecular weight
                Descriptors.ExactMolWt(mol),  # Exact molecular weight (isotope-specific)
                Descriptors.MolLogP(mol),  # LogP (lipophilicity)
                Descriptors.TPSA(mol),  # Topological Polar Surface Area
                Descriptors.NumValenceElectrons(mol),  # Total valence electrons
                rdMolDescriptors.CalcNumRotatableBonds(mol),  # Rotatable bonds
                rdMolDescriptors.CalcNumHBA(mol),  # Hydrogen bond acceptors
                rdMolDescriptors.CalcNumHBD(mol),  # Hydrogen bond donors
                rdMolDescriptors.CalcFractionCSP3(mol),  # Fraction of sp3 carbons
                rdMolDescriptors.CalcNumAromaticRings(mol),  # Number of aromatic rings
                rdMolDescriptors.CalcNumSaturatedRings(mol),  # Number of saturated rings
                rdMolDescriptors.CalcNumHeteroatoms(mol),  # Number of heteroatoms
                rdMolDescriptors.CalcNumHeavyAtoms(mol),  # Number of heavy atoms
                rdMolDescriptors.CalcNumSpiroAtoms(mol),  # Number of spiro atoms
                rdMolDescriptors.CalcNumBridgeheadAtoms(mol),  # Number of bridgehead atoms
                Descriptors.FpDensityMorgan1(mol),  # Morgan fingerprint density (radius=1)
                Descriptors.FpDensityMorgan2(mol),  # Morgan fingerprint density (radius=2)
                Descriptors.FpDensityMorgan3(mol),  # Morgan fingerprint density (radius=3)
                Descriptors.qed(mol),  # Quantitative Estimate of Drug-likeness
                rdMolDescriptors.CalcNumLipinskiHBA(mol),  # Lipinski Hydrogen Bond Acceptors
                rdMolDescriptors.CalcNumLipinskiHBD(mol),  # Lipinski Hydrogen Bond Donors
                rdMolDescriptors.CalcNumRings(mol),  # Total number of rings
                rdMolDescriptors.CalcNumAmideBonds(mol),  # Number of amide bonds
                Descriptors.BalabanJ(mol),  # Balaban’s connectivity index
                Descriptors.BertzCT(mol),  # Bertz complexity
                Descriptors.Chi0(mol),  # Chi connectivity index (order 0)
                Descriptors.Chi1(mol),  # Chi connectivity index (order 1)
                Descriptors.Chi2n(mol),  # Chi connectivity index (order 2, non-H)
                Descriptors.Kappa1(mol),  # Kappa Shape Index (order 1)
                Descriptors.Kappa2(mol),  # Kappa Shape Index (order 2)
            ]
        else:
            return [np.nan] * 30  # Return NaN for missing values
    except:
        return [np.nan] * 30  # Return NaN for exceptions
    

def parse_perovskite_formula(formula):
    # Define allowed species (order matters for multi-letter elements)
    allowed_species = ["FA", "MA", "CS", "Rb", "Pb", "Sn", "I", "Br", "Cl"]

    # if is the nan we return component dictionary with all zeros
    if formula is np.nan:
        formula = ""    
    
    # Dictionary to store parsed results (initialize with 0.0 for all species)
    parsed_result = {species: 0.0 for species in allowed_species}

    # Step 1: Handle groups in parentheses with coefficients (e.g., (FAPbI3)0.95)
    pattern_group = r"\(([^)]+)\)\s*([0-9\.]+)"

    
    
    groups = re.findall(pattern_group, formula)

    if groups:
        for group, coef in groups:
            coef = float(coef)  # Convert coefficient to float
            elements = re.findall(r"(FA|MA|CS|Rb|Pb|Sn|I|Br|Cl)\s*([\d\.]*)", group)
            for element, count in elements:
                count = float(count) if count else 1.0
                parsed_result[element] += count * coef  # Distribute coefficient

    # Step 2: Handle formulas without parentheses (e.g., FA1-xMAxPbI3)
    remaining_formula = re.sub(r"\([^)]*\)\s*[0-9\.]+", "", formula)  # Remove processed groups
    elements = re.findall(r"(FA|MA|CS|Rb|Pb|Sn|I|Br|Cl)\s*([\d\.]*)", remaining_formula)

    for element, count in elements:
        count = float(count) if count and 'x' not in count else 1.0  # Ignore '-x' or 'x'
        parsed_result[element] += count

    # Round to 2 decimal places for all values
    parsed_result = {k: round(v, 2) for k, v in parsed_result.items()}

    return parsed_result

def clean_pin_nip_structure(df, column_name='pin_nip_structure'):
    # Define a mapping for known categories
    mapping = {
        'NIP': 'NIP',
        'PIN': 'PIN',
        'n-i-p': 'NIP',
        'p-i-n': 'PIN',
        'Inverted': 'PIN',
        'pin': 'PIN',
        'ITO/SnO2/perovskite/Spiro-OMeTAD/Ag': 'NIP'
    }
    
    # Apply the mapping and fill unknowns with 'Other'
    df[column_name] = df[column_name].map(mapping).fillna('Other')
    
    return df

def get_models(random_state):
    return {
        'Random Forest': RandomForestRegressor(random_state=random_state),
        'Linear Regression': LinearRegression(),
        'Support Vector Regressor': SVR(),
        'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
        'AdaBoost': AdaBoostRegressor(random_state=random_state)
    }

def run_prediction(prediction_df_path):
    prediction_df = pd.read_csv(prediction_df_path)
    prediction_df = prediction_df.apply(fix_retained_percentage, axis=1)

    df = prediction_df.copy()

    df = df[[
        'perovskite_composition', 'electron_transport_layer', 'hole_transport_layer', 'pin_nip_structure',
        'passivating_molecule', 'control_pce', 'control_voc', 'treated_pce', 'treated_voc', 'passivator_smiles', 'time', 
        'retained_percentage_tret'
    ]]
    df['pce_percent_change'] = ((df['treated_pce'] - df['control_pce']) / df['control_pce']) * 100

    # If it uses C60 as its electron_transport_layer
    df['C60'] = (df['electron_transport_layer'] == 'C60')

    # if it uses Spiro-OMeTAD as its hole_transport_layer
    df['Spiro-OMeTAD'] = (df['hole_transport_layer'] == 'Spiro-OMeTAD')

    mol_features = df['passivator_smiles'].apply(compute_molecular_features)

    # Convert list to DataFrame
    mol_features_df = pd.DataFrame(mol_features.tolist(), 
                                columns=[
                                    'MolWt', 'ExactMolWt', 'LogP', 'TPSA', 'NumValenceElectrons',
                                    'NumRotBonds', 'NumHBA', 'NumHBD', 'FractionCSP3', 'AromaticRings',
                                    'SaturatedRings', 'Heteroatoms', 'HeavyAtoms', 'SpiroAtoms', 
                                    'BridgeheadAtoms', 'FpDensityMorgan1', 'FpDensityMorgan2', 
                                    'FpDensityMorgan3', 'QED', 'LipinskiHBA', 
                                    'LipinskiHBD', 'NumRings', 'NumAmideBonds', 'BalabanJ', 
                                    'BertzCT', 'Chi0', 'Chi1', 'Chi2n', 'Kappa1', 'Kappa2'
                                ],
                                index=df.index)

    # Merge with original dataset
    df = pd.concat([df, mol_features_df], axis=1)
    df = df.dropna(subset=['passivator_smiles'])
    df = clean_pin_nip_structure(df)
    temp = df['perovskite_composition'].apply(parse_perovskite_formula).apply(pd.Series)
    df = pd.concat([df, temp], axis=1)
    df = df.dropna(subset=['NumRotBonds', 'treated_pce', 'control_pce', 'control_voc', 'treated_voc'])
    df = df[df['treated_pce'] > 10]
    df = df[df['treated_pce'] <= 35]
    df = df[df['control_pce'] > 10]
    df = df[df['control_pce'] <= 35]
    df = df[df['pce_percent_change'] <= 35]
    df = df[df['control_pce'] <= 35]

    # Define feature columns
    categorical_features = ['pin_nip_structure']
    numerical_features = ['control_voc', 'C60', 'Spiro-OMeTAD']
    smiles_features=[
                                    'MolWt', 'ExactMolWt', 'LogP', 'TPSA', 'NumValenceElectrons',
                                    'NumRotBonds', 'NumHBA', 'NumHBD', 'FractionCSP3', 'AromaticRings',
                                    'SaturatedRings', 'Heteroatoms', 'HeavyAtoms', 'SpiroAtoms', 
                                    'BridgeheadAtoms', 'FpDensityMorgan1', 'FpDensityMorgan2', 
                                    'FpDensityMorgan3', 'QED', 'LipinskiHBA', 
                                    'LipinskiHBD', 'NumRings', 'NumAmideBonds', 'BalabanJ', 
                                    'BertzCT', 'Chi0', 'Chi1', 'Chi2n', 'Kappa1', 'Kappa2'
                                ]
    composition_feature = ['FA', 'MA','CS', 'Rb', 'Pb', 'Sn', 'I', 'Br', 'Cl']
    other = ['control_pce']

    include_columns = categorical_features + smiles_features + composition_feature + other

    target_column = 'pce_percent_change'

    # Store performance metrics for each model and each random seed
    model_performance = []

    # Drop rows with missing values in both features and target column
    data_clean = df.dropna(subset=[target_column] + include_columns)
    data_clean.to_csv('../data/prediction/df_prediction_1.csv')

    # Prepare features and target again
    X = data_clean.drop(columns=[target_column])[include_columns]
    y = data_clean[target_column]

    random_seeds = random.sample(range(0, 1000), 100)

    # Iterate over multiple random seeds
    for seed in random_seeds:
        # Split the data into training and testing sets with the current random seed
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

        # One-hot encode categorical variables, ensuring consistent columns
        X_train = pd.get_dummies(X_train, columns=['pin_nip_structure'], drop_first=False)
        X_test = pd.get_dummies(X_test, columns=['pin_nip_structure'], drop_first=False)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        
        models = get_models(42)
        for name, model in models.items():
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            # Store metrics with random seed info
            model_performance.append({
                'Model': name,
                'Seed': seed,
                'R2': r2,
                'MAE': mae,
                'MSE': mse
            })

    # Convert to DataFrame
    performance_df = pd.DataFrame(model_performance)

    # Calculate average performance across random seeds
    average_performance = performance_df.groupby('Model').agg({
        'R2': 'mean',
        'MAE': 'mean',
        'MSE': 'mean'
    }).reset_index()

    # Sort models by average R2 performance
    average_performance = average_performance.sort_values(by='R2', ascending=False)
    average_performance.to_csv('../data/performance_results/prediction_avg_performance.csv')
    performance_df.to_csv('../data/performance_results/prediction_performance.csv')