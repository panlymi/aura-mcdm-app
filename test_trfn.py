import pandas as pd
from fuzzy_parser import parse_fuzzy_matrix, parse_fuzzy_weights
from fuzzy_aras_calculator import calculate_fuzzy_aras

def main():
    # Construct the data exactly like the Excel
    data = {
        'C1 Participation': ['0.7, 0.9, 1, 1', '0.5, 0.7, 0.7, 0.9', '0.3, 0.5, 0.5, 0.7'],
        'C2 Flexibility': ['0.1, 0.3, 0.3, 0.5', '0.5, 0.7, 0.7, 0.9', '0.7, 0.9, 1, 1'],
        'C3 Overall Effectiveness': ['0.3, 0.5, 0.5, 0.7', '0.5, 0.7, 0.7, 0.9', '0.5, 0.7, 0.7, 0.9']
    }
    df = pd.DataFrame(data, index=['A1 Traditional Lecture', 'A2 Problem-Based Learning', 'A3 Online Instruction'])
    
    weights = {
        'C1 Participation': '0.333, 0.333, 0.334, 0.334',
        'C2 Flexibility': '0.333, 0.333, 0.334, 0.334',
        'C3 Overall Effectiveness': '0.333, 0.333, 0.334, 0.334'
    }
    
    # Parse Data
    parsed_matrix = parse_fuzzy_matrix(df, method="TFNs/TrFNs")
    parsed_weights = parse_fuzzy_weights(weights, method="TFNs/TrFNs")
    
    directions = {col: 'maximize' for col in df.columns}
    
    # Calculate
    results, steps = calculate_fuzzy_aras(parsed_matrix, parsed_weights, directions, return_steps=True)
    
    print("\n--- Final Results ---")
    print(results[['S_i (Fuzzy)', 'S_i (Crisp)', 'K_i (Utility Degree)', 'Rank']])
    print("\nTrFN calculation successful!")

if __name__ == "__main__":
    main()
