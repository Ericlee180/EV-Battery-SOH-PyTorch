def extract_china_features(cycle_data):
    fast_charge_ratio = count_fast_charges(cycle_data) / total_cycles
    high_temp_cycles = cycle_data[cycle_data['temp'] > 35]
    temp_stress = len(high_temp_cycles) / len(cycle_data)
    shallow_cycles = cycle_data[
        (cycle_data['dod'] < 0.8) & (cycle_data['dod'] > 0.2)
    ]
    shallow_ratio = len(shallow_cycles) / len(cycle_data)
    return [fast_charge_ratio, temp_stress, shallow_ratio]