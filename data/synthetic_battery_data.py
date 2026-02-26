import numpy as np
import pandas as pd

def generate_china_battery_data(n_batteries=10, cycles_per_battery=100):

    """
    生成模拟中国电动车电池数据：
    - 快充比例高 (30% cycles are fast charge)
    - 高温循环 (20% cycles at > 35°C)
    - 浅充浅放 (60% cycles between 20%-80% SoC)
    """

    data = []
    for bat_id in range(n_batteries):
        soh = 1.0
        for cycle in range(cycles_per_battery):
            is_fast_charge = np.random.rand() < 0.3
            temp = np.random.uniform(25, 45) if np.random.rand() < 0.2 else np.random.uniform(15, 30)
            dod = np.random.uniform(0.2, 0.8) if np.random.rand() < 0.6 else np.random.uniform(0.0, 1.0)

            degradation = 0.001 + (0.0005 if is_fast_charge else 0) + (0.0003 if temp > 35 else 0)
            soh = max(0.6, soh - degradation)

            data.append({
                'battery_id': bat_id,
                'cycle': cycle,
                'temp': temp,
                'dod': dod,
                'is_fast_charge': int(is_fast_charge),
                'soh': soh
            })

    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_china_battery_data()
    df.to_csv("data/synthetic_battery.csv", index=False)
    print("Synthetic battery data saved to data/synthetic_battery.csv")