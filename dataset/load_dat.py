import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_keel_dat(filepath):
    """
    加载 KEEL 数据集（.dat 文件），自动处理数值型和类别型特征。

    - 自动识别所有类别型列（不限位置）并使用 LabelEncoder 编码
    - 自动处理缺失值：数值列填均值
    - 返回 DataFrame，其中所有列都为数值类型（float 或 int）

    Args:
        filepath (str): .dat 文件路径

    Returns:
        df (pd.DataFrame): 编码后的数据集
        encoders (dict): 每个类别列的 LabelEncoder 对象，列名为键
    """
    with open(filepath, 'r') as file:
        lines = file.readlines()

    attributes = []
    data = []
    data_start = False

    for line in lines:
        line = line.strip()
        if line.lower().startswith("@attribute"):
            parts = line.split()
            attribute_name = parts[1]
            attributes.append(attribute_name)
        elif line.lower().startswith("@data"):
            data_start = True
        elif data_start and line:
            data.append(line.split(','))

    # 构造 DataFrame
    df = pd.DataFrame(data, columns=attributes)

    # 初始化编码器容器
    encoders = {}

    # 判断每一列的数据类型
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
            # 如果转换成功，说明是数值型
            df[col] = df[col].astype(float)
        except:
            # 如果不能转换为数字，说明是类别型
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # 填补数值列缺失值（NaN）
    df.fillna(df.mean(numeric_only=True), inplace=True)

    return df