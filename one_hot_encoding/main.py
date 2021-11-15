import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def label_encoding(values: np.ndarray) -> tuple:
    # Label to incremental integer
    label_encoder = LabelEncoder()
    # Work with unique, ordered list [cold: 0, hot: 1, warm: 2]
    integer_encoded = label_encoder.fit_transform(values)
    print(f'integer_encoded: {integer_encoded}')
    return label_encoder, integer_encoded


def one_hot_encoding(values: np.ndarray) -> np.ndarray:
    _, integer_encoded = label_encoding(values)

    one_hot_encoder = OneHotEncoder(sparse=False)
    # Reshape because it's (9,) and we want (9, 1)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
    print(f'one_hot_encoded: {one_hot_encoded}')

    return one_hot_encoded


def reverse_one_hot(values: np.ndarray, one_hot_element: int) -> np.ndarray:
    label_encoder, _ = label_encoding(values)

    # Reverse operation
    inverted = label_encoder.inverse_transform([np.argmax(one_hot_element)])
    print(f"Reverse {np.argmax(one_hot_element)} to {inverted}")
    return inverted


def convert_to32(one_hot_encoded: np.ndarray) -> np.ndarray:
    # Change type, float64 -> float32
    one_hot_encoded = one_hot_encoded.astype('float32')
    print(f"Type: {one_hot_encoded.dtype}")
    return one_hot_encoded


def main():
    # Categories - Nominal
    data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm']
    values = np.array(data)
    print(values)

    one_hot_encoded64 = one_hot_encoding(values)
    one_hot_encoded32 = convert_to32(one_hot_encoded64)
    one_hot_element = one_hot_encoded32[1, :]
    _ = reverse_one_hot(values, one_hot_element)


if __name__ == '__main__':
    main()
