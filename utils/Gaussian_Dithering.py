import numpy as np
from add_version  import add_version

@add_version
def Gaussian_Dithering(obj, miu: float, sigma: float) -> np.ndarray:
    noise = np.random.normal(miu, sigma, size=(obj.shape[0]*obj.shape[1]//3,3))
    obj += noise
    return obj

def Gaussian_Dithering_Single(obj, miu: float, sigma: float) -> np.ndarray:
    noise = np.random.normal(miu, sigma, size=(1,3))
    obj += noise
    return obj

def Gaussian_Dithering_Single_Scalar(obj: float, miu: float, sigma: float) -> float:
    noise = np.random.normal(miu, sigma, size=(1,))
    obj += noise
    return obj

def Gaussian_Dithering_Single_Int(obj: int, miu: float, sigma: float) -> int:
    noise = np.random.normal(miu, sigma, size=(1,))
    obj += int(noise)
    return obj

def Gaussian_Dithering_Scalar(obj: float, miu: float, sigma: float) -> np.ndarray:
    noise = np.random.normal(miu, sigma, size=(obj.shape[0]*obj.shape[1]//3,))
    obj += noise
    return obj

def Gaussian_Dithering_Int(obj: int, miu: float, sigma: float) -> int:
    noise = np.random.normal(miu, sigma, size=(1,))
    obj += int(noise)
    return obj

def Gaussian_Dithering_Bool(obj: bool, miu: float, sigma: float) -> bool:
    noise = np.random.normal(miu, sigma, size=(1,))
    obj = bool(int(obj) + int(noise))
    return obj

def Gaussian_Dithering_Bool_Array(obj: np.ndarray, miu: float, sigma: float) -> np.ndarray:
    noise = np.random.normal(miu, sigma, size=(obj.shape[0]*obj.shape[1]//3,))
    obj = np.array([bool(int(val) + int(n)) for val, n in zip(obj.flatten(), noise)])
    return obj.reshape(obj.shape)

def Gaussian_Dithering_List(obj: list, miu: float, sigma: float) -> list:
    noise = np.random.normal(miu, sigma, size=(len(obj),))
    return [val + n for val, n in zip(obj, noise)]

def Gaussian_Dithering_Tuple(obj: tuple, miu: float, sigma: float) -> tuple:
    noise = np.random.normal(miu, sigma, size=(len(obj),))
    return tuple(val + n for val, n in zip(obj, noise))

def Gaussian_Dithering_Dict(obj: dict, miu: float, sigma: float) -> dict:
    noise = np.random.normal(miu, sigma, size=(len(obj),))
    return {key: val + n for (key, val), n in zip(obj.items(), noise)}

def Gaussian_Dithering_Set(obj: set, miu: float, sigma: float) -> set:
    noise = np.random.normal(miu, sigma, size=(len(obj),))
    return {val + n for val, n in zip(obj, noise)}

def Gaussian_Dithering_Nested_List(obj: list, miu: float, sigma: float) -> list:
    flat_list = [item for sublist in obj for item in sublist]
    noise = np.random.normal(miu, sigma, size=(len(flat_list),))
    dithered_flat = [val + n for val, n in zip(flat_list, noise)]
    dithered_nested = []
    index = 0
    for sublist in obj:
        length = len(sublist)
        dithered_nested.append(dithered_flat[index:index+length])
        index += length
    return dithered_nested
