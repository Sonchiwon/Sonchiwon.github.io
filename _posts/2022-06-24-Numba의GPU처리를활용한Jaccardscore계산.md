---
title: Numba의 GPU 처리를 활용한 Jaccard score 계산
date: 2022-06-24 00:00:00 +0900
tags: [Python, Numba, Jaccard score]
description: Numba는 Numpy 배열 연산을 빠르게 처리하는데 특화된 Python package 입니다. 이 문서에서는 Jaccard score 계산 예제를 통해 Numba로 GPU 처리를 활용하는 방법을 소개합니다.
math: true
mermaid: true
author: Sonchiwon
---
> {{ page.description }} 
{: .prompt-tip }


Jaccard score는 두 집합의 유사한 정도를 측정하는 방법 중 하나로 $J(A, B) = {| A \cup B | \div | A \cap B |}$ 로 계산되며 0과 1 사이의 값을 가집니다. 
Python에서 Jaccard score를 계산하는 가장 간단한 방법은 scikit-learn의 `jaccard_score` 함수를 호출하는 것입니다.
```python
from sklearn.metrics import jaccard_score  
import numpy as np  
  
a = np.random.choice(2, size=8)  # [0 1 0 1 1 1 0 0]  
b = np.random.choice(2, size=8)  # [0 0 1 0 1 1 1 1]  
print(jaccard_score(a, b))       # 0.2857142857142857
```
Numba는 Python으로 작성된 함수를 GPU에서 호출 가능한 함수로 변환하는데, 이 때 변환될 Python 함수를 구현하고 호출하는데 몇가지 규칙이 있습니다. 
이 규칙들을 살펴보기 위해 Jaccard score 계산식을 참고하여 `simple_jaccard_score`를 다음과 같이 구현해 보았습니다.
```python
def simple_jaccard_score(a: np.ndarray, b: np.ndarray) -> float:  
    count_and, count_or = 0, 0  
    for i in range(len(a)):  
        count_and += a[i] & b[i]  
        count_or += a[i] | b[i]  
    return count_and / count_or
```
이제 `simple_jaccard_score`를 약간 수정하여 Numba GPU 함수로 만들어 보겠습니다.

```python
from numba import cuda  
  
@cuda.jit('void(int64[:], int64[:], float64[:])')  
def gpu_jaccard_score(a, b, out):  
    count_and, count_or = 0, 0  
    for i in range(len(a)):  
        count_and += a[i] & b[i]  
        count_or += a[i] | b[i]  
    out[0] = count_and / count_or
```

`gpu_jaccard_score`에서 살펴볼 Numba GPU 함수의 구현 규칙은 다음과 같습니다.
-   GPU에서 실행할 함수에 `numba.cuda.jit` decorator를 사용합니다.
-   `numba.cuda.jit` decorator의 첫번째 인수(signature)는 생략 가능하지만, Numba GPU 함수에서 사용할 수 없는 type hint를 대체할 수 있습니다.
-   Numba GPU 함수는 Python과 Numpy의 기본 연산으로 작성되어야 합니다.
-   Numba GPU 함수는 return을 사용하지 않고, 인수로 입력받은 버퍼에 계산 결과를 채우는 방식으로 값을 반환해야 합니다.

그 다음 `gpu_jaccard_score`를 실행시켜 보고 Numba GPU 함수의 호출 규칙을 살펴 보겠습니다.

```python
_a = cuda.to_device(a)  
_b = cuda.to_device(b)  
_out = cuda.device_array(1, np.float64)  
gpu_jaccard_score[1, 1](_a, _b, _out)  
print(_out.copy_to_host())  # [0.28571429]
```

-   우선 Numba GPU 함수에 입력될 인수를 GPU 배열 변수로 만듭니다. `cuda.to_device`는 Numpy 배열을 복사하고, `cuda.device_array`는 빈 배열을 생성합니다.
-   Numba GPU 함수를 호출할 때 함수이름과 인수목록 사이에 배열 표기로 Thread의 갯수를 지정합니다. 지금은 하나의 Thread가 동작한다는 것 정도로만 참고해 주세요.
-   Numba GPU 함수가 종료된 후 반환값이 저장된 GPU 배열 변수에서 `cuda.copy_to_host`로 저장된 값을 가져옵니다.

이제 `gpu_jaccard_score` 의 구현부와 호출부를 합치면서 `sklearn.metrics.jaccard_score` 와 입출력을 맞추겠습니다.

```python
def numba_gpu_jaccard_score(a: np.ndarray, b: np.ndarray) -> float:  
  
    @cuda.jit('void(int64[:], int64[:], float64[:])')  
    def gpu_jaccard_score(a, b, out):  
        count_and, count_or = 0, 0  
        for i in range(len(a)):  
            count_and += a[i] & b[i]  
            count_or += a[i] | b[i]  
        out[0] = count_and / count_or  
  
    _a = cuda.to_device(a)  
    _b = cuda.to_device(b)  
    _out = cuda.device_array(1, np.float64)  
    gpu_jaccard_score[1, 1](_a, _b, _out)  
    return _out.copy_to_host()[0]
```
마지막으로 새롭게 작성된 Numba GPU 버전의 jaccard score 함수, `numba_gpu_jaccard_score`가 제대로 동작하는지 확인합니다.

```python
for _ in range(10):  
    a = np.random.choice(2, size=8)  
    b = np.random.choice(2, size=8)  
    assert jaccard_score(a, b) == numba_gpu_jaccard_score(a, b)
```

지금까지 Numba GPU 함수를 작성하는 규칙을 크게 구현부와 호출부로 나누어 살펴 보았습니다. 이 문서에서는 다루지 못했지만, Numba GPU 함수 작성 규칙과 관련한 좀 더 심도 있는 내용은 아래 Numba 공식 문서를 참고해 주세요  
-   [Decorator, `numba.cuda.jit` 의 명세](https://numba.readthedocs.io/en/stable/cuda-reference/kernel.html#numba.cuda.jit)
-   [`numba.cuda.jit`의 첫번째 인수, `signature` 기입 방법](https://numba.readthedocs.io/en/stable/reference/types.html?highlight=signature#signatures)
-   [Numba GPU 함수 작성 시 사용할 수 있는 Python 기능](https://numba.readthedocs.io/en/stable/cuda/cudapysupported.html)


