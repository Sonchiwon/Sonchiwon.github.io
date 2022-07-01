---
title: Numba로 Pairwise Jaccard score 계산 가속
date: 2022-06-29 00:00:00 +0900
tags: [Python, Numba, Jaccard score]
description: Numba의 목적은 Python 함수의 실행 시간을 단축 시키는 것입니다. 이 문서에서는 binary array 들의 집합에서 모든 쌍의 Jaccard score 계산에 GPU 병렬 처리를 적용하여, 실행 시간이 얼마나 단축되는지 비교하겠습니다.
math: true
mermaid: true
author: Sonchiwon
---
> {{ page.description }}
{: .prompt-tip }


먼저 [이전 게시물](https://sonchiwon.github.io/posts/Numba%EC%9D%98GPU%EC%B2%98%EB%A6%AC%EB%A5%BC%ED%99%9C%EC%9A%A9%ED%95%9CJaccardscore%EA%B3%84%EC%82%B0/)을 참고하여 
scikit-learn 버전의 Pairwise Jaccard score 계산 동작을 구현해 보겠습니다.

```python
from typing import Iterable  
from sklearn.metrics import jaccard_score  
import itertools  
import numpy as np  
  
binarrs_number = 240  
binarrs_bits = 160  
binarrs = np.random.choice(2, size=(binarrs_number, binarrs_bits))  
for a, b in itertools.combinations(binarrs, 2):  
    print(jaccard_score(a, b))
```

비교를 위해 동일한 `binarrs`를 입력받는 scikit-learn과 numba-gpu 버전의 두 함수를 준비합니다.

```python
def pjs_sklearn(binarrs: np.ndarray) -> Iterable[float]:  
    for a, b in itertools.combinations(binarrs, 2):  
        yield jaccard_score(a, b)  
  
def pjs_nbgpu(binarrs: np.ndarray) -> Iterable[float]:  
    pass
```

이제 `pjs_nbgpu` 함수 내부를 채워 봅시다.


이전 게시물에서 살펴본 Numba GPU 함수의 호출부는 일종의 관용구로 모든 Numba GPU 함수는 동일한 형식을 따릅니다. 
이 관용구를 활용해서 우선 `pjs_nbgpu` 형태를 잡아 보겠습니다.

```python
from numba import cuda  
  
def pjs_nbgpu(binarrs: np.ndarray) -> Iterable[float]:  
  
    @cuda.jit('void(uint64[:, :], uint64[:])')  
    def inner_pjs_nbgpu(binarrs, out):  
        pass  
  
    length = len(binarrs) * (len(binarrs) - 1) // 2  
    _binarrs = cuda.to_device(binarrs)  
    _out = cuda.device_array(length, np.float64)  
    inner_pjs_nbgpu[length, 1](_binarrs, _out)  
    out = _out.copy_to_host()  
    yield from out
```

그런데 `pjs_nbgpu` 내부에서 정의된 Numba GPU 함수, `inner_pjs_nbgpu`의 계산 결과값을 저장하는 out 버퍼의 크기는 얼마일까요? 
out 버퍼의 크기는 `pjs_sklearn` 함수의 반환값 순회인 `itertools.combinations(n, 2)`와 크기가 같아야 합니다. 
이 값은 n개 원소 중 순서에 상관없이 2개의 원소를 선택하는 경우의 수, 즉 

$$
{}_nC_2 = {_nP_2 \over 2!} = {n \times (n - 1) \over 2}
$$

이므로, 이 수식으로 `out` 배열의 크기 `length` 를 구할 수 있습니다.

위의 `pjs_nbgpu` 간략 버전에서 눈여겨봐야 할 또 한가지 중요한 부분은 Numba GPU 함수를 호출하는 구문
`inner_pjs_nbgpu[length, 1](_binarrs, _out)` 입니다. 
Numba는 GPU 함수를 동시에 실행(multi-threading)하여 처리를 가속하는데, 이 때 배열 표기로 Thread의 갯수를 지정합니다. 
`[length, 1]`은 CUDA Programming Model에서 전체 block 중 *length* 개의 block을 사용하고, 한 block 당 1 개의 Thread를 생성한다는 뜻입니다. 
즉, 이 구문에 의해 `inner_pjs_nbgpu` 함수는 `out` 배열의 크기인 $length \times 1$ 개가 생성됩니다. 
그리고 각 Thread(`inner_pjs_nbgpu`)들이 동시에 실행될 때 `cuda.blockIdx.x` 로 [0 ~ length-1] 범위의 숫자가 전달됩니다. 
참고로 `cuda.threadIdx.x` 에는 배열 표기의 두 번째 원소로 고유값이 할당되는데, 
이 예제에서는 그 범위가 [0 ~ (1–1)] 이므로 모든 Thread가 0 값의 `cuda.threadIdx.x`를 가지게 됩니다. 

```python
@cuda.jit('void(uint64[:, :], uint64[:])')  
def inner_pjs_nbgpu(binarrs, out):  
    thread_index = cuda.blockIdx.x  
    a, b = combindex_of_iterttools(thread_index)  
    out[thread_index] = pair_jaccard_score(a, b)
```
위 코드는 `inner_pjs_nbgp`의 간소화 버전입니다. 
가독성을 위해 `combindex_of_iterttools`, `pair_jaccard_score` 함수가 사용된 것을 확인해 주세요.
모든 `inner_pjs_nbgpu`는 동시에 실행되면서 서로 다른 `cuda.blockIdx.x` 값으로 `thread_index` 값을 지정하고 
`out` 배열에서 자신의 `thread_index` 자리에 `pair_jaccard_score` 계산 결과를 채웁니다.

`pair_jaccard_score` 는 이전 게시물에서 다룬 Numba GPU 버전의 jaccard score 함수와 거의 동일하므로 
`combindex_of_iterttools` 를 한 번 살펴보겠습니다.


```python
@cuda.jit('uint64(uint64, uint64)', device=True)  
def combindex_of_iterttools(_length, _index):  
    i = _index  
    m = n = _length - 1  
    while i >= n:  
        i -= n  
        n -= 1  
    m -= n  
  
    index_of_a = m  
    index_of_b = m + i + 1  
    return index_of_a * _length + index_of_b
```

`combindex_of_iterttools`는 `pjs_sklearn`에서 사용하는 `itertools.combinations(binarrs, 2)` 를 모사하는데, `thread_index`를 binary array 배열 내의 인덱스 두 개로 매핑해 줍니다. 
이 구현부는 [선배님들의 지혜](https://stackoverflow.com/questions/40308722/how-to-know-combination-of-elements-from-its-index-in-list-of-all-combionations)를 참고하였습니다.
약간의 차이점은 Numba GPU 함수는 메모리 할당이 필요한 배열값을 반환할 수 없어, 두 개의 index 값을 하나의 uint64 변수로 반환하도록 반환값을 수정한 것 정도입니다.

설명을 이어가기 전에 용어를 짧게 정리하겠습니다. 사실 지금까지 Python 함수와 구분짓기 위해 사용했던 *Numba GPU 함수*는 **<U>Kernel 함수</U>**라는 정확한 이름이 있습니다. 
Kernel 함수의 정의는 Python에서 호출되는 Numba GPU 함수입니다. 
그리고 Kernel 함수에서 호출되는 Numba GPU 함수는 이와 구분짓기 위해 **<U>Device 함수</U>**라고 부릅니다. 
그러니까 `combindex_of_iterttools` 는 Device 함수입니다. 
Device 함수는 `cuda.jit` decorator에 `device=True` 를 명시해야 하고, Kernel 함수와 달리 반환값을 가질 수 있다는 차이점이 있습니다.

이제 지금까지 살펴본 내용을 합쳐 `pjs_nbgpu` 를 완성해 보겠습니다.
```python
def pjs_nbgpu(binarrs: np.ndarray) -> Iterable[float]:  
  
    @cuda.jit('uint64(uint64, uint64)', device=True)  
    def combindex_of_iterttools(_length, _index):  
        i = _index  
        m = n = _length - 1  
        while i >= n:  
            i -= n  
            n -= 1  
        m -= n  
        index_of_a, index_of_b = m, m + i + 1  
        return index_of_a * _length + index_of_b  
  
    @cuda.jit('float64(int64[:], int64[:])', device=True)  
    def pair_jaccard_score(a, b):  
        count_and, count_or = 0, 0  
        for i in range(len(a)):  
            count_and += a[i] & b[i]  
            count_or += a[i] | b[i]  
        return count_and / count_or  
          
    @cuda.jit('void(int64[:, :], float64[:])')  
    def inner_pjs_nbgpu(binarrs, out):  
        thread_index = cuda.blockIdx.x  
        _length = len(binarrs)  
        ab = combindex_of_iterttools(_length, thread_index)  
        a, b = ab // _length, ab % _length  
        out[thread_index] = pair_jaccard_score(binarrs[a], binarrs[b])  
  
    length = len(binarrs) * (len(binarrs) - 1) // 2  
    _binarrs = cuda.to_device(binarrs)  
    _out = cuda.device_array(length, np.float64)  
    inner_pjs_nbgpu[length, 1](_binarrs, _out)  
    out = _out.copy_to_host()  
    yield from out
```

Python 함수 `pjs_nbgpu` 는 두 개의 Device 함수, 하나의 Kernel 함수의 구현부와 Kernel 함수의 호출부로 구성됩니다. 
Kernel 함수는 `binarrs` 의 조합 갯수만큼 생성되어 전체 조합에 대한 Jaccard score가 동시에 계산됩니다. 
계산된 결과는 `itertools.combinations` 의 순회 순서에 맞추어 1차원 배열 `out` 변수에 저장되고 
모든 계산이 끝나면 `pjs_nbgpu`는 이 배열을 순회하며 값을 하나씩 반환합니다.

```python
import time  
start = time.time()  
result_of_pjs_sklearn = list(pjs_sklearn(binarrs))  
seconds_of_pjs_sklearn = time.time() - start  
start = time.time()  
result_of_pjs_nbgpu = list(pjs_nbgpu(binarrs))  
seconds_of_pjs_nbgpu = time.time() - start  
assert result_of_pjs_sklearn == result_of_pjs_nbgpu  
print(seconds_of_pjs_sklearn, seconds_of_pjs_nbgpu)

>>>  22.644657373428345 0.30959367752075195
```

마지막으로 scikit-learn 버전과 비교하여, 계산값이 일치하고 계산 시간이 단축된 실험 결과를 확인할 수 있습니다.

Numba는 Kernel 함수를 Multi-threading 하여 계산을 가속합니다. 
Kernel 함수의 Thread 갯수는 Kernel 함수 호출 시 `[Block의 갯수, 한 Block의 Thread 갯수]` 표현으로 
(Block의 갯수) $ \times $ (한 Block의 Thread 갯수) 를 지정합니다. 
Kernel 함수 호출 시 입력되는 인수는 모든 Thread가 공유하며 각 Thread는 고유 인덱스를 `cuda.blockIdx`, `cuda.threadIdx`로 전달 받아 자신의 업무를 처리할 수 있습니다.

이 문서의 Pairwise Jaccard score 예제에서는 하나의 binary array pair에 대한 Jaccard score 계산을 Kernel 함수로 만들고, 
이 Kernel 함수를 Multi-threading 하는 방법을 살펴 보았습니다. 
이 때 전체 pairwise 갯수만큼 생성된 각 Thread는 자신 고유의 값 BlockIdx로 계산할 binary array pair와 결과값 저장 위치를 구함으로써
모든 Thread가 서로 독립적인 작업을 한 번에 완료하는 것으로 계산 가속의 이득을 얻었습니다.
