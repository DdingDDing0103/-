# 차원축소 기법이 신용카드 부도 예측에 미치는 영향: PCA · t-SNE · Isomap · UMAP 비교 분석

**Members:**
- 이름1, 학과, 이메일
- 이름2, 학과, 이메일
- 이름3, 학과, 이메일
- 이름4, 학과, 이메일

**Video/Audio link:** *(추후 추가 — 5~10분 분량 유튜브 링크)*

---

## 목차
- [I. Proposal](#i-proposal)
- [II. Datasets](#ii-datasets)
- [III. Methodology](#iii-methodology)
- [IV. Evaluation & Analysis](#iv-evaluation--analysis)
- [V. Related Work](#v-related-work)
- [VI. Conclusion](#vi-conclusion)

---

## I. Proposal

### Motivation

금융 데이터, 특히 신용 부도 예측 문제는 머신러닝의 고전적 응용 분야 중 하나다. 그런데 실제 금융 데이터셋은 다음과 같은 두 가지 특성을 동시에 가진다.

1. **상대적으로 고차원이다.** 한 사용자에 대해 수십~수백 개의 속성(소득, 과거 연체 이력, 청구 패턴 등)이 함께 기록된다.
2. **변수 간 상관관계가 크다.** 예를 들어 매월의 청구 금액(BILL_AMT)들은 서로 강하게 상관되어 있다.

이런 데이터에 분류 모델을 적용할 때, 우리는 보통 **차원축소(Dimensionality Reduction)** 라는 전처리 단계를 거친다. 차원축소는 (a) 학습 시간을 줄이고, (b) 다중공선성을 완화하며, (c) 시각화를 가능하게 한다. 그런데 차원축소 기법은 한 가지가 아니다. 가장 익숙한 PCA부터 t-SNE, Isomap, UMAP까지 — 각각 수학적 가정과 보존하려는 구조가 모두 다르다.

> **그렇다면 같은 데이터셋, 같은 분류기를 사용했을 때 어떤 차원축소 기법이 분류 성능에 가장 도움이 될까? 그리고 그 차이는 어디서 오는가?**

이 질문에 답해보는 것이 이번 프로젝트의 출발점이다.

### What do you want to see at the end?

프로젝트가 끝났을 때 우리는 다음을 가지고 있을 것이다.

- 4가지 차원축소 기법(PCA · t-SNE · Isomap · UMAP)의 수학적 직관과 차이에 대한 정리
- 동일한 신용 부도 데이터에 4가지 기법을 적용했을 때의 2D 임베딩 시각화
- 차원축소 후 3가지 분류기(Logistic Regression · Random Forest · XGBoost)를 적용한 성능 비교 표
- "왜 t-SNE/Isomap은 일반적으로 전처리용으로 쓰이지 않는가"에 대한 실증적 답변

본 프로젝트의 목표는 **최고 성능의 모델을 만드는 것이 아니라**, 차원축소 기법 간의 성격 차이가 분류 성능에 어떻게 반영되는지를 단계별로 보여주는 데 있다.

---

## II. Datasets

### Default of Credit Card Clients Dataset

- **출처:** UCI Machine Learning Repository / Kaggle
- **URL:** https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
- **샘플 수:** 30,000명
- **특성 수:** 23개 (타깃 제외)
- **타깃 변수:** `default.payment.next.month` (1 = 다음 달 연체, 0 = 정상)
- **클래스 분포:** 정상 약 77.88% / 연체 약 22.12% — 중간 정도의 불균형

이 데이터셋은 2005년 대만의 한 신용카드사 고객 데이터로, **UCI 머신러닝 저장소에 등재된 클래식 데이터셋** 중 하나다. 30,000개라는 규모는 t-SNE/Isomap 같이 계산량이 많은 알고리즘도 (시간은 좀 걸리지만) 실험 가능한 규모라는 점에서 본 프로젝트에 적합하다.

### 특성 설명

| 변수 | 의미 |
|---|---|
| `LIMIT_BAL` | 신용 한도 (NT달러) |
| `SEX` | 성별 (1: 남, 2: 여) |
| `EDUCATION` | 학력 (1: 대학원, 2: 대학, 3: 고교, 4: 기타) |
| `MARRIAGE` | 결혼 상태 (1: 기혼, 2: 미혼, 3: 기타) |
| `AGE` | 나이 |
| `PAY_0` ~ `PAY_6` | 최근 6개월 상환 상태 (-2~9: 음수는 정상상환, 양수는 연체 개월 수) |
| `BILL_AMT1` ~ `BILL_AMT6` | 최근 6개월 청구 금액 |
| `PAY_AMT1` ~ `PAY_AMT6` | 최근 6개월 실제 지불 금액 |

### 데이터 전처리

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 데이터 로드
df = pd.read_csv('UCI_Credit_Card.csv')

# 불필요 열 제거 (ID)
df = df.drop(columns=['ID'])

# 타깃 분리
y = df['default.payment.next.month']
X = df.drop(columns=['default.payment.next.month'])

# 범주형 변수: 원-핫 인코딩
categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# 학습/평가 분리 (불균형을 고려한 stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 스케일링 (차원축소에 필수)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

> **왜 스케일링이 필수인가?** PCA는 분산을 기준으로 축을 찾는데, `LIMIT_BAL`(수십만 단위)과 `AGE`(20~80) 같이 스케일이 다른 변수가 섞이면 큰 스케일 변수가 결과를 지배해버린다. t-SNE·Isomap·UMAP도 거리 기반이라 동일한 문제가 발생한다.

---

## III. Methodology

### 전체 파이프라인

```
[Raw 데이터] → [전처리 (인코딩 + 스케일링)] 
            → [차원축소 (4가지 중 1개)] 
            → [분류기 (3가지 중 1개)] 
            → [성능 평가 (Accuracy, F1, ROC-AUC)]
```

총 조합: **4 × 3 = 12가지 실험** + 차원축소를 거치지 않은 기준선(baseline) 3가지.

### 1) 차원축소 기법 4종

#### (a) PCA — Principal Component Analysis

**핵심 아이디어:** 데이터의 분산을 최대한 보존하는 방향(주성분)으로 좌표축을 새로 잡는다. 공분산 행렬의 고유벡터를 찾는 문제로 환원된다.

- **선형 변환**이다 — 즉, 비선형적 구조는 잡아내지 못한다.
- **전역 구조**를 보존한다 — 멀리 떨어진 점들 사이의 관계도 유지하려 한다.
- 매우 빠르고 결정적(deterministic)이다.
- 새로운 데이터에 대해 그대로 transform 적용 가능 (= 분류 전처리에 적합).

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)
print(f"설명된 분산 비율: {pca.explained_variance_ratio_.sum():.3f}")
```

#### (b) t-SNE — t-distributed Stochastic Neighbor Embedding

**핵심 아이디어:** 고차원에서 가까운 점들은 저차원에서도 가깝게, 먼 점들은 멀게 — 단, "가까움"을 확률 분포로 정의하고 두 분포 간 KL divergence를 최소화하는 식으로.

- **비선형**이며 강력한 **지역 구조 보존** 능력.
- 시각화에 매우 뛰어나지만, **본질적으로 시각화용 도구**다.
- `perplexity` 하이퍼파라미터에 민감 (보통 5~50).
- **결정적이지 않다** — 매번 다른 결과가 나올 수 있음.
- **transform 메서드가 없다** — 새 데이터를 같은 공간에 매핑할 수 없어 일반적으로 전처리용으로 부적합. *(본 프로젝트에서는 train+test를 한 번에 fit_transform한 뒤 분리하는 방식으로 실험한다. 이는 데이터 누수 우려가 있으므로 한계로 명시한다.)*

```python
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
X_tsne = tsne.fit_transform(X_train_scaled)
```

#### (c) Isomap — Isometric Mapping

**핵심 아이디어:** 데이터가 고차원 공간에 휘어진 매니폴드 위에 놓여 있다고 가정. 두 점 사이의 진짜 거리는 직선 거리(유클리디안)가 아니라 **매니폴드 위에서의 측지선 거리(geodesic distance)** 다. k-최근접 그래프로 측지선 거리를 근사한 후 MDS를 적용한다.

- **비선형**이며 **전역 매니폴드 구조 보존**.
- `n_neighbors` 하이퍼파라미터가 핵심.
- 새 데이터에 transform 적용 가능.
- 30,000개 정도면 계산 시간이 상당하다 — 메모리 issue가 생기면 서브샘플링 권장.

```python
from sklearn.manifold import Isomap
isomap = Isomap(n_components=2, n_neighbors=10)
X_isomap = isomap.fit_transform(X_train_scaled)
```

#### (d) UMAP — Uniform Manifold Approximation and Projection

**핵심 아이디어:** 위상수학(topology)과 리만 기하학에 기반. t-SNE처럼 지역 구조를 잘 보존하면서도 t-SNE보다 빠르고 **전역 구조도 어느 정도 보존**한다는 평가.

- t-SNE보다 빠르며 transform 메서드도 제공.
- 분류 전처리용으로 t-SNE보다 자연스럽게 쓸 수 있음.
- `n_neighbors`, `min_dist` 두 하이퍼파라미터로 결과 조절.

```python
import umap
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_train_scaled)
```

### 2) 분류기 3종

| 분류기 | 선택 이유 |
|---|---|
| Logistic Regression | 단순 선형 모델 기준선. 차원축소의 효과가 가장 잘 드러남. |
| Random Forest | 트리 기반 비선형 모델. 원래 차원에서도 잘 작동하는 모델이라 차원축소의 영향을 보기 좋음. |
| XGBoost | 부스팅 계열 강자. 정형 데이터의 표준. |

### 3) 평가지표

클래스 불균형이 있으므로 단순 Accuracy만 보면 안 된다.

- **Accuracy** — 전체 정확도
- **F1-score** (positive class) — 정밀도와 재현율의 조화평균
- **ROC-AUC** — 임계값에 무관한 분류 성능 (가장 중요하게 볼 지표)

---

## IV. Evaluation & Analysis

> **⚠️ 이 섹션은 실제 실험을 돌린 뒤 채워야 한다. 아래는 템플릿/예상 구조.**

### 1) 2D 임베딩 시각화

각 기법으로 2차원에 임베딩한 결과를 산점도로 비교한다. 정상(파랑) vs 연체(빨강) 클래스가 어떻게 분포하는지가 핵심.

*(여기에 4개의 산점도를 한 그림에 넣을 것 — `plt.subplots(2, 2)`)*

**예상 관찰 포인트:**
- PCA: 두 클래스가 얼마나 분리되는가? (선형 분리 가능성)
- t-SNE: 군집(cluster)이 형성되는가? 군집 내 클래스 분포는?
- Isomap: 매니폴드의 모양이 보이는가?
- UMAP: t-SNE와 비교해 어떤 차이가 있는가?

### 2) 분류 성능 비교 (테이블 템플릿)

| 차원축소 | 분류기 | Accuracy | F1 | ROC-AUC | 학습 시간(s) |
|---|---|---|---|---|---|
| None (Baseline) | Logistic Regression | _._ | _._ | _._ | _._ |
| None (Baseline) | Random Forest | _._ | _._ | _._ | _._ |
| None (Baseline) | XGBoost | _._ | _._ | _._ | _._ |
| PCA (2D) | Logistic Regression | _._ | _._ | _._ | _._ |
| PCA (2D) | Random Forest | _._ | _._ | _._ | _._ |
| PCA (2D) | XGBoost | _._ | _._ | _._ | _._ |
| t-SNE (2D) | Logistic Regression | _._ | _._ | _._ | _._ |
| t-SNE (2D) | Random Forest | _._ | _._ | _._ | _._ |
| t-SNE (2D) | XGBoost | _._ | _._ | _._ | _._ |
| Isomap (2D) | Logistic Regression | _._ | _._ | _._ | _._ |
| Isomap (2D) | Random Forest | _._ | _._ | _._ | _._ |
| Isomap (2D) | XGBoost | _._ | _._ | _._ | _._ |
| UMAP (2D) | Logistic Regression | _._ | _._ | _._ | _._ |
| UMAP (2D) | Random Forest | _._ | _._ | _._ | _._ |
| UMAP (2D) | XGBoost | _._ | _._ | _._ | _._ |

### 3) 추가 분석: 차원 수에 따른 변화

2차원만으로는 정보가 너무 손실되므로, 보조 실험으로 PCA·UMAP에 대해 차원 수를 늘려가며(2, 5, 10, 15) 성능 곡선을 그려본다. t-SNE는 보통 2~3차원으로 제한되므로 비교에서 제외.

### 4) 토론 (예상 결과)

실제 결과를 보기 전이지만 사전적으로 예상하는 시나리오:

- **PCA + 강한 분류기(RF, XGBoost) 조합**: 원본 대비 성능이 떨어질 가능성 — PCA의 선형 가정은 트리 모델이 잡아내는 비선형 상호작용 정보를 손실시킴.
- **t-SNE / UMAP은 군집은 잘 보이지만 분류 정확도는 낮을 것** — 본질적으로 시각화용 도구이며, 군집과 클래스 라벨이 일치한다는 보장이 없기 때문.
- **Isomap**: `n_neighbors` 설정에 따라 결과 편차가 클 것으로 예상.
- 결국 **차원축소가 분류 정확도를 무조건 올려주는 마법이 아니라는 것**이 핵심 메시지가 될 가능성이 높다.

*(실제 실험 결과로 위 예상이 맞았는지/틀렸는지 검증하여 작성)*

---

## V. Related Work

### 사용한 라이브러리 및 도구
- **scikit-learn** — PCA, t-SNE, Isomap, Logistic Regression, Random Forest 구현체 ([공식 문서](https://scikit-learn.org/stable/))
- **umap-learn** — UMAP 구현체 ([공식 문서](https://umap-learn.readthedocs.io/))
- **XGBoost** — Gradient Boosting 구현체 ([공식 문서](https://xgboost.readthedocs.io/))
- **pandas / numpy / matplotlib / seaborn** — 데이터 처리 및 시각화

### 참고 문헌 / 자료
- Yeh, I-Cheng, and Lien, Che-hui. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications. (원본 데이터셋 논문)
- Van der Maaten, L., & Hinton, G. (2008). *Visualizing data using t-SNE.* Journal of Machine Learning Research.
- Tenenbaum, J. B., De Silva, V., & Langford, J. C. (2000). *A global geometric framework for nonlinear dimensionality reduction.* Science. (Isomap 원논문)
- McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.* arXiv:1802.03426
- scikit-learn user guide: [Manifold learning](https://scikit-learn.org/stable/modules/manifold.html)
- *(작성 중 추가로 참고한 블로그·캐글 노트북은 여기에 계속 추가)*

---

## VI. Conclusion

### 결론 (실험 후 채울 부분)

- 가장 좋은 조합은 [____ + ____] 이었으며, ROC-AUC 기준 [_.__]를 달성했다.
- 차원축소가 분류 성능에 미치는 영향은 [긍정적 / 부정적 / 분류기에 따라 다름] 이었다.
- 특히 [기법 이름]은 [관찰된 특성] 측면에서 흥미로운 결과를 보였다.
- 한계점: 2차원 임베딩은 정보 손실이 크므로 실제 산업 적용 시에는 더 높은 차원을 사용해야 함. t-SNE의 transform 부재로 인한 데이터 누수 우려 등.

### Discussion: 우리가 배운 것

- 차원축소 기법은 **목적에 따라 선택해야 한다** — 시각화에는 t-SNE/UMAP, 전처리에는 PCA가 일반적으로 적합.
- 같은 데이터셋이라도 기법에 따라 드러나는 구조가 다르다.
- 금융 데이터처럼 변수 간 의미가 강한 도메인에서는 차원축소가 해석가능성을 떨어뜨릴 수 있다.

### 역할 분담

- **이름1**: 데이터 전처리, EDA, t-SNE 실험
- **이름2**: PCA · Isomap 실험, 시각화
- **이름3**: UMAP 실험, 분류기 학습 및 평가
- **이름4**: 블로그 작성, 유튜브 영상 녹화, 발표
- *(실제 분담에 맞춰 수정)*
