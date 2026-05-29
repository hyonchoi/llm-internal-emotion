# LLM Internal Emotion Trajectory Extractor — Design Spec

**Date**: 2026-05-28  
**Based on**: "Emotion Concepts and their Function in a Large Language Model" (Sofroniew et al., Anthropic, arXiv:2604.07729v1)  
**Target model family**: Mistral / Mixtral (HuggingFace open-source)

---

## 1. Goal

LLM(대형 언어 모델)이 대화를 처리할 때 내부적으로 어떤 감정 상태를 거치는지를 추출하고 시각화하는 파이프라인을 구축한다.

**핵심 질문**: "모델이 이 문장을 처리할 때 내부적으로 얼마나 '두려움', '기쁨', '불안'을 나타내고 있는가?"

### 비전문가를 위한 한 줄 설명
> 인간이 말을 할 때 뇌 속에서 감정 관련 영역이 활성화되듯, LLM도 각 단어를 처리할 때 내부 신경망의 특정 방향이 활성화된다. 이 코드는 그 활성화 패턴을 측정하여 모델이 대화 중 어떤 감정 궤적을 따라가는지 지도처럼 그려준다.

---

## 2. 논문 핵심 방법론 요약 (비전문가용)

### 2.1 Emotion Vector란?

논문은 LLM 내부의 수천 차원 공간에서 각 감정이 **하나의 방향(벡터)**으로 표현된다는 것을 발견했다. 이 방향을 찾는 방법:

1. "그는 매우 행복했다"처럼 특정 감정이 들어간 짧은 이야기를 여러 편 생성한다
2. 동일한 상황이지만 감정이 없는(중립적인) 이야기를 생성한다
3. 모델이 각 이야기를 처리할 때 특정 레이어에서 나오는 **내부 활성화값(hidden state)**을 추출한다
4. (감정 있는 이야기의 평균 활성화) − (중립 이야기의 평균 활성화) = **감정 벡터**

이 벡터는 그 감정의 "방향"을 나타낸다.

### 2.2 Trajectory(궤적)란?

새로운 대화가 주어졌을 때, 모델이 각 단어(토큰)를 처리하는 순간의 내부 상태를 감정 벡터에 **투영(dot product)**하면, 그 순간의 감정 강도를 수치로 얻을 수 있다. 이 수치를 대화 전체에 걸쳐 시각화하면 **감정 궤적**이 된다.

### 2.3 논문의 주요 발견

- 모델은 각 토큰을 처리하는 **현재 맥락**에서 operative한 감정을 표현한다
- `Human:` 턴과 `Assistant:` 턴에서 서로 다른 감정이 활성화된다
- Assistant의 `:` 토큰(응답 시작 직전)은 곧 생성될 응답의 감정 톤을 예측한다
- 레이어에 따라 감정 표현이 다름: 초반 레이어는 현재 토큰의 감정적 함축, 중후반 레이어는 다음에 생성될 내용의 감정

---

## 3. 아키텍처

```
llm-internal-emotion/
├── emotion_tracer/
│   ├── __init__.py
│   ├── config.py            # 모델, 레이어, 감정 목록 설정
│   ├── model_wrapper.py     # HuggingFace hook으로 hidden states 추출
│   ├── story_generator.py   # 감정별 합성 스토리 생성
│   ├── emotion_vectors.py   # Approach 1: mean-diff / Approach 2: logistic probe
│   ├── trajectory.py        # 대화 토큰별 emotion projection 추출
│   └── visualizer.py        # heatmap, trajectory plot, JSON/CSV 저장
├── scripts/
│   ├── build_vectors.py     # 감정 벡터 사전 계산 (1회 실행)
│   └── trace.py             # 대화 입력 → trajectory 출력
├── data/
│   └── emotion_vectors/     # 계산된 벡터 캐시 (.npz)
├── docs/
│   ├── superpowers/specs/   # 이 설계 문서
│   └── explanation/         # 비전문가용 설명 문서
├── examples/
│   └── example_conversation.py
├── tests/
│   └── test_emotion_vectors.py
├── pyproject.toml
└── README.md
```

---

## 4. 컴포넌트 상세 설계

### 4.1 `config.py`

모든 설정을 한 곳에서 관리한다.

```python
@dataclass
class EmotionTracerConfig:
    model_name: str           # "mistralai/Mistral-7B-Instruct-v0.3"
    layer_indices: list[int]  # 추출할 레이어 인덱스 (기본: 모델 깊이의 2/3 지점)
    emotions: list[str]       # 추적할 감정 목록
    stories_per_emotion: int  # 감정 벡터 계산용 스토리 수 (기본: 20)
    device: str               # "cuda" / "cpu" / "mps"
    cache_dir: str            # 벡터 캐시 저장 경로
```

**기본 감정 목록** (논문의 130개 중 핵심 20개):
`happy, sad, angry, afraid, calm, anxious, loving, hopeful, frustrated, curious, disgusted, surprised, guilty, proud, lonely, grateful, excited, bored, confused, nervous`

사용자는 논문의 전체 130개로 확장하거나 커스텀 감정 단어를 추가할 수 있다.

### 4.2 `model_wrapper.py`

PyTorch의 `register_forward_hook`을 사용해 특정 레이어의 residual stream(hidden state)을 추출한다.

```
입력 텍스트 → tokenize → forward pass (hook 등록) → 레이어 N의 hidden states 캡처
```

- 각 토큰 위치 `t`에서 레이어 `l`의 hidden state: shape `[seq_len, hidden_dim]`
- 메모리 효율을 위해 지정된 레이어만 캡처

### 4.3 `story_generator.py`

감정 벡터 계산에 필요한 합성 데이터셋을 생성한다.

**생성 전략**:
- **감정 있는 스토리**: `"Write a short story (3-4 sentences) where the main character feels {emotion}."` 프롬프트로 생성
- **중립 스토리**: 동일한 설정이지만 감정 묘사 없음 (`"Write a short neutral story (3-4 sentences) about a person's daily routine."`)

이미 생성된 데이터셋이 있으면 재사용한다(캐시).

### 4.4 `emotion_vectors.py`

두 가지 방법으로 emotion vector를 계산한다:

**Method 1: Mean Difference (기본, 빠름)**
```
emotion_vector[e] = mean(h_positive[e]) - mean(h_neutral)
emotion_vector[e] = L2_normalize(emotion_vector[e])
```
여기서 `h_positive[e]`는 감정 `e`가 포함된 스토리들의 hidden states.

**Method 2: Logistic Regression Probe (더 정확)**
- 감정별로 이진 분류기 학습: "이 hidden state가 감정 e를 나타내는가?"
- 학습 데이터: 5가지 시나리오 (자연스러운 표현 / 숨긴 감정 / 미표현 등) - 논문의 Section 2.2.4 방식
- 분류기의 coefficient 벡터 = emotion vector

두 결과 모두 `.npz` 파일로 캐시에 저장한다.

### 4.5 `trajectory.py`

새로운 대화에 대해 토큰별 emotion projection을 계산한다.

```
for each token position t:
    h_t = hidden_state[t]           # shape: [hidden_dim]
    for each emotion e:
        score[e, t] = dot(h_t, emotion_vector[e])
```

출력: `scores` — shape `[num_emotions, seq_len]`의 numpy array

### 4.6 `visualizer.py`

두 가지 시각화를 생성한다:

**Plot 1: Emotion Heatmap**
- X축: 토큰 위치 (원문 텍스트 레이블 포함)
- Y축: 감정 종류
- 색상: 활성화 강도 (빨강=강한 활성화, 파랑=억제)
- Human 턴과 Assistant 턴을 배경색으로 구분

**Plot 2: Top-K Emotion Timeline**
- 가장 강하게 활성화된 K개 감정의 라인 그래프
- 대화의 흐름에 따른 감정 강도 변화를 한눈에 파악

**저장 형식**:
- PNG 이미지 (heatmap, timeline)
- JSON: `{token: str, position: int, emotion_scores: {emotion: float, ...}}` per token
- CSV: 행=토큰, 열=감정

---

## 5. 데이터 흐름

```
[1단계: 벡터 구축] (1회 실행, ~10-30분)
  story_generator.py
      └─> model_wrapper.py (hidden states 추출)
          └─> emotion_vectors.py (mean-diff + logistic probe)
              └─> data/emotion_vectors/*.npz (캐시)

[2단계: 궤적 추출] (반복 실행, ~수 초)
  trace.py (대화 텍스트 입력)
      └─> model_wrapper.py (token별 hidden states)
          └─> trajectory.py (emotion projection)
              └─> visualizer.py
                  ├─> output/heatmap.png
                  ├─> output/timeline.png
                  └─> output/trajectory.json + .csv
```

---

## 6. 사용 예시 (CLI)

```bash
# 1단계: 감정 벡터 사전 계산
python scripts/build_vectors.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --emotions happy sad angry afraid calm \
    --method both

# 2단계: 대화 trajectory 추출
python scripts/trace.py \
    --input "Human: I just lost my job. Assistant: I'm so sorry to hear that..." \
    --emotions happy sad anxious calm \
    --output results/
```

---

## 7. 문서화 전략

비전문가도 이해할 수 있는 문서를 두 계층으로 제공한다:

### Layer 1: README.md (프로젝트 최상위)
- "이 프로젝트가 무엇을 하는가" — 비유와 그림 위주
- 설치 및 빠른 시작 가이드
- 결과 해석 방법 ("빨간색이 많다는 것은 무엇을 의미하는가")

### Layer 2: docs/explanation/ (심화 설명)
- `what_are_emotion_vectors.md`: 감정 벡터란 무엇인가 (수식 없이)
- `how_to_read_the_plots.md`: 시각화 결과 읽는 법
- `paper_summary.md`: 논문 핵심 요약 (비전문가용)
- `limitations.md`: 이 접근법의 한계 (논문 Section 5.1 기반)

---

## 8. 의존성

```toml
[dependencies]
torch = ">=2.0"
transformers = ">=4.40"
scikit-learn = ">=1.3"    # LogisticRegression probe
numpy = ">=1.24"
matplotlib = ">=3.7"
seaborn = ">=0.12"        # heatmap 스타일링
pandas = ">=2.0"          # CSV 저장
tqdm = ">=4.65"           # 진행 표시
```

GPU 없이도 실행 가능하나, 7B 모델 기준 CPU에서 속도가 매우 느림.  
Apple Silicon(MPS) 지원 포함.

---

## 9. 범위 외 (Out of Scope)

- **Activation Steering**: 감정 벡터로 모델 행동을 조종하는 기능 (논문 Part 2) — 별도 프로젝트
- **실시간 스트리밍**: 생성 중 실시간 trajectory 업데이트
- **Claude/GPT 등 API 전용 모델**: 내부 hidden states 접근 불가
- **멀티모달 입력**: 텍스트 전용

---

## 10. 테스트 계획

| 테스트 | 검증 내용 |
|--------|-----------|
| `test_model_wrapper` | hook이 올바른 shape의 hidden state를 반환하는가 |
| `test_emotion_vectors` | mean-diff 벡터가 L2 정규화되어 있는가 |
| `test_probe` | logistic probe의 in-distribution 정확도 > 70% |
| `test_trajectory` | 명백히 슬픈 텍스트에서 sad 감정이 상위권인가 |
| `test_visualizer` | PNG/JSON/CSV 파일이 올바르게 생성되는가 |

---

## 11. 한계 (비전문가용)

이 도구가 **보여주지 못하는** 것들:

1. **LLM이 실제로 감정을 '느끼는지'**: 이 숫자들은 패턴이지 주관적 경험이 아니다
2. **대화 전체에 걸친 지속적 감정 상태**: 모델은 각 토큰마다 맥락에서 감정을 새로 읽는다 (뇌처럼 감정이 유지되지 않음)
3. **Claude 등 상용 모델**: API로는 내부 상태에 접근 불가
4. **정답**: 어떤 감정이 '진짜'인지 확인할 외부 기준이 없다

이 도구는 **패턴 탐색 도구**이지 감정 측정기가 아니다.
