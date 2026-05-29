# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

LLM 내부 감정 궤적(emotion trajectory) 추출 파이프라인. Mistral/Mixtral 계열 HuggingFace 모델에서 residual stream hidden states를 추출하여 감정 벡터를 계산하고, 대화 토큰별 감정 활성화를 시각화한다.

**논문**: "Emotion Concepts and their Function in a Large Language Model" (Sofroniew et al., Anthropic, arXiv:2604.07729v1)  
**설계 스펙**: `docs/superpowers/specs/2026-05-28-emotion-trajectory-design.md`

## Commands

프로젝트 설정 후 사용할 주요 명령어 (구현 완료 시점 기준):

```bash
# 의존성 설치
uv sync

# 1단계: 감정 벡터 사전 계산 (1회 실행, GPU 권장)
python scripts/build_vectors.py --model mistralai/Mistral-7B-Instruct-v0.3 --emotions happy sad angry afraid calm --method both

# 2단계: 대화 trajectory 추출
python scripts/trace.py --input "conversation.txt" --emotions happy sad anxious calm --output results/

# 테스트 실행
pytest tests/ -v

# 단일 테스트
pytest tests/test_emotion_vectors.py::test_probe -v
```

## Architecture

**2단계 파이프라인**:

```
[build_vectors.py]  →  story_generator → model_wrapper → emotion_vectors → data/emotion_vectors/*.npz
[trace.py]          →  model_wrapper → trajectory → visualizer → PNG + JSON + CSV
```

**`emotion_tracer/` 모듈 역할**:

- `config.py` — `EmotionTracerConfig` dataclass. 모든 하이퍼파라미터의 단일 진입점 (모델명, 레이어 인덱스, 감정 목록, 캐시 경로)
- `model_wrapper.py` — PyTorch `register_forward_hook`으로 지정 레이어의 hidden states를 캡처. 지정 레이어만 메모리에 보관
- `story_generator.py` — 감정별 합성 스토리(emotion-present) + 중립 스토리(neutral) 생성. 캐시 존재 시 재사용
- `emotion_vectors.py` — **Method 1**: `mean(h_positive) - mean(h_neutral)` L2 정규화. **Method 2**: sklearn `LogisticRegression` one-vs-rest probe. 결과는 `.npz`로 캐시
- `trajectory.py` — 각 토큰 위치 `t`에서 `dot(h_t, emotion_vector[e])`를 계산. 출력: `[num_emotions, seq_len]` numpy array
- `visualizer.py` — Heatmap (X: 토큰, Y: 감정, 색: 강도) + Top-K timeline 라인그래프. PNG/JSON/CSV 저장

**레이어 선택 원칙**: 모델 깊이의 약 2/3 지점 ("mid-late" layer). 초반 레이어는 현재 토큰의 표면적 감정 함축, 중후반 레이어는 다음 토큰 생성에 operative한 감정을 인코딩한다 (논문 Section 2.2.3).

**감정 벡터 계산 방식 선택**:
- `mean-diff` — 빠르고 단순, 레이블이 깨끗한 스토리 데이터에 적합
- `logistic-probe` — 더 정확하나 다양한 시나리오(자연 표현/숨긴 감정/미표현) 데이터 필요. 논문 Section 2.2.4 방식

## Key Design Decisions

- `data/emotion_vectors/` 캐시를 항상 확인하고, 있으면 재계산하지 않는다
- `Human:` 턴 토큰과 `Assistant:` 턴 토큰을 별도 구간으로 처리한다 (시각화에서 배경색 구분)
- `Assistant:` 턴 시작의 `:` 토큰은 특별히 마킹한다 — 이 토큰의 emotion projection이 응답 감정 톤을 예측한다
- Activation steering(감정 벡터로 모델 행동 조종)은 이 프로젝트 범위 밖

## Documentation

비전문가 설명 문서는 `docs/explanation/`에 위치한다:
- `what_are_emotion_vectors.md` — 수식 없이 감정 벡터 개념 설명
- `how_to_read_the_plots.md` — 시각화 결과 해석법
- `paper_summary.md` — 논문 핵심 요약
- `limitations.md` — 이 접근법의 한계
