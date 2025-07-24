# Neural Adaptive Bias (NAB) Ablation Study

본 가이드는 Gating Neural Adaptive Bias와 Naive Neural Adaptive Bias 간의 효과성을 검증하기 위한 ablation 실험 설정 방법을 설명합니다.

## 개요

두 가지 Neural Adaptive Bias (NAB) 방법이 구현되어 있습니다:

1. **GatingNeuralAdaptiveBias**: 복잡한 attention-based gating mechanism
   - FiLM modulation, learnable scaling, attention gates 사용
   - 더 많은 파라미터와 복잡한 계산

2. **NaiveNeuralAdaptiveBias**: 간단한 MLP-based approach
   - 각도, 거리, 지속시간 매트릭스를 concat하여 MLP로 처리
   - 더 적은 파라미터와 단순한 계산

## 구현된 파일들

### 수정된 파일들:
- `rrnco/models/nn/attn_freenet.py`: NAB type 선택 기능 추가
- `rrnco/models/encoder.py`: `nab_type` 파라미터 추가
- `rrnco/models/policy.py`: `nab_type` 파라미터 추가
- `configs/experiment/rrnet.yaml`: Gating NAB 사용 설정
- `configs/experiment/rrnet_naive.yaml`: Naive NAB 사용 설정 (새로 생성)

### 테스트 파일:
- `test_matrix_fusion.py`: 구현 검증용 테스트 스크립트

## 실험 실행 방법

### 1. 코드 검증 (선택사항)
```bash
python test_matrix_fusion.py
```

### 2. Gating NAB 실험 (기본 설정)
```bash
python -m rrnco experiment=rrnet
```

### 3. Naive NAB 실험 (ablation)
```bash
python -m rrnco experiment=rrnet_naive
```

### 4. 사용자 정의 설정
YAML 파일에서 `nab_type` 파라미터를 직접 설정할 수 있습니다:

```yaml
model:
  policy:
    nab_type: "gating"      # 또는 "naive"
```

명령줄에서 직접 오버라이드:
```bash
python -m rrnco experiment=rrnet model.policy.nab_type=naive
```

## 설정 옵션

`nab_type` 파라미터는 다음 값들을 지원합니다:
- `"gating"` (기본값): 복잡한 attention-based gating mechanism
- `"naive"`: 간단한 MLP-based approach

## 예상되는 차이점

### 모델 복잡도:
- **GatingNeuralAdaptiveBias**: 더 많은 파라미터 (FiLM layers, attention gates 등)
- **NaiveNeuralAdaptiveBias**: 더 적은 파라미터 (단순한 MLP)

### 계산 복잡도:
- **GatingNeuralAdaptiveBias**: 더 복잡한 계산 (attention mechanism, multiple transformations)
- **NaiveNeuralAdaptiveBias**: 더 빠른 계산 (단순한 concatenation + MLP)

### 성능:
- **GatingNeuralAdaptiveBias**: 더 정교한 feature fusion으로 잠재적으로 더 나은 성능
- **NaiveNeuralAdaptiveBias**: 단순하지만 효율적인 baseline

## 로깅 및 추적

WandB 로깅에서 다음과 같이 구분됩니다:
- Gating NAB 실험: `rrnet-{env_name}{num_loc}`
- Naive NAB 실험: `rrnet-naive-nab-{env_name}{num_loc}`

태그를 통해서도 구분 가능:
- Gating NAB: `["rrnet", "rcvrp"]`
- Naive NAB: `["rrnet", "naive_nab", "rcvrp"]`

## 주의사항

1. 두 실험을 동일한 하이퍼파라미터와 시드로 실행하여 공정한 비교가 되도록 하세요.
2. 충분한 에포크 수로 훈련하여 수렴을 확인하세요.
3. 여러 시드로 실험하여 결과의 안정성을 확인하는 것을 권장합니다.

## 기대 결과

이 ablation study를 통해 다음을 분석할 수 있습니다:
- 복잡한 gating mechanism의 실제 효과
- 계산 비용 대비 성능 향상
- Neural Adaptive Bias의 핵심 구성 요소 식별 