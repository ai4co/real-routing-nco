#!/usr/bin/env python3
"""
설정 테스트 스크립트
rcvrptw.yaml 설정이 올바르게 작동하는지 확인합니다.
"""

import os
import sys
import hydra
from omegaconf import DictConfig

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


@hydra.main(version_base="1.3", config_path="configs", config_name="main.yaml")
def test_config(cfg: DictConfig):
    """설정 테스트"""

    print("=== 설정 테스트 시작 ===")

    try:
        # 환경 인스턴스화
        print("환경 인스턴스화 중...")
        env = hydra.utils.instantiate(cfg.env)

        print(f"환경 타입: {type(env)}")
        print(f"환경 이름: {env.name}")
        print(f"생성기 타입: {type(env.generator)}")

        # 생성기 정보 출력
        if hasattr(env.generator, "chunk_size"):
            print(f"청킹 크기: {env.generator.chunk_size}")
        if hasattr(env.generator, "num_loc"):
            print(f"노드 수: {env.generator.num_loc}")

        # 간단한 데이터 생성 테스트
        print("\n데이터 생성 테스트...")
        batch_size = [4]  # 작은 배치 크기
        td = env.generator(batch_size)

        print(f"생성된 데이터 형태: {td.batch_size}")
        print(f"데이터 키들: {list(td.keys())}")

        # 환경 리셋 테스트
        print("\n환경 리셋 테스트...")
        td_reset = env.reset(td)
        print(f"리셋 후 데이터 형태: {td_reset.batch_size}")

        print("\n=== 테스트 성공! ===")
        return True

    except Exception as e:
        print(f"테스트 실패: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_config()
    if success:
        print("설정이 올바르게 작동합니다!")
    else:
        print("설정에 문제가 있습니다.")
        sys.exit(1)
