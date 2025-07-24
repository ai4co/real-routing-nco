#!/usr/bin/env python3
"""
LazyRMTVRPGenerator 테스트 스크립트
메모리 효율적인 생성기가 제대로 작동하는지 확인합니다.
"""

import os
import sys
import time
import psutil
import torch
from rl4co.utils.pylogger import get_pylogger

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rrnco.envs.rmtvrp import RMTVRPEnv
from rrnco.envs.rmtvrp.generator_lazy import LazyRMTVRPGenerator

log = get_pylogger(__name__)


def get_memory_usage():
    """현재 메모리 사용량 반환 (MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def test_lazy_generator():
    """LazyRMTVRPGenerator 테스트"""

    print("=== LazyRMTVRPGenerator 테스트 시작 ===")

    # 초기 메모리 사용량
    initial_memory = get_memory_usage()
    print(f"초기 메모리 사용량: {initial_memory:.2f} MB")

    try:
        # 1. 직접 LazyRMTVRPGenerator 사용
        print("\n1. 직접 LazyRMTVRPGenerator 테스트")
        start_time = time.time()

        generator = LazyRMTVRPGenerator(
            num_loc=50,  # 작은 크기로 테스트
            chunk_size=10,  # 작은 청크 크기
            data_path="../../../data/dataset",
            file_name="splited_cities_list",
        )

        # 배치 생성
        batch_size = [20]  # 작은 배치 크기
        td = generator(batch_size)

        end_time = time.time()
        memory_after_generation = get_memory_usage()

        print(f"생성 시간: {end_time - start_time:.2f}초")
        print(f"생성 후 메모리: {memory_after_generation:.2f} MB")
        print(f"메모리 증가: {memory_after_generation - initial_memory:.2f} MB")
        print(f"생성된 데이터 형태: {td.batch_size}")
        print(f"데이터 키들: {list(td.keys())}")

        # 2. 환경을 통한 테스트
        print("\n2. 환경을 통한 LazyRMTVRPGenerator 테스트")

        env = RMTVRPEnv(
            generator_params={
                "_target_": "rrnco.envs.rmtvrp.generator_lazy.LazyRMTVRPGenerator",
                "num_loc": 50,
                "chunk_size": 10,
                "data_path": "../../../data/dataset",
                "file_name": "splited_cities_list",
            }
        )

        # 환경 리셋 테스트
        td_env = env.generator(batch_size)
        td_reset = env.reset(td_env)

        print(f"환경 리셋 후 메모리: {get_memory_usage():.2f} MB")
        print(f"환경 리셋 데이터 형태: {td_reset.batch_size}")

        # 3. 메모리 정리 테스트
        print("\n3. 메모리 정리 테스트")

        if hasattr(generator, "clear_cache"):
            generator.clear_cache()
            print("캐시 정리 완료")

        memory_after_cleanup = get_memory_usage()
        print(f"정리 후 메모리: {memory_after_cleanup:.2f} MB")

        print("\n=== 테스트 완료 ===")
        print(f"최종 메모리 사용량: {memory_after_cleanup:.2f} MB")
        print(f"전체 메모리 증가: {memory_after_cleanup - initial_memory:.2f} MB")

        return True

    except Exception as e:
        print(f"테스트 실패: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_memory_efficiency():
    """메모리 효율성 테스트"""

    print("\n=== 메모리 효율성 테스트 ===")

    # 다양한 청크 크기로 테스트
    chunk_sizes = [10, 50, 100]
    batch_size = [100]

    for chunk_size in chunk_sizes:
        print(f"\n청크 크기: {chunk_size}")

        initial_memory = get_memory_usage()
        start_time = time.time()

        try:
            generator = LazyRMTVRPGenerator(
                num_loc=50,
                chunk_size=chunk_size,
                data_path="../../../data/dataset",
                file_name="splited_cities_list",
            )

            td = generator(batch_size)

            end_time = time.time()
            final_memory = get_memory_usage()

            print(f"  생성 시간: {end_time - start_time:.2f}초")
            print(f"  메모리 사용량: {final_memory - initial_memory:.2f} MB")
            print(f"  데이터 형태: {td.batch_size}")

            # 캐시 정리
            if hasattr(generator, "clear_cache"):
                generator.clear_cache()

        except Exception as e:
            print(f"  오류: {e}")


if __name__ == "__main__":
    print("LazyRMTVRPGenerator 테스트 시작")
    print(f"현재 작업 디렉토리: {os.getcwd()}")

    # 기본 테스트
    success = test_lazy_generator()

    if success:
        # 메모리 효율성 테스트
        test_memory_efficiency()

    print("\n테스트 완료!")
