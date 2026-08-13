"""
main.py
=======
실행 진입점.

사용 예:
    python main.py                 # 기본 GUI 실행
    python main.py --headless -t 120   # 화면 없이 120초 실행 후 로그 저장
    python main.py -n 1200         # 닭 수 변경
    python main.py --habituation   # habituation ON 으로 시작
"""

import argparse
import config as C


def parse_args():
    p = argparse.ArgumentParser(description="2D 육계 군집–로봇 상호작용 시뮬레이터")
    p.add_argument("-n", "--count", type=int, default=None,
                   help="닭 수 (default: config.CHICKEN_COUNT)")
    p.add_argument("-t", "--time", type=float, default=None,
                   help="최대 시뮬레이션 시간(s). GUI 에서는 이 시간에 자동 종료")
    p.add_argument("--headless", action="store_true",
                   help="렌더링 없이 실행(로그만 저장)")
    p.add_argument("--habituation", action="store_true",
                   help="habituation ON 으로 시작")
    p.add_argument("--no-robot", action="store_true",
                   help="로봇 비활성화 (BASE 군집만)")
    p.add_argument("--seed", type=int, default=None, help="랜덤 시드")
    p.add_argument("--scale", type=float, default=None,
                   help="PX_PER_M (화면 스케일)")
    return p.parse_args()


def apply_overrides(args):
    if args.count is not None:
        C.CHICKEN_COUNT = args.count
    if args.habituation:
        C.HABITUATION_ON = True
    if args.no_robot:
        C.ROBOT_ENABLED = False
    if args.seed is not None:
        C.RANDOM_SEED = args.seed
    if args.scale is not None:
        C.PX_PER_M = args.scale


def main():
    args = parse_args()
    apply_overrides(args)

    # config 반영 후에 import (모듈들이 import 시점에 config 값을 읽는 경우 대비)
    from simulation import Simulation

    sim = Simulation(headless=args.headless)
    sim.run(max_time=args.time)


if __name__ == "__main__":
    main()
