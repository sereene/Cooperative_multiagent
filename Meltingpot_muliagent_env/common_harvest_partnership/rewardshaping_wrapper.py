from pettingzoo.utils.wrappers import BaseParallelWrapper

class RewardShapingWrapper(BaseParallelWrapper):
    def __init__(self, env, sharing_coeff=0.5):
        super().__init__(env)
        self.sharing_coeff = sharing_coeff

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        
        # [1] 보상 공유 로직 (기존 유지)
        if "player_0" in rewards and "player_1" in rewards:
            r0 = rewards["player_0"]
            r1 = rewards["player_1"]
            rewards["player_0"] += r1 * self.sharing_coeff
            rewards["player_1"] += r0 * self.sharing_coeff

        # [2] 사망(Zap) 감지 로직 초기화
        for agent_id in infos:
            infos[agent_id]["was_zapped"] = False

        # [3] 이벤트 파싱 및 타겟 변환 (여기가 핵심!)
        # zap_hit 이벤트는 보통 '쏜 사람(Source)'의 info에 들어있을 확률이 높으므로
        # 모든 에이전트의 info를 순회하며 이벤트를 찾습니다.
        for agent_id, agent_info in infos.items():
            if "events" in agent_info:
                for event in agent_info["events"]:
                    # Shimmy/MeltingPot 버전에 따라 event가 dict 또는 객체일 수 있음
                    if isinstance(event, dict):
                        name = event.get("name")
                        target = event.get("target")
                    else:
                        name = getattr(event, "name", None)
                        target = getattr(event, "target", None)

                    if name == "zap_hit":
                        # [중요] target은 Lua Index (1, 2, ...)로 들어옵니다.
                        # 이를 PettingZoo의 "player_0", "player_1" 형태로 변환해야 합니다.
                        try:
                            target_idx = int(target)  # 예: 1 -> player_0
                            target_agent_id = f"player_{target_idx - 1}"
                            
                            # 만약 타겟이 현재 infos에 존재한다면(죽은 사람이 player_0 or 1이라면) 플래그 설정
                            if target_agent_id in infos:
                                infos[target_agent_id]["was_zapped"] = True
                                # 디버깅용 (필요시 주석 해제)
                                # print(f"[DEBUG] Zap Detected! Target: {target} -> {target_agent_id}")
                                
                        except (ValueError, TypeError):
                            continue

        return obs, rewards, terminations, truncations, infos