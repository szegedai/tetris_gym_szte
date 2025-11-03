from gymnasium.wrappers import RecordVideo

def evaluate(env, model, ep_num=100):
    env_test = env

    sum_reward = 0

    for _ in range(ep_num): 
      
        obs, _ = env_test.reset()
        score = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, _ = env_test.step(action)

            score += reward

        sum_reward += score
        
    env_test.close()

    return sum_reward / ep_num

def evaluate_agent(env, agent, ep_num=100):
    env_test = env

    sum_reward = 0

    for _ in range(ep_num): 
      
        obs, _ = env_test.reset()
        score = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = agent.act(obs)

            obs, reward, terminated, truncated, _ = env_test.step(action)

            score += reward

        sum_reward += score
        
    env_test.close()

    return sum_reward / ep_num

def evaluate_agent_competition(max_task_num=400, seed=0, disable_progress_bar=True):
    ep_num_multiplier = int(max_task_num / 40)
    
    box_size_combinations = [
        (5, 1, 5),
        (5, 2, 5),
        (6, 1, 5),
        (6, 2, 5),
        (6, 3, 4),
        (7, 2, 4),
        (7, 3, 3),
        (8, 3, 2),
        (8, 4, 2),
        (9, 4, 2),
        (10, 4, 2),
        (10, 5, 1)
    ]

    # Környezet létrehozása
    correct_results = 0

    for size, num_boxes, num_episodes in tqdm(box_size_combinations):
        env = SokobanEnv(size=(size, size), padded_size=(10, 10), num_boxes=num_boxes, render_mode='rgb_array')

        agent = Agent(env)

        correct_results += evaluate_agent(env, agent, num_episodes*ep_num_multiplier, seed=seed, disable_progress_bar=disable_progress_bar)
    
    print()
    print(f"Result: {int(correct_results)} correct tasks out of {ep_num_multiplier*40}")
    print()

    return int(correct_results)

def create_videos(env, model, ep_num=2, folder="videos"):
    env = RecordVideo(env, folder, fps=4) 

    sum_reward = 0

    for _ in range(ep_num): 
      
        obs, _ = env.reset()
        score = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(action)

            score += reward

        sum_reward += score
        
    env.close()

    return sum_reward / ep_num
