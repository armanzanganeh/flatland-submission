import time
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.timetable_generators import timetable_generator
from submission.my_observation_builder import MyObservationBuilder
from submission.my_policy import MyPolicy

def run_level_20_test():
    print("=======================================================")
    print("=== 💀 WELCOME TO LEVEL 20: THE APOCALYPSE TEST 💀 ===")
    print("=======================================================")
    
    # تنظیمات لول 20 (وحشتناک‌ترین حالت ممکن که ارور نده)
    WIDTH = 350               
    HEIGHT = 350              
    NUM_AGENTS = 1600         # 1600 قطار (بیشتر از این رم کم میاره)
    MAX_CITIES = 70           
    MAX_STEPS = 400          # 500 استپ پردازش
    
    print("WARNING: Generating a 500x500 map with 120 cities and 2500 trains...")
    print("This might take 1 to 2 minutes JUST to build the map. Please be patient!")
    
    try:
        env = RailEnv(
            width=WIDTH,
            height=HEIGHT,
            rail_generator=sparse_rail_generator(
                max_num_cities=MAX_CITIES,
                grid_mode=False,
                max_rails_between_cities=6, # ظرفیت ریل‌ها رو بردیم بالا تا کرش نکنه
                max_rail_pairs_in_city=6
            ),
            line_generator=sparse_line_generator(),
            timetable_generator=timetable_generator,
            number_of_agents=NUM_AGENTS,
            obs_builder_object=MyObservationBuilder()
        )
        
        # زمان‌گیری برای ساخت نقشه
        build_start = time.time()
        obs, info = env.reset(random_seed=42)
        build_end = time.time()
        
        print("-------------------------------------------------------")
        print(f"Map generated successfully in {round(build_end - build_start, 2)} seconds!")
        print(f"Simulating {NUM_AGENTS} trains on a {WIDTH}x{HEIGHT} grid for {MAX_STEPS} steps...")
        print("-------------------------------------------------------")
        
        start_time = time.time()
        
        for step in range(MAX_STEPS):
            actions = {}
            
            # گرفتن اکشن برای 2500 قطار
            for handle in env.get_agent_handles():
                agent_obs = obs.get(handle) if (obs and isinstance(obs, dict)) else None
                actions[handle] = policy.act(handle, None, env, agent_obs, info)
            
            # حرکت در محیط
            obs, rewards, dones, info = env.step(actions)
            
            if (step + 1) % 50 == 0:
                print(f" -> Step {step + 1} completed. Still surviving...")
                
            if dones['__all__']:
                print("\nINCREDIBLE! All 2500 trains reached their destinations!")
                break
                
        end_time = time.time()
        total_time = end_time - start_time
        
        print("=======================================================")
        print("=== 🏆 LEVEL 20 APOCALYPSE SURVIVED 🏆 ===")
        print(f"Total processing time for {MAX_STEPS} steps: {round(total_time, 4)} seconds!")
        print("=======================================================")
        
    except Exception as e:
        print("An error occurred. The map might be too massive for the RAM:")
        print(str(e))

if __name__ == '__main__':
    policy = MyPolicy()
    run_level_20_test()