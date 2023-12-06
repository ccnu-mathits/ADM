import os
import json
import numpy as np

def load_dataset(read_path,sqlength,min,max):
    print("Checking the dumps...")
    if not os.path.exists("./ProcessedData"):
        os.makedirs("./ProcessedData")
    save_path = "./ProcessedData/Lumosity"+str(sqlength)+"_"+str(min)+"_"+str(max)+".json"

    if os.access(save_path, os.F_OK):
        print("Lumosity dataset is existent!")
        path = save_path
        with open(path, "r") as f:
            all = json.load(f)
            users = all["users"]
            games = all["games"]
            squences = all["squences"]
            return users, games, squences
    else:
        users, games, squences = read_dataset(read_path,sqlength,min,max)
        #Save the processed dataset
        with open(save_path, "w") as f:
            json.dump({"users": users, "games": games, "squences": squences}, f)
            f.close()
            print("The processed data is dumped in:")
            print(save_path)
        return users, games, squences

def read_dataset(read_path,sqlength,min,max):
    # Read and renumber the userinfo
    f_u = open(read_path+"/data_user.csv").readlines()
    user_reindex = {}
    index = 0
    user_heads = f_u.pop(0).strip("\n").split(",")
    user_heads.pop(0)
    user_info = []
    for line in f_u:
        data = line.strip("\n").split(",")
        uid = data.pop(0)
        # Education_level: fill the null value with the average value (4.0)
        if data[4] == '0.0':
            data[4] = '4.0'
        if user_reindex.__contains__(uid):
            print("The info of the user",uid,"is repetitive!!!")
        else:
            user_reindex[uid] = index
            index += 1
        uinfo = [float(i) for i in data]
        user_info.append(uinfo)
    user_num = index
    users = {}
    users["user_num"] = user_num
    users["user_heads"] = user_heads
    users["user_info"] = user_info
    print(user_num,"users' info are loaded...")
    ################################
    # Read and renumber the gameinfo
    f_g = open(read_path + "/master_game_info_one_hot.csv").readlines()
    game_reindex = {}
    index = 0
    game_heads = f_g.pop(0).strip("\n").split(",")
    game_heads.pop(0)
    game_info = []
    for line in f_g:
        data = line.strip("\n").split(",")
        gid = data.pop(0)
        if game_reindex.__contains__(gid):
            print("The info of the game", gid, "is repetitive!!!")
        else:
            game_reindex[gid] = index
            index += 1
        ginfo = [float(i) for i in data]
        game_info.append(ginfo)
    game_num = index
    games = {}
    games["game_num"] = game_num
    games["game_heads"] = game_heads
    games["game_features"] = game_info
    print(game_num, "games' info are loaded...")
    ################################
    # Read the game logs
    f_log = open(read_path + "/nlog_0.9_"+str(sqlength)+".csv").readlines()
    log_heads = f_log.pop(0).strip("\n").split(",")
    squences = {}
    log_num = f_log.__len__()
    for line in f_log:
        data = line.strip("\n").split(",")
        uid = user_reindex[data[0]]
        if not squences.__contains__(uid):
            squences[uid] = {}
            squences[uid]["q_squences"] = []
            squences[uid]["level_squences"] = []
            squences[uid]["score_squences"] = []
        q_id = game_reindex[data[1]]
        level = float(data[2])
        score = float(data[3])*(max-min)+min
        squences[uid]["q_squences"].append(q_id)
        squences[uid]["level_squences"].append(level)
        squences[uid]["score_squences"].append(score)
    print(log_num, "logs are loaded...")

    ################################
    # Generate the context info
    f_G_info = open(read_path + "/master_game_info.csv").readlines()
    f_G_info.pop(0)
    g_area = {}
    g_attribute = {}

    for line in f_G_info:
        data = line.strip("\n").split(",")
        g_id = data[0]
        area = data[1]
        attribute = data[2]
        g_area[game_reindex[g_id]] = area
        g_attribute[game_reindex[g_id]] = attribute
    context = {}
    for uid in squences:
        context[uid] = {}
        sq_g = squences[uid]["q_squences"]
        sq_area = [g_area[i] for i in sq_g]
        sq_attribute = [g_attribute[i] for i in sq_g]
        game_count = get_count(sq_g)
        area_count = get_count(sq_area)
        attribute_count = get_count(sq_attribute)

        game_interval = get_interval(sq_g)
        area_interval = get_interval(sq_area)
        attribute_interval = get_interval(sq_attribute)

        squences[uid]['game_count'] = game_count
        squences[uid]['area_count'] = area_count
        squences[uid]['attribute_count'] = attribute_count

        squences[uid]['game_interval'] = game_interval
        squences[uid]['area_interval'] = area_interval
        squences[uid]['attribute_interval'] = attribute_interval
    print("Context information is generated...")

    return users, games, squences

def get_count(sq):
    count_index = {}
    count = []
    for i in sq:
        if count_index.__contains__(i):
            count_index[i] += 1
        else:
            count_index[i] = 0
        count.append(count_index[i])
    return count

def get_interval(sq):
    place_index = {}
    count = 0
    interval = []
    for i in sq:
        count += 1
        if place_index.__contains__(i):
            interval.append(count-place_index[i])
        else:
            interval.append(0)
        place_index[i] = count
    return interval



if __name__ == '__main__':
    users, games, squences = load_dataset('../Dataset/Lumosity/processed_data',5000,0.01,0.8)

