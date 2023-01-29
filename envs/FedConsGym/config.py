import math

CONFIG = {
    'CartPole':{
        'Pos2':{
            "constraints":[
                ["pos", [[-2.4, -2.3], [-1.3, -1.2], [-0.1, 0.0], [1.1, 1.2], [2.2, 2.3]]],
                ["pos", [[-2.3, -2.2], [-1.2, -1.1], [0.0, 0.1], [1.2, 1.3], [2.3, 2.4]]],
            ],
            "budget":[20.0, 20.0]
        },
    },
    "Acrobot":{
        'Emptyleftbottom':{
            "constraints":[
                ["theta1", "leftbottom", 0],
                ["theta1", "leftbottom", 2]
            ],
            "budget": [10.0, 10.0]
        },
        'EasierEmptyleftbottom':{
            "constraints":[
                ["theta1", "leftbottom", 0],
                ["theta1", "leftbottom", 2]
            ],
            "budget": [40.0, 40.0]
        },
        'VelAngle2':{
            "constraints":[
                ["vel_theta1", "neg", 2],
                ["vel_theta2", "neg", 2]
            ],
            "budget": [50.0, 50.0]
        },
        'theta1-sin':{
            "constraints":[
                ["vel_theta1", "neg", 2],
                ["sin_theta1", "neg"]
            ],
            "budget": [30.0, 30.0]     
        },
        'VelTheta2':{
            "constraints":[
                ["vel_theta1", "neg", 2],
                ["vel_theta1", "pos", 0]
            ],
            "budget": [30.0, 30.0]  
        },
        'Ban+1':{
            "constraints":[
                ["cos_theta1", "pos", 2],
            ],
            "budget": [50.0]
        },
        'Ban+1-2':{
            "constraints":[
                ["theta1", "leftbottom", 2],
                ["theta1", "rightbottom", 2],
            ],
            "budget": [25.0, 25.0]
        },
        'BanVel2':{
            "constraints":[
                ["theta1", "leftbottom", 2],
                ["vel_theta1", "pos", 2]
            ],
            "budget": [25.0, 25.0]
        },
        "Theta1-2":{
            "constraints":[
                ["vel_theta1", "neg", 2],
                ["vel_theta1", "pos", 0]
            ],
            "budget": [50.0, 50.0]
        },
        'RevVelAngle2':{
            "constraints":[
                ["vel_theta1", "neg", 2],
                ["vel_theta2", "pos", 0]
            ],
            "budget": [50.0, 50.0]
        },
    },
    "InvertedPendulum":{
        'Pos2':{
            "constraints":[
                ["pos", [[-2.4, -2.0], [-1.3, -0.7], [-0.1, 0.0]]],
                ["pos", [[0.0, 0.1], [0.7, 1.3], [2.0, 2.4]]],
            ],
            "budget":[20.0, 20.0]
        },
        "Adj2":{
            "constraints":[
                ["pos", [[-2.4, -2.0], [-1.3, -0.8], [-0.2, 0.2], [0.8, 1.3], [2.0, 2.4]]],
                ["pos", [[-1.9, -1.3], [-0.5, -0.2], [0.2, 0.5], [1.3, 1.9]]],
            ],
            "budget":[20.0, 20.0]    
        },
        "ADJ2":{
            "constraints":[
                ["pos", [[-2.4, -1.7], [-0.6, -0.2], [0.0, 0.2], [0.8, 1.5]]],
                ["pos", [[-1.5, -0.8], [-0.2, 0.0], [0.2, 0.6], [1.7, 2.4]]],
            ],
            "budget":[50.0, 50.0] 
        },
        "ADJ3":{
            "constraints":[
                ["pos", [[-2.4, -1.7], [-0.2, 0.0], [0.8, 1.2]]],
                ["pos", [[-1.6, -1.2], [-0.6, -0.2], [0.2, 0.6], [1.2, 1.6]]],
                ["pos", [[-1.2, -0.8], [0, 0.2], [1.7, 2.4]]]
            ],
            "budget":[15.0, 15.0, 15.0] 
        },
        'CP2':{
            "constraints":[
                ["pos", [[-2.4, -2.3], [-1.3, -1.2], [-0.1, 0.0], [1.1, 1.2], [2.2, 2.3]]],
                ["pos", [[-2.3, -2.2], [-1.2, -1.1], [0.0, 0.1], [1.2, 1.3], [2.3, 2.4]]],
            ],
            "budget":[20.0, 20.0]
        },
        'FarStop2':{
            "constraints":[
                ["pos", [[0.0, 0.1], [0.4, 1.5], [1.6, 2.3]]],
                ["pos", [[-2.3, -1.6], [-1.5, -0.4], [-0.1, 0.0]]],
            ],
            "budget":[20.0, 20.0]
        },
        'Mod2':{
            "constraints":[
                ["pos", [[-0.1, 0.1]]],
                ["pos", "mod", 0.5, 2, 1],
            ],
            "budget":[20.0, 20.0]
        },
    }
}

STR_CONSTRAINT = {
    "neg": [[-100.0, 0.0]],
    "pos": [[0.0, 100.0]],
    "leftbottom": [[-0.5*math.pi, 0.0]],
    "rightbottom": [[0.0, 0.5*math.pi]]
}

def load_config(env_name, config_name):
    if env_name not in CONFIG.keys():
        raise NotImplementedError
    if config_name not in CONFIG[env_name].keys():
        raise NotImplementedError
    return CONFIG[env_name][config_name]
