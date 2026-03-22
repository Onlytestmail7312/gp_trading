import dill
from gp_individual import setup_gp_toolbox
from config import DAILY_FEATURES

setup_gp_toolbox(feature_names=DAILY_FEATURES)

with open('gp_output/best_model_v1_fitness17.pkl', 'rb') as f:
    model = dill.load(f)

print(f'Fitness  : {model["fitness"]:.4f}')
print(f'Nodes    : {model["tree_size"]}')
print(f'Formula  : {str(model["individual"])}')
