from comet import download_model, load_from_checkpoint
import json

model_path = download_model("Unbabel/wmt22-cometkiwi-da")
model = load_from_checkpoint(model_path)

with open('translation_output.txt', 'r', encoding='utf-8') as f:
    data = json.load(f)

model_output = model.predict(data, batch_size=8, gpus=1)

with open('cometKiwi_results.txt', 'w', encoding='utf-8') as f:
    f.write("Raw Output:\n")
    f.write("=" * 50 + "\n")
    f.write(str(model_output))
    f.write("\n\n")
    
    f.write("Formatted Output:\n")
    f.write("=" * 50 + "\n")
    f.write(f"System Score: {model_output.system_score}\n")
    f.write(f"Individual Scores: {model_output.scores}")