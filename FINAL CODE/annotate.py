import pandas as pd

#Load generated pickup lines for evaluation
generated_lines = pd.read_csv("generated_lines.csv")

#Function to collect human annotations for coherence and relevance
def collect_annotations(generated_lines):
    annotations = []
    for index, row in generated_lines.iterrows():
        print(f"Pickup Line {index + 1}: {row['generated_line']}")
        coherence = input("Rate Coherence (1-5): ")
        relevance = input("Rate Relevance (1-5): ")
        row_data = {"Pickup Line": row['generated_line'], "Coherence": int(coherence), "Relevance": int(relevance)}
        annotations.append(row_data)
    return annotations

#Collect human annotations
annotations = collect_annotations(generated_lines)

#Convert annotations to a DataFrame
annotations_df = pd.DataFrame(annotations)

#Save annotations to a CSV file
annotations_df.to_csv("human_annotations.csv", index=False)
