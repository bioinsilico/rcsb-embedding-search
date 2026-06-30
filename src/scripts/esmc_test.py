import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from huggingface_hub import login


if __name__ == "__main__":
    # login with your Hugging Face credentials
    # login()

    # example GFP sequence
    sequences = ["MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"]

    model_path = "biohub/ESMC-600M"
    model = AutoModelForMaskedLM.from_pretrained(
        model_path,
        device_map="cpu",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    inputs = tokenizer(sequences, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        output = model(**inputs)
        print(output.last_hidden_state.shape)